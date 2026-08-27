"""VRoom 백엔드 서버 — 오케스트레이션 계층.

이 파일은 '누가 누구에게 무엇을 언제 보내는가'만 담당한다.
면접 로직은 session.py, 개입 판정은 bargein.py, 대사 생성은 llm.py 가 맡는다.

[데이터 흐름]
    Unity  --(마이크 오디오 WS)-->  STT 워커  --(전사 텍스트 WS)-->  [이 서버]
    Unity  <--(제어 WS: JSON 패킷)--  [이 서버]  --(대사)-->  TTS 워커
    Unity  <--(음성 PCM)--  STT 워커  <--(음성 릴레이)--  [이 서버]

[음성이 STT 워커를 경유하는 이유]
    STT 워커가 자기가 내보낸 면접관 음성을 알고 있어야 에코를 걸러낼 수 있다.
    이 때문에 오디오는 제어 채널보다 한 홉 늦게 도착한다 —
    '재생 완료' 판정에 제어 채널의 audio_end 를 쓰면 안 되는 이유가 이것이다.

[엔드포인트]
    WS   /ws/control        Unity 제어 채널 (양방향)
    WS   /ws/tts            STT 워커 채널 (전사 수신 + 음성 릴레이)
    POST /process           STT 워커 HTTP 경로 (레거시, 현재 미사용)
    POST /session/prepare   씬 진입 전 첫 질문 선합성 (프리웜)
    POST /vision            Vision 워커의 턴별 웹캠 피쳐 수신
    GET  /health            상태 확인
"""
from __future__ import annotations

import asyncio
import json
import random
import time

import websockets
from websockets.protocol import State
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import bargein, llm, session_log, tts_client
from .config import BargeInConfig, settings
from .domain import (
    AnswerRequest,
    BehaviorPacket,
    ExpressionID,
    GestureID,
    LLMTurn,
)
from .session import InterviewSession

app = FastAPI(title="VRoom Backend", version="1.0")


# ===========================================================================
# 1. 세션 / 소켓 레지스트리
# ===========================================================================
class Hub:
    """모든 세션과 소켓을 한곳에 모아 두는 레지스트리.

    소켓이 세 종류라 헷갈리기 쉬우므로 이름으로 구분한다.
      sockets      : Unity 제어 채널   (백엔드가 서버)
      stt_sockets  : STT 워커 채널     (백엔드가 서버)
      tts_sockets  : TTS 워커 연결     (백엔드가 클라이언트)
    """

    def __init__(self):
        self.sessions: dict[str, InterviewSession] = {}      # sid -> 면접 세션 상태머신
        self.sockets: dict[str, WebSocket] = {}              # sid -> Unity 제어 소켓
        self.stt_sockets: dict[str, WebSocket] = {}          # sid -> STT 워커 소켓
        self.tts_sockets: dict[str, websockets.WebSocketClientProtocol] = {}
                                                             # sid -> TTS 워커 영속 연결
        self.prepared: dict[str, dict] = {}                  # sid -> 프리웜 캐시 {packet, chunks}
        self.last_active: str | None = None                  # 세션 ID 없이 온 요청의 폴백
        self.lock = asyncio.Lock()                           # (예약) 동시성 보호용

    # ── 수명 관리 ───────────────────────────────────────────────────
    async def register(self, sid: str, ws: WebSocket):
        """Unity 제어 소켓을 등록하고 TTS 워커 연결을 준비한다."""
        self.sockets[sid] = ws
        self.last_active = sid

        # 프리웜에서 이미 열어둔 연결이 살아 있으면 그대로 재사용한다.
        existing = self.tts_sockets.get(sid)
        if existing is not None and existing.state != State.CLOSED:
            print(f"[{sid}] 기존 TTS Worker 연결 재사용 (프리웜)")
            return

        try:
            self.tts_sockets[sid] = await websockets.connect(settings.tts_ws_url, max_size=None)
            print(f"[{sid}] TTS Worker 영속 연결 수립")
        except Exception as e:
            self.tts_sockets[sid] = None
            print(f"[{sid}] [경고] TTS Worker 연결 실패: {e}")

    async def unregister(self, sid: str):
        """세션 관련 소켓과 캐시를 모두 정리한다."""
        self.sockets.pop(sid, None)
        self.stt_sockets.pop(sid, None)
        self.prepared.pop(sid, None)

        tts_ws = self.tts_sockets.pop(sid, None)
        if tts_ws:
            try:
                await tts_ws.close()
                print(f"[{sid}] TTS Worker 연결 종료")
            except Exception as e:
                print(f"[{sid}] TTS Worker 연결 종료 실패: {e}")

    async def get_or_connect_tts_ws(self, sid: str):
        """TTS 워커 연결을 돌려준다. 끊겨 있으면 그 자리에서 다시 연결한다."""
        tts_ws = self.tts_sockets.get(sid)
        if tts_ws is None or tts_ws.state == State.CLOSED:
            print(f"[{sid}] TTS WebSocket 끊김 - 재연결 시도")
            try:
                tts_ws = await websockets.connect(settings.tts_ws_url, max_size=None)
                self.tts_sockets[sid] = tts_ws
                print(f"[{sid}] TTS Worker 재연결 성공")
            except Exception as e:
                self.tts_sockets[sid] = None
                print(f"[{sid}] TTS Worker 재연결 실패: {e}")
                return None
        return tts_ws

    # ── Unity 로 보내기 (실패하면 세션을 정리한다) ──────────────────
    async def send_packet(self, sid: str, packet: BehaviorPacket):
        """행동 지시 패킷을 Unity 제어 채널로 보낸다."""
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_text(packet.model_dump_json())
            except Exception:
                await self.unregister(sid)

    async def send_json(self, sid: str, obj: dict):
        """임의 JSON 메시지를 Unity 제어 채널로 보낸다 (컷인 명령, audio_end 등)."""
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_text(json.dumps(obj, ensure_ascii=False))
            except Exception:
                await self.unregister(sid)

    async def send_audio(self, sid: str, chunk: bytes):
        """PCM 청크를 Unity 제어 채널로 직접 보낸다 (STT 워커가 없을 때의 폴백)."""
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_bytes(chunk)
            except Exception:
                await self.unregister(sid)


hub = Hub()   # 프로세스 전역 단일 인스턴스


# ===========================================================================
# 2. 발화 송출 (공통 경로)
# ===========================================================================
async def _wait_stt_socket(sid: str, tries: int = 30):
    """STT 워커 소켓이 붙을 때까지 최대 3초(0.1s × 30) 기다린다. 없으면 None."""
    for _ in range(tries):
        ws = hub.stt_sockets.get(sid)
        if ws:
            return ws
        await asyncio.sleep(0.1)
    return None


async def _relay_chunk(sid: str, stt_ws, chunk):
    """TTS 청크 하나를 목적지로 보낸다.

    STT 워커가 있으면 그쪽으로(정상 경로), 없으면 Unity 로 직접 보낸다(폴백).
    청크는 PCM(bytes) 이거나 자막 JSON(str) 둘 중 하나다.
    """
    if stt_ws:
        try:
            if isinstance(chunk, bytes):
                await stt_ws.send_bytes(chunk)
            else:
                await stt_ws.send_text(chunk)
        except Exception as e:
            print(f"[{sid}] STT 소켓 릴레이 실패: {e}")
        return

    if isinstance(chunk, bytes):
        await hub.send_audio(sid, chunk)
    else:
        ws_ctrl = hub.sockets.get(sid)
        if ws_ctrl:
            try:
                await ws_ctrl.send_text(chunk)
            except Exception:
                pass


async def _signal_utterance_end(sid: str, stt_ws):
    """한 발화가 끝났음을 두 채널 모두에 알린다.

    Unity 는 audio_end 를, STT 워커는 {"type":"end"} 를 기대한다.
    주의: Unity 의 '재생 완료' 판정은 audio_end 가 아니라 STT 경유 tts_end 로 해야 한다.
          audio_end 는 제어 채널 직행이라 오디오보다 먼저 도착한다.
    """
    await hub.send_json(sid, {"type": "audio_end"})
    if stt_ws:
        try:
            await stt_ws.send_json({"type": "end"})
        except Exception:
            pass


async def speak(sid: str, packet: BehaviorPacket, on_first_chunk=None):
    """행동 패킷 push -> TTS 합성 -> 음성 스트리밍 -> 종료 신호.

    on_first_chunk: 첫 '오디오' 청크가 나가는 순간 1회 호출되는 콜백.
        speak() 는 TTS 스트림을 끝까지 소비한 뒤 반환하므로 함수 전체 소요시간은
        '합성 완료까지'이지 '첫 발성까지'가 아니다. 개입 지연(설계서 9.6)은
        첫 발성 기준이어야 하므로 이 콜백으로 따로 잰다.
    """
    # (1) 제스처/표정/대사를 먼저 보낸다. 몸짓이 소리보다 앞서야 자연스럽다.
    await hub.send_packet(sid, packet)

    # (2) TTS 생략 모드 — Node B 없이 상태 전이만 테스트할 때.
    if settings.skip_tts:
        await _signal_utterance_end(sid, hub.stt_sockets.get(sid))
        return

    # (3) 음성 합성 후 릴레이.
    stt_ws = await _wait_stt_socket(sid)
    tts_ws = await hub.get_or_connect_tts_ws(sid)

    if tts_ws:
        try:
            async for chunk in tts_client.synthesize_ws_stream(tts_ws, packet.dialogue):
                # 자막 JSON(str)도 섞여 오므로 bytes 일 때만 '첫 발성'으로 센다.
                if on_first_chunk is not None and isinstance(chunk, bytes):
                    on_first_chunk()
                    on_first_chunk = None
                await _relay_chunk(sid, stt_ws, chunk)
        except Exception as e:
            print(f"[{sid}] TTS 릴레이 중 에러 - 음성 생략: {e}")
    else:
        print(f"[{sid}] TTS 소켓 유실 - 음성 생략")

    # (4) 발화 종료 신호.
    await _signal_utterance_end(sid, stt_ws)


async def speak_prepared(sid: str) -> bool:
    """프리웜 때 미리 합성해 둔 첫 발화를 즉시 흘려보낸다.

    TTS 합성 대기가 0이므로 씬 진입 직후 바로 말하기 시작한다.
    캐시가 없으면 False 를 돌려주고, 호출자가 일반 경로로 폴백한다.
    """
    data = hub.prepared.pop(sid, None)
    if not data:
        return False

    packet: BehaviorPacket = data["packet"]
    chunks: list = data["chunks"]

    await hub.send_packet(sid, packet)
    print(f"[{sid}] [프리웜 재생] {packet.dialogue} (chunks={len(chunks)})")

    # 음성 캐시가 없으면 대사만 보내고 끝낸다.
    if settings.skip_tts or not chunks:
        await _signal_utterance_end(sid, hub.stt_sockets.get(sid))
        return True

    stt_ws = await _wait_stt_socket(sid)
    for chunk in chunks:
        try:
            await _relay_chunk(sid, stt_ws, chunk)
        except Exception as e:
            print(f"[{sid}] 프리웜 청크 전송 실패: {e}")
            break
        await asyncio.sleep(0)      # 이벤트 루프에 양보 (한 프레임에 몰아치지 않게)

    await _signal_utterance_end(sid, stt_ws)
    return True


async def _send_thinking(sid: str, session: InterviewSession):
    """'생각 중' 더미 모션을 즉시 띄워 LLM 대기 시간을 가린다.

    단, 개입 직후에는 보내지 않는다. 방금 말을 끊은 면접관이 이력서를
    검토하는 모션을 취하면 개입 연출이 무너진다.
    """
    if session.pending_cutoff is not None or session.awaiting_reanswer:
        return

    await hub.send_packet(sid, BehaviorPacket(
        type="thinking", session_id=sid, stage=session.stage.value,
        persona=session.persona.value, persona_value=session.last_persona_value,
        dialogue="", expression_id=ExpressionID.THINKING.value,
        gesture_id=GestureID.REVIEW_RESUME.value, score=-1,
    ))


# ===========================================================================
# 3. 개입(끼어들기) 오케스트레이션
# ===========================================================================
# Type B 개입 대사 템플릿.
#
# LLM 을 쓰지 않는 이유: Type B 는 백엔드에 답변 텍스트가 없어서(Unity 가 시간
# 값만 보냄) 대사에 인용할 내용이 애초에 없다. LLM 은 같은 뜻을 매번 다르게
# 쓰는 일만 하고 대가로 1초 지연과 실패 가능성을 얻는다.
# 세션당 최대 3회이므로 reason 별 3개면 반복이 느껴지지 않는다.
# 효과: 첫 발성까지 약 1.7초 -> 0.7초(TTS 합성만).
_INTERVENTION_TEMPLATES = {
    "LONG_ANSWER": [
        "네, 거기까지 듣겠습니다. 답변이 너무 길어지고 있습니다.",
        "잠시 끊겠습니다. 요점만 간결하게 말씀해 주셔야 합니다.",
        "여기까지 하겠습니다. 답변 시간 관리도 평가 대상입니다.",
    ],
    "LONG_SILENCE": [
        "답변이 어려우신 것 같군요. 이 질문은 여기까지 하고 넘어가겠습니다.",
        "시간이 지체되고 있습니다. 다음 질문으로 진행하도록 하겠습니다.",
        "네, 준비가 안 되신 것 같습니다. 아쉽지만 다음으로 넘어가겠습니다.",
    ],
}


async def handle_bargein_signal(sid: str, session: InterviewSession,
                                reason: str, elapsed: float):
    """Type B 진입점. Unity 로컬 타이머 신호를 받아 게이팅하고 개입을 시작한다.

    Type B 는 발화를 둘로 쪼갠다.
      발화 1 = 개입 대사 (템플릿, 여기서 시작)
      발화 2 = 다음 질문 (잘린 전사 도착 후 on_user_answer 가 생성)
    발화 1 재생 시간이 발화 2 생성 지연을 덮어 자연스럽게 이어진다.
    """
    decision = await bargein.evaluate_signal(session, reason, elapsed)
    if not decision.granted:
        return      # 거부는 Unity 에 알리지 않는다. 아무 일 없던 것처럼 계속 답변을 받는다.

    session.commit_bargein(decision)

    # (1) 컷인 반사 명령 — 즉시 발송. 실시간 왕복은 이것뿐이다.
    #     컷인 오디오 프리셋이 없어도 반드시 보낸다.
    #     Unity 의 상태 전이 / 발화 강제 확정 / 표정 변경이 여기에 달려 있다.
    await hub.send_json(sid, bargein.build_cutin_message(session, decision))

    # (2) 발화 2가 발화 1을 앞지르지 못하게 막는 게이트를 연다.
    session.bargein_speech_done = asyncio.Event()

    # (3) 발화 1 생성·발송은 별도 태스크로.
    #     여기서 await 하면 /ws/control 수신 루프가 막혀 다른 메시지를 못 받는다.
    asyncio.create_task(_speak_intervention(sid, session, decision))

    # (4) 워치독 — STT 가 빈 전사를 반환하면 텍스트가 영영 오지 않는다.
    asyncio.create_task(_cutoff_watchdog(sid, session))


async def handle_partial_transcript(sid: str, session: InterviewSession,
                                    cumulative: str) -> None:
    """Type A 진입점. 발화 도중의 누적 전사를 받아 주제 이탈 여부를 판정한다.
 
    Type B(handle_bargein_signal)와 달리 Unity 가 아니라 STT 워커가 보낸다.
    Unity 는 길이/침묵만 재고, 의미 판정은 백엔드가 한다.
 
    호출 빈도가 높으므로(발화 중 1~2초 간격) 거부 로그는 게이트가 바뀔 때만
    찍는다. 매번 찍으면 콘솔이 부분 전사로 뒤덮여 다른 로그를 못 본다.
    """
    if not cumulative:
        return
 
    # 이미 개입해 재답변을 기다리는 중이면 다시 판정하지 않는다.
    if session.awaiting_reanswer:
        return
 
    decision = await bargein.evaluate_partial(session, cumulative)
 
    if not decision.granted:
        last = getattr(session, "_last_partial_deny", None)
        if decision.denied_by != last:
            session._last_partial_deny = decision.denied_by
            print(f"[개입] REDIRECT 거부 ({decision.denied_by}) "
                  f"stage={session.stage.value} persona={session.persona.value} "
                  f"chars={len(cumulative.replace(' ', ''))}")
        return
 
    session._last_partial_deny = None
    session.commit_bargein(decision)
 
    # (1) 컷인 반사 명령 — 대사 생성을 기다리지 않고 먼저 보낸다.
    #     Unity 의 상태 전이(UserAnswering -> BargeInPending)와 발화 강제 확정,
    #     표정 변경이 전부 이 메시지에 달려 있다.
    await hub.send_json(sid, bargein.build_cutin_message(session, decision))
 
    # (2) 개입 대사는 별도 태스크로. 여기서 await 하면 수신 루프가 막힌다.
    #     Type B 와 달리 워치독은 걸지 않는다. Type A 는 단계를 전진시키지
    #     않으므로 전사가 늦게 와도 세션이 멈추지 않는다.
    asyncio.create_task(_speak_redirect(sid, session, decision))
 
 
async def _speak_redirect(sid: str, session: InterviewSession, decision) -> None:
    """Type A 개입 대사를 생성해 발송한다.
 
    Type B 는 템플릿(_INTERVENTION_TEMPLATES)으로 처리하지만, Type A 는
    "무엇에서 벗어났는지"를 인용해야 하므로 LLM(L2)이 필요하다.
    """
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    question = session.current_question_text
    try:
        turn = await llm.generate_intervention(
            stage=session.stage,
            persona=session.persona,
            bargein_type=decision.bargein_type,
            reason=decision.reason,
            question=question,
            partial_answer=decision.meta.get("partial_text", ""),
            info=session.info,
        )
 
        # 개입 자체는 채점 이벤트가 아니고, 비언어는 게이트 상수로 고정한다.
        turn.score = -1
        turn.expression_id = ExpressionID.FIRM_STOP.value
        turn.gesture_id = GestureID.ARMS_CROSSED.value
 
        # update_question=False 가 중요하다. 여기서 덮어쓰면 잘린 답변 채점과
        # 재답변 채점이 원래 질문이 아니라 개입 대사를 기준으로 돌아간다.
        session._record("interviewer", turn.dialogue, update_question=False)
        packet = session._to_packet(turn, is_final=False,
                                    bargein_type=decision.bargein_type)
 
        def _mark_first_chunk():
            ms = int((loop.time() - t0) * 1000)
            session.note_speech_latency(ms)
            print(f"[개입] REDIRECT 첫 발성까지 {ms}ms")
 
        await speak(sid, packet, on_first_chunk=_mark_first_chunk)
 
        full_ms = int((loop.time() - t0) * 1000)
        if session.bargein_log:
            session.bargein_log[-1]["latency_full_ms"] = full_ms
        print(f"[개입] REDIRECT 대사 완료 (총 {full_ms}ms) \"{turn.dialogue}\"")
 
    except Exception as e:
        # 대사 생성이 실패해도 Unity 는 이미 컷인을 받아 개입 상태로 전이했다.
        # 아무 말도 안 하면 면접관이 노려보기만 하므로 폴백 대사라도 내보낸다.
        print(f"[개입] REDIRECT 대사 실패: {e}")
        try:
            fallback = LLMTurn(
                dialogue="잠시만요. 제가 여쭌 질문으로 다시 돌아가 주시겠습니까?",
                expression_id=ExpressionID.FIRM_STOP.value,
                gesture_id=GestureID.ARMS_CROSSED.value,
                score=-1,
            )
            session._record("interviewer", fallback.dialogue, update_question=False)
            await speak(sid, session._to_packet(
                fallback, is_final=False, bargein_type=decision.bargein_type))
        except Exception as e2:
            print(f"[개입] REDIRECT 폴백도 실패: {e2}")
 

async def _speak_intervention(sid: str, session: InterviewSession, decision):
    """발화 1 — 개입 대사를 템플릿에서 골라 즉시 발송한다.

    단계 전진도, 채점도 하지 않는다. 그건 발화 2(on_user_answer)의 몫이다.
    """
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    try:
        pool = _INTERVENTION_TEMPLATES.get(decision.reason)
        if not pool:
            # Type A(OFF_TOPIC)는 답변 내용을 인용해야 해서 LLM 이 필요하다.
            # 이 함수는 Type B 전용이다.
            print(f"[개입] 템플릿 없는 reason={decision.reason} - 발화1 생략")
            return

        dialogue = _pick_intervention_line(session, pool)
        turn = LLMTurn(
            dialogue=dialogue,
            expression_id=ExpressionID.FIRM_STOP.value,
            gesture_id=GestureID.ARMS_CROSSED.value,   # 제지 제스처는 리소스 미확보
            score=-1,                                  # 개입 자체는 채점 이벤트가 아니다
        )

        session._record("interviewer", turn.dialogue, update_question=False)
        packet = session._to_packet(turn, is_final=False,
                                    bargein_type=decision.bargein_type)

        def _mark_first_chunk():
            """첫 발성 시점을 논문 지표로 기록한다."""
            ms = int((loop.time() - t0) * 1000)
            session.note_speech_latency(ms)
            print(f"[개입] 발화1 첫 발성까지 {ms}ms")

        await speak(sid, packet, on_first_chunk=_mark_first_chunk)

        full_ms = int((loop.time() - t0) * 1000)
        if session.bargein_log:
            session.bargein_log[-1]["latency_full_ms"] = full_ms
        print(f"[개입] 발화1 합성 완료 (총 {full_ms}ms) \"{dialogue}\"")

    except Exception as e:
        print(f"[개입] 발화1 실패: {e}")
    finally:
        # 성공이든 실패든 반드시 게이트를 연다. 안 열면 다음 질문이 영구 대기한다.
        session.bargein_speech_done.set()


def _pick_intervention_line(session: InterviewSession, pool: list[str]) -> str:
    """같은 문구가 연속으로 나오지 않게 최근 사용분을 제외하고 고른다."""
    used = getattr(session, "_used_intervention_lines", set())
    candidates = [d for d in pool if d not in used] or pool
    picked = random.choice(candidates)
    used.add(picked)
    session._used_intervention_lines = used
    return picked


async def _cutoff_watchdog(sid: str, session: InterviewSession):
    """잘린 전사가 끝내 오지 않을 때 세션을 강제로 진행시킨다.

    STT 가 빈 전사(stt_skip)를 반환하면 /ws/tts 로 텍스트가 영영 오지 않아
    면접이 멈춘 것처럼 보인다.
    """
    await asyncio.sleep(BargeInConfig.FINAL_WAIT_TIMEOUT)
    if session.pending_cutoff is None:
        return      # 정상적으로 전사가 도착해 소비되었다

    print(f"[{sid}] [개입] 잘린 전사 미도착 - 워치독으로 강제 진행")
    packet = await session.on_user_answer("(답변이 중단되어 전사되지 않음)", {})
    if packet.type != "ignored":
        await _speak_after_bargein(sid, session, packet)


async def _await_bargein_speech(sid: str, session: InterviewSession):
    """발화 1 발송이 끝날 때까지 기다린다. 개입이 없었으면 즉시 통과한다.

    왜 락이 아니라 이벤트인가:
      락은 '동시에 쓰지 못한다'만 보장하고 '누가 먼저'는 보장하지 않는다.
      발화 2가 발화 1보다 먼저 나가면 대화 순서가 뒤집히고,
      두 발화가 같은 STT 소켓에 동시에 PCM 을 흘리면 오디오가 뒤섞인다.
    """
    ev = getattr(session, "bargein_speech_done", None)
    if ev is None:
        return

    if not ev.is_set():
        try:
            await asyncio.wait_for(ev.wait(), timeout=BargeInConfig.FINAL_WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            print(f"[{sid}] [개입] 발화1 완료 대기 타임아웃 - 그대로 진행")

    session.bargein_speech_done = None   # 게이트 소비 (다음 턴에 영향 없게)


async def _speak_after_bargein(sid: str, session: InterviewSession, packet: BehaviorPacket):
    """speak() 를 쓰는 경로용 래퍼. 워치독과 /process 에서 사용한다.

    /ws/tts 는 speak() 대신 자체 릴레이를 하므로 거기서는
    _await_bargein_speech() 만 직접 호출한다.
    """
    await _await_bargein_speech(sid, session)
    await speak(sid, packet)


# ===========================================================================
# 4. 엔드포인트 — Unity 제어 채널
# ===========================================================================
@app.websocket("/ws/control")
async def ws_control(ws: WebSocket):
    """Unity 와의 양방향 채널. 수신 메시지 종류별로 핸들러에 위임한다."""
    sid: str | None = None
    try:
        await ws.accept()
        while True:
            msg = json.loads(await ws.receive_text())
            mtype = msg.get("type")

            if mtype == "init":
                sid = await _on_init(ws, msg)

            elif mtype == "utterance_started":
                _on_utterance_started(sid)

            elif mtype == "utterance_end":
                _on_utterance_end(sid, msg)

            elif mtype == "bargein_signal":
                await _on_bargein_signal(sid, msg)

            elif mtype == "bargein_yield":
                _on_bargein_yield(sid, msg)

            elif mtype == "request_feedback":
                await _on_request_feedback(sid)

    except (WebSocketDisconnect, RuntimeError) as e:
        print(f"[ws_control] 연결 종료 ({sid}): {e}")
    except Exception as e:
        print(f"[ws_control] 예기치 않은 에러 ({sid}): {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sid:
            # 결과 화면까지 못 가고 끊긴 세션도 원자료는 남긴다.
            # 이미 정상 저장됐으면 log_written 플래그로 무시된다.
            s = hub.sessions.get(sid)
            if s is not None:
                session_log.dump(s, None, exit_reason="disconnect")
            await hub.unregister(sid)


async def _on_init(ws: WebSocket, msg: dict) -> str:
    """세션을 만들거나 프리웜 세션을 인계받고, 첫 질문을 발화한다."""
    sid = msg.get("session_id") or "default"
    prepared = hub.prepared.get(sid)

    if prepared is None:
        hub.sessions[sid] = InterviewSession(
            session_id=sid,
            company=msg.get("company", ""),
            job_title=msg.get("job_title", ""),
            resume=msg.get("resume", ""),
            condition=msg.get("condition", "C"),
        )
    else:
        s = hub.sessions.get(sid)
        if s is not None:
            # 프리웜 세션은 prepare 시점에 만들어져 조건을 모른다. 여기서 채운다.
            s.condition = msg.get("condition", "C")
            # prepare 시점부터 셋업 화면 대기 시간까지 포함되므로,
            # 개입 시각(triggered_at)의 기준을 '면접 시작'으로 맞춘다.
            s.session_started_at = time.time()

    # 실험 유효성 헤더. 두 경로 모두에서 찍히도록 if/else 밖에 둔다.
    _s = hub.sessions.get(sid)
    print(f"[{sid}] 세션 시작 — condition={_s.condition if _s else '?'}, "
          f"prewarmed={prepared is not None}, "
          f"force_negative={settings.bargein_force_negative}")

    await hub.register(sid, ws)

    if prepared is not None:
        await speak_prepared(sid)
    else:
        packet = await hub.sessions[sid].first_question()
        await speak(sid, packet)

    return sid


def _on_utterance_started(sid: str | None):
    """개입 유예(G6) 계산의 기준 시각을 찍는다. Unity VAD 발화 시작 시 1회 온다."""
    if sid and sid in hub.sessions:
        hub.sessions[sid].utterance_started_at = time.time()


def _on_utterance_end(sid: str | None, msg: dict):
    """(사용 안 함) Unity 는 음성 피쳐를 STT 워커 경유로 보낸다.

    여기서 _collect_features 를 부르면 turn_stages 없이 리스트만 늘어나
    인덱스 정합이 깨진다. 레거시 호환이 필요해지면 turn_stages.append 를
    함께 넣을 것.
    """
    return


async def _on_bargein_signal(sid: str | None, msg: dict):
    """Unity 로컬 타이머가 잡은 개입 트리거(LONG_ANSWER / LONG_SILENCE)."""
    if sid and sid in hub.sessions:
        await handle_bargein_signal(
            sid, hub.sessions[sid],
            msg.get("reason", "LONG_ANSWER"),
            float(msg.get("elapsed", 0.0)),
        )


def _on_bargein_yield(sid: str | None, msg: dict):
    """개입 후 사용자가 입을 다물기까지의 시간. LONG_ANSWER 개입에서만 온다."""
    if sid and sid in hub.sessions:
        yield_time = float(msg.get("yield_time", 0.0))
        hub.sessions[sid].note_yield_time(yield_time)
        print(f"[{sid}] 양보 시간 {yield_time}s")


async def _on_request_feedback(sid: str | None):
    """면접 종료 후 최종 리포트를 만들어 Unity 결과 화면으로 보낸다."""
    print(f"[ws_control] request_feedback 수신 (sid={sid})")
    if not (sid and sid in hub.sessions):
        return

    session = hub.sessions[sid]
    report = None
    try:
        report = await session.build_feedback()
        await hub.send_json(sid, report.model_dump())
        print(f"[ws_control] feedback_report 전송 완료 (sid={sid})")
    except Exception as e:
        print(f"[ws_control] build_feedback 에러: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 리포트 생성에 실패해도 원자료는 남긴다. 실험 데이터가 더 중요하다.
        session_log.dump(session, report, exit_reason="normal")

# ===========================================================================
# 5. 엔드포인트 — STT 워커 채널
# ===========================================================================
@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    """STT 워커가 사용자 답변 텍스트를 보내는 입구이자, 면접관 음성이 나가는 출구.

    STT 워커 입장에서는 기존 TTS 서버와 동일하게 보인다.
    (음성 청크를 받고 마지막에 {"type":"end"} 를 받는다.)
    """
    sid = ws.query_params.get("session_id", "default")
    try:
        await ws.accept()
        hub.stt_sockets[sid] = ws
        print(f"[/ws/tts] STT 워커 연결됨 - Session ID: {sid}")

        while True:
            msg = json.loads(await ws.receive_text())
            mtype = msg.get("type", "")

            # ── Type A 부분 전사 ───────────────────────────────────────
            # STT 워커(V1)가 발화 도중 누적 전사를 주기적으로 보낸다.
            # 이 분기가 없으면 아래 `if not user_text` 에서 통째로 버려진다.
            if mtype == "partial_transcript":
                p_sid = msg.get("session_id") or sid or hub.last_active or "default"
                p_session = hub.sessions.get(p_sid)
                if p_session is not None:
                    await handle_partial_transcript(
                        p_sid, p_session, msg.get("cumulative", ""))
                continue

            user_text = msg.get("text", "")
            if not user_text:
                continue

            msg_sid = msg.get("session_id") or sid or hub.last_active or "default"
            session = hub.sessions.get(msg_sid)
            if session is None:
                print(f"[/ws/tts] 활성 세션 없음 ({msg_sid}) - Unity init 이 먼저 필요")
                await ws.send_json({"type": "end"})
                continue

            print(f"[STT→백엔드 수신 ({msg_sid})] {user_text}")
            await _send_thinking(msg_sid, session)

            # (1) 채점 + 다음 질문 생성. 개입 중이면 여기가 발화 2에 해당한다.
            packet = await session.on_user_answer(user_text, msg.get("features", {}))
            if packet.type == "ignored":
                await ws.send_json({"type": "end"})
                continue

            print(f"[백엔드→TTS 대사] {packet.dialogue} "
                  f"(stage={packet.stage}, persona={packet.persona}, score={packet.score})")

            # (2) 발화 1이 아직 나가는 중이면 끝날 때까지 기다린다.
            #     LLM 생성은 이 앞에서 이미 끝났으므로, 생성은 발화 1 재생과
            #     병렬로 진행되어 지연이 자연스럽게 가려진다.
            await _await_bargein_speech(msg_sid, session)

            # (3) 자막 + 행동 패킷을 Unity 로.
            await hub.send_packet(msg_sid, packet)

            # (4) 대사를 TTS 로 합성해 음성/자막을 STT 워커로 릴레이.
            tts_ws = await hub.get_or_connect_tts_ws(msg_sid)
            if tts_ws:
                try:
                    async for chunk in tts_client.synthesize_ws_stream(tts_ws, packet.dialogue):
                        if isinstance(chunk, bytes):
                            await ws.send_bytes(chunk)
                        else:
                            await ws.send_text(chunk)
                except Exception as e:
                    print(f"[/ws/tts] [{msg_sid}] TTS 릴레이 실패: {e}")
            else:
                print(f"[/ws/tts] [{msg_sid}] TTS 소켓 유실로 릴레이 생략")

            # (5) 발화 끝 신호. STT 워커가 이걸 받고 Unity VAD 잠금을 푼다.
            await ws.send_json({"type": "end"})

    except (WebSocketDisconnect, RuntimeError):
        print(f"[/ws/tts] STT 워커 연결 종료 - Session ID: {sid}")
    except Exception as e:
        print(f"[/ws/tts] 에러 (Session ID {sid}): {e}")
    finally:
        hub.stt_sockets.pop(sid, None)


@app.post("/process")
async def process(req: AnswerRequest):
    """STT 워커의 HTTP 경로 (레거시). 현재 STT 워커는 /ws/tts 를 쓴다.

    proxy_audio_to_stt 가 True 면 음성을 HTTP 응답으로 되돌려주는 호환 모드로 동작한다.
    """
    sid = req.session_id or hub.last_active or "default"
    session = hub.sessions.get(sid)
    if session is None:
        return JSONResponse(
            {"error": "no active session. Unity must send 'init' first."}, status_code=409)

    print(f"[STT→백엔드 수신] {req.text}")
    await _send_thinking(sid, session)

    packet = await session.on_user_answer(req.text, req.features)
    if packet.type == "ignored":
        stt_ws = hub.stt_sockets.get(sid)
        if stt_ws:
            try:
                await stt_ws.send_json({"type": "end"})
            except Exception:
                pass
        return {"ok": True, "status": "ignored"}

    print(f"[백엔드→TTS 대사] {packet.dialogue} "
          f"(stage={packet.stage}, persona={packet.persona}, score={packet.score})")

    # 호환 모드: STT 워커가 음성을 되받길 기대하는 경우.
    if settings.proxy_audio_to_stt:
        await hub.send_packet(sid, packet)

        async def audio_gen():
            async for chunk in tts_client.synthesize_stream(packet.dialogue):
                yield chunk

        return StreamingResponse(audio_gen(), media_type="application/octet-stream")

    # 기본 모드: 행동 패킷과 음성을 모두 백엔드 -> Unity WS 로 직접 보낸다.
    await _speak_after_bargein(sid, session, packet)
    return {"ok": True, "stage": packet.stage,
            "persona": packet.persona, "score": packet.score}


# ===========================================================================
# 6. 엔드포인트 — 세션 준비 / 상태
# ===========================================================================
class PrepareRequest(BaseModel):
    """Setup 씬에서 면접 씬으로 넘어가기 전에 보내는 프리웜 요청."""
    session_id: str = "default"
    company: str = ""
    job_title: str = ""
    resume: str = ""


@app.post("/session/prepare")
async def session_prepare(req: PrepareRequest):
    """첫 질문을 미리 만들고 음성까지 합성해 캐시한다.

    씬 진입 직후 면접관이 곧바로 말하기 시작하도록 만드는 것이 목적이다.
    실험 조건은 이 시점에 모르므로 init 에서 나중에 채워 넣는다.
    """
    sid = req.session_id or "default"

    # Setup 재진입에 대비해 이전 잔여 세션/캐시를 지운다.
    hub.prepared.pop(sid, None)
    hub.sessions.pop(sid, None)

    session = InterviewSession(
        session_id=sid, company=req.company,
        job_title=req.job_title, resume=req.resume,
    )
    hub.sessions[sid] = session
    hub.last_active = sid

    # (1) 첫 질문 생성 — 템플릿이면 지연 0.
    if settings.template_first_question:
        packet = session.template_first_question()
    else:
        packet = await session.first_question()

    # (2) 음성 선합성. 실패해도 런타임 합성으로 폴백되므로 치명적이지 않다.
    chunks: list = []
    if not settings.skip_tts:
        tts_ws = await hub.get_or_connect_tts_ws(sid)
        if tts_ws:
            try:
                async for chunk in tts_client.synthesize_ws_stream(tts_ws, packet.dialogue):
                    chunks.append(chunk)
            except Exception as e:
                print(f"[{sid}] [prepare] TTS 선합성 실패(런타임 합성으로 폴백): {e}")
                chunks = []
        else:
            print(f"[{sid}] [prepare] TTS 소켓 없음 - 음성 캐시 생략")

    hub.prepared[sid] = {"packet": packet, "chunks": chunks}

    # (3) LLM 커넥션 예열. 응답을 막지 않도록 백그라운드로 돌린다.
    if settings.warmup_llm_on_prepare:
        asyncio.create_task(llm.warmup())

    audio_bytes = sum(len(c) for c in chunks if isinstance(c, bytes))
    print(f"[{sid}] [prepare] 완료 - dialogue='{packet.dialogue[:30]}...' "
          f"chunks={len(chunks)} bytes={audio_bytes}")

    return {
        "ok": True,
        "session_id": sid,
        "dialogue": packet.dialogue,
        "audio_chunks": len(chunks),
        "audio_bytes": audio_bytes,
    }


@app.get("/health")
async def health():
    """기동 확인용. Unity Setup 씬이 백엔드 가용성을 판단하는 데 쓴다."""
    return {"status": "ok", "provider": settings.llm_provider,
            "active_sessions": len(hub.sessions)}


# ===========================================================================
# 7. 엔드포인트 — Vision 워커
# ===========================================================================
class VisionRequest(BaseModel):
    """Vision 워커가 턴 종료 시 보내는 웹캠 피쳐 묶음."""
    session_id: str = ""
    features: dict = {}


@app.post("/vision")
async def vision(req: VisionRequest):
    """턴 단위 웹캠 피쳐를 세션에 적재한다. 채점은 면접 종료 시 한 번에 한다."""
    sid = req.session_id or hub.last_active or "default"
    session = hub.sessions.get(sid)
    if session is None:
        return JSONResponse({"error": "no active session"}, status_code=409)

    session.collect_vision_features(req.features)
    print(f"[Vision→백엔드] stage={req.features.get('stage')} "
          f"phase={req.features.get('phase', 'NORMAL')} "
          f"frames={req.features.get('frameCount')} "
          f"dur={req.features.get('durationSec')}s "
          f"gaze={req.features.get('gazeOnTargetRatio')} "
          f"sway={req.features.get('bodySwayStd')}")
    return {"ok": True, "collected": len(session.vision_turns)}