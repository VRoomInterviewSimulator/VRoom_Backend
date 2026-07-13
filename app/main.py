"""
VRoom 백엔드 서버 (착수보고서 4.2.4 백엔드 서버 및 통신)

역할:
  - Unity 와 WebSocket 양방향 통신(/ws/control): 행동 지시 패킷(JSON 텍스트 프레임)
    과 면접관 음성(바이너리 PCM 프레임)을 같은 채널로 내려보낸다.
  - STT 워커(Node A)가 전사 텍스트를 POST(/process) 하면, LLM 두뇌(session/llm)를
    돌려 채점 + 다음 질문 + 비언어 메타데이터를 만든 뒤
        1) 행동 패킷을 Unity로 push        (제스처/표정 먼저 트리거)
        2) 대사를 TTS(Node B)로 합성해 음성을 Unity로 스트리밍
    하는 오케스트레이션을 수행한다.

  데이터 흐름:
    Unity --(mic audio WS)--> Node A(STT) --(POST text+features)--> [이 서버]
    Unity <--(control WS: JSON 패킷 + PCM 오디오)-- [이 서버] --(POST 대사)--> Node B(TTS)
"""
from __future__ import annotations

import asyncio
import json
import websockets
from websockets.protocol import State

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .domain import AnswerRequest, BehaviorPacket, ExpressionID, GestureID
from .session import InterviewSession
from . import tts_client

app = FastAPI(title="VRoom Backend", version="1.0")


# ---------------------------------------------------------------------------
# 세션 / WebSocket 레지스트리
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.sessions: dict[str, InterviewSession] = {}
        self.sockets: dict[str, WebSocket] = {}
        self.stt_sockets: dict[str, WebSocket] = {}
        self.tts_sockets: dict[str, websockets.WebSocketClientProtocol] = {}
        self.last_active: str | None = None  # session_id 미지정 POST 라우팅용
        self.lock = asyncio.Lock()

    async def register(self, sid: str, ws: WebSocket):
        self.sockets[sid] = ws
        self.last_active = sid
        try:
            tts_ws = await websockets.connect(settings.tts_ws_url, max_size=None)
            self.tts_sockets[sid] = tts_ws
            print(f"[{sid}] Persistent connection to TTS Worker established.")
        except Exception as e:
            self.tts_sockets[sid] = None
            print(f"[Warning] Failed to establish persistent connection to TTS Worker: {e}")

    async def unregister(self, sid: str):
        self.sockets.pop(sid, None)
        self.stt_sockets.pop(sid, None)
        tts_ws = self.tts_sockets.pop(sid, None)
        if tts_ws:
            try:
                await tts_ws.close()
                print(f"[{sid}] Persistent connection to TTS Worker closed.")
            except Exception as e:
                print(f"[{sid}] Error closing persistent connection to TTS Worker: {e}")

    async def get_or_connect_tts_ws(self, sid: str):
        tts_ws = self.tts_sockets.get(sid)
        if tts_ws is None or tts_ws.state == State.CLOSED:
            print(f"[{sid}] TTS WebSocket offline. Attempting lazy reconnect...")
            try:
                tts_ws = await websockets.connect(settings.tts_ws_url, max_size=None)
                self.tts_sockets[sid] = tts_ws
                print(f"[{sid}] Reconnected to TTS Worker successfully.")
            except Exception as e:
                self.tts_sockets[sid] = None
                print(f"[{sid}] Lazy reconnect to TTS Worker failed: {e}")
                return None
        return tts_ws

    async def send_packet(self, sid: str, packet: BehaviorPacket):
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_text(packet.model_dump_json())
            except Exception:
                await self.unregister(sid)

    async def send_json(self, sid: str, obj: dict):
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_text(json.dumps(obj, ensure_ascii=False))
            except Exception:
                await self.unregister(sid)

    async def send_audio(self, sid: str, chunk: bytes):
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_bytes(chunk)
            except Exception:
                await self.unregister(sid)


hub = Hub()


async def speak(sid: str, packet: BehaviorPacket):
    """행동 패킷 push -> TTS 합성 -> 음성 스트리밍 -> 종료 신호."""
    await hub.send_packet(sid, packet)                     # 1) 제스처/표정/대사 먼저
    
    if settings.skip_tts:                                  # TTS 생략 모드 (Node B 없이 테스트)
        await hub.send_json(sid, {"type": "audio_end"})
        stt_ws = hub.stt_sockets.get(sid)
        if stt_ws:
            try:
                await stt_ws.send_json({"type": "end"})
            except Exception:
                pass
        return

    # STT 소켓 연결 대기 (초기 동기화 안정성 확보)
    stt_ws = None
    for _ in range(30):
        stt_ws = hub.stt_sockets.get(sid)
        if stt_ws:
            break
        await asyncio.sleep(0.1)

    tts_ws = await hub.get_or_connect_tts_ws(sid)
    if tts_ws:
        try:
            async for chunk in tts_client.synthesize_ws_stream(tts_ws, packet.dialogue):  # 2) 음성/자막
                if stt_ws:
                    try:
                        if isinstance(chunk, bytes):
                            await stt_ws.send_bytes(chunk)
                        else:
                            await stt_ws.send_text(chunk)
                    except Exception as se:
                        print(f"[{sid}] Failed to relay chunk to STT socket: {se}")
                else:
                    if isinstance(chunk, bytes):
                        await hub.send_audio(sid, chunk)
                    else:
                        ws_ctrl = hub.sockets.get(sid)
                        if ws_ctrl:
                            try:
                                await ws_ctrl.send_text(chunk)
                            except:
                                pass
        except Exception as e:
            print(f"[{sid}] [TTS 릴레이 중 에러 - 음성 생략] {e}")
    else:
        print(f"[{sid}] [TTS 소켓 유실 - 음성 생략]")

    await hub.send_json(sid, {"type": "audio_end"})        # 3) 한 발화 끝 (Control 채널)
    if stt_ws:
        try:
            await stt_ws.send_json({"type": "end"})        # STT 채널에도 전송
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket: Unity 제어 채널
# ---------------------------------------------------------------------------
@app.websocket("/ws/control")
async def ws_control(ws: WebSocket):
    await ws.accept()
    sid: str | None = None
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "init":
                sid = msg.get("session_id") or "default"
                hub.sessions[sid] = InterviewSession(
                    session_id=sid,
                    company=msg.get("company", ""),
                    job_title=msg.get("job_title", ""),
                    resume=msg.get("resume", ""),
                )
                await hub.register(sid, ws)
                # 면접관이 먼저 자기소개를 요청 (첫 발화 + 음성)
                packet = await hub.sessions[sid].first_question()
                await speak(sid, packet)

            elif mtype == "utterance_end":
                # (옵션) Unity가 STT를 거치지 않고 직접 피쳐만 보낼 때 집계.
                if sid and sid in hub.sessions:
                    hub.sessions[sid]._collect_features(msg.get("features", {}))

            elif mtype == "request_feedback":
                if sid and sid in hub.sessions:
                    report = await hub.sessions[sid].build_feedback()
                    await hub.send_json(sid, report.model_dump())

    except WebSocketDisconnect:
        pass
    finally:
        if sid:
            await hub.unregister(sid)


# ---------------------------------------------------------------------------
# HTTP: STT 워커(Node A) -> 백엔드 전사 텍스트 전달
#   Node A 의 .env TTS_WORKER_URL 을 이 엔드포인트로 바꾸면 된다.
# ---------------------------------------------------------------------------
@app.post("/process")
async def process(req: AnswerRequest):
    sid = req.session_id or hub.last_active or "default"
    session = hub.sessions.get(sid)
    if session is None:
        return JSONResponse({"error": "no active session. Unity must send 'init' first."}, status_code=409)

    print(f"[STT→백엔드 수신] {req.text}")
    
    # '생각 중' 더미 모션을 즉시 띄워 인지적 대기시간을 가린다 (RTT 제어).
    await hub.send_packet(sid, BehaviorPacket(
        type="thinking", session_id=sid, stage=session.stage.value,
        dialogue="", expression_id=ExpressionID.THINKING.value,
        gesture_id=GestureID.REVIEW_RESUME.value, score=-1,
    ))

    packet = await session.on_user_answer(req.text, req.features)
    print(f"[백엔드→TTS 대사] {packet.dialogue}  (stage={packet.stage}, persona={packet.persona}, score={packet.score})")
    # 호환 모드: Node A 가 음성을 되받길 기대하면 HTTP 응답으로 스트리밍.
    if settings.proxy_audio_to_stt:
        await hub.send_packet(sid, packet)

        async def audio_gen():
            async for chunk in tts_client.synthesize_stream(packet.dialogue):
                yield chunk
        return StreamingResponse(audio_gen(), media_type="application/octet-stream")

    # 기본 모드: 행동 패킷 + 음성 모두 백엔드->Unity WS 로 직접 전송.
    await speak(sid, packet)
    return {"ok": True, "stage": packet.stage, "persona": packet.persona, "score": packet.score}


@app.get("/health")
async def health():
    return {"status": "ok", "provider": settings.llm_provider, "active_sessions": len(hub.sessions)}

@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    """
    STT 워커가 '사용자 답변 텍스트'를 보내는 입구.
    백엔드가 채점/페르소나/질문 생성 후,
      - 자막+행동패킷을 Unity(/ws/control)로 push
      - 면접관 대사를 진짜 TTS(/ws/tts)로 합성해 음성을 STT로 릴레이
    STT 입장에선 기존 TTS와 동일하게 (음성청크 + {"type":"end"}) 를 받는다.
    """
    await ws.accept()
    sid = ws.query_params.get("session_id", "default")
    hub.stt_sockets[sid] = ws
    print(f"[/ws/tts] STT 워커 연결됨 - Session ID: {sid}")
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            user_text = msg.get("text", "")
            if not user_text:
                continue

            msg_sid = msg.get("session_id") or sid or hub.last_active or "default"
            session = hub.sessions.get(msg_sid)
            if session is None:
                print(f"[/ws/tts] 활성 세션 없음 ({msg_sid}) (Unity init 먼저 필요)")
                await ws.send_json({"type": "end"})
                continue

            print(f"[STT→백엔드 수신 ({msg_sid})] {user_text}")

            # '생각 중' 모션을 Unity로 먼저
            await hub.send_packet(msg_sid, BehaviorPacket(
                type="thinking", session_id=msg_sid, stage=session.stage.value,
                dialogue="", expression_id=ExpressionID.THINKING.value,
                gesture_id=GestureID.REVIEW_RESUME.value, score=-1,
            ))

            # 채점 + 다음 질문 생성
            features = msg.get("features", {})
            packet = await session.on_user_answer(user_text, features)
            print(f"[백엔드→TTS 대사] {packet.dialogue} "
                  f"(stage={packet.stage}, persona={packet.persona}, score={packet.score})")

            # 자막 + 행동패킷을 Unity로 (자막=면접관 대사)
            await hub.send_packet(msg_sid, packet)

            # 면접관 대사를 진짜 TTS로 합성 → 음성/자막을 STT로 릴레이
            tts_ws = await hub.get_or_connect_tts_ws(msg_sid)
            if tts_ws:
                try:
                    async for chunk in tts_client.synthesize_ws_stream(tts_ws, packet.dialogue):
                        if isinstance(chunk, bytes):
                            await ws.send_bytes(chunk)   # 음성을 STT로 릴레이
                        else:
                            await ws.send_text(chunk)    # 자막 JSON을 STT로 릴레이
                except Exception as e:
                    print(f"[/ws/tts] [{msg_sid}] TTS 릴레이 실패: {e}")
            else:
                print(f"[/ws/tts] [{msg_sid}] TTS 소켓 유실로 릴레이 생략")

            # 한 발화 끝 신호 (STT가 이걸 받고 Unity VAD 잠금 해제)
            await ws.send_json({"type": "end"})

    except WebSocketDisconnect:
        print(f"[/ws/tts] STT 워커 연결 종료 - Session ID: {sid}")
    except Exception as e:
        print(f"[/ws/tts] Error for Session ID {sid}: {e}")
    finally:
        hub.stt_sockets.pop(sid, None)
