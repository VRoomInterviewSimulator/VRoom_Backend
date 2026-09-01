"""면접 세션 상태머신. 사용자 한 명당 하나씩 만들어 들고 있는다.

이 클래스가 책임지는 것:
  1) 단계(Stage) 진행       — 답변 하나당 한 칸 전진
  2) 동적 페르소나          — 자기소개에서 정보를 추출해 이후 모든 질문에 주입
  3) 페르소나 가변 전환     — 점수 -> 긍정/중립/부정, 연속 저점 시 압박 고착
  4) 대화 기록 누적         — 면접이 6턴 내외로 짧아 전체를 그대로 보관
  5) 멀티모달 피쳐 집계     — 음성(발화시간/침묵 등) + 웹캠(시선/자세 등)
  6) 개입 상태 관리         — 어느 단계에서 몇 번 개입했는지, 잘린 답변 점수 보관
  7) 종료 시 최종 채점      — 10항목 점수 + LLM 총평

설계 원칙: **시나리오 진행과 페르소나 결정은 이 파일(코드)이 통제한다.**
LLM 은 '주어진 단계/페르소나에 맞는 대사와 점수'만 만든다.
"""
from __future__ import annotations

import math
import time

from . import llm
from .config import settings
from .domain import (
    BehaviorPacket,
    ExpressionID,
    ExtractedInfo,
    FeedbackReport,
    GestureID,
    InterviewScore,
    LLMTurn,
    Persona,
    STAGE_ORDER,
    Stage,
    StageScore,
    persona_from_score,
    persona_value_from_score,
)


class InterviewSession:

    # =======================================================================
    # 1. 생성 / 상태 정의
    # =======================================================================
    def __init__(self, session_id: str, company: str = "", job_title: str = "",
                 resume: str = "", condition: str = "C"):
        # ── 세션 식별 / 입력값 ──────────────────────────────────────
        self.session_id = session_id                 # 세션 식별자
        self.company = company                       # Unity init 이 준 지원 기업
        self.job_title = job_title                   # Unity init 이 준 지원 직무
        self.resume = resume                         # 이력서 원문
        self.condition: str = condition              # 실험 조건 "A" | "B" | "C"

        # ── 동적 페르소나 ──────────────────────────────────────────
        self.info = ExtractedInfo(company_name=company, job_role=job_title)
                                                     # 자기소개에서 추출한 지원자 정보
        self._info_extracted = False                 # 추출을 이미 했는지 (1회만 수행)

        # ── 진행 상태 ──────────────────────────────────────────────
        self.stage: Stage = Stage.INIT               # 현재 단계
        self.persona: Persona = Persona.NEUTRAL      # 현재 면접관 태도
        self.consecutive_low = 0                     # 연속 저점 횟수 (압박 고착 판단용)
        self.last_persona_value = 0.0                # 마지막으로 Unity 에 보낸 감정 강도

        # ── 대화 기록 ──────────────────────────────────────────────
        self.turns: list[dict] = []                  # 전체 발화 {"role","stage","text"}
        self.current_question_text: str = ""         # 직전 면접관 질문 (이탈 판정 기준)

        # ── 채점 원자료: 답변 품질 ─────────────────────────────────
        self.stage_scores: list[tuple[str, int]] = []   # (단계명, 점수) 목록

        # ── 채점 원자료: 음성 피쳐 (모두 턴 단위, 인덱스가 서로 대응) ──
        self.turn_stages: list[str] = []             # 각 턴이 어느 단계였는지
        self.speaking_times: list[float] = []        # 턴별 발화 시간(초)
        self.cps_list: list[float] = []              # 턴별 초당 글자 수
        self.meaningful_pauses: list[int] = []       # 턴별 의미 있는 침묵 횟수
        self.volume_variances: list[float] = []      # 턴별 볼륨 분산
        self.low_volume_ratios: list[float] = []     # 턴별 작은 목소리 구간 비율
        self.response_times: list[float] = []        # 턴별 반응 시간(초)
        self.average_volumes: list[float] = []       # 턴별 평균 볼륨

        # ── 채점 원자료: 웹캠 피쳐 ─────────────────────────────────
        self.vision_turns: list[dict] = []           # Vision 워커가 POST 한 턴별 피쳐

        # ── 개입 제어 ──────────────────────────────────────────────
        self.bargein_used_stages: set[str] = set()   # 이미 개입한 단계 (G3)
        self.bargein_total: int = 0                  # 세션 총 개입 횟수 (G4)
        self.pending_cutoff = None                   # 진행 중인 Type B 결정 (BargeInDecision)
        self.pending_redirect = None                 # 진행 중인 Type A 결정 (BargeInDecision)
        self.awaiting_reanswer: bool = False         # Type A 재답변 대기 중인가
        self.truncated_captured: bool = False        # Type A 잘린 전사를 이미 받았는가
        self.pending_truncated_text: str = ""        # 보관 중인 잘린 답변 텍스트
        self.pending_truncated_score: int = -1       # 보관 중인 잘린 답변 점수
        self.pending_partial_text: str = ""          # (Type A) 이탈 판정에 쓴 부분 전사.
        self.utterance_started_at: float = 0.0       # 현재 발화 시작 시각 (G6 유예 계산용)

        # ── 실험 로그 ──────────────────────────────────────────────
        self.session_started_at: float = time.time() # 세션 시작 시각 (상대 시각 계산 기준)
        self.bargein_log: list[dict] = []            # 개입 이벤트별 상세 기록
        self.turn_features: list[dict] = []          # turn_stages 와 인덱스가 1:1 로 대응되는 로그
        self.log_written: bool = False           # 세션 로그를 이미 파일로 떨어뜨렸는가

    # =======================================================================
    # 2. 내부 유틸 — 기록 / 단계 / 패킷 변환
    # =======================================================================
    def _record(self, role: str, text: str, update_question: bool = True):
        """발화 한 건을 기록한다. 면접관 발화면 '직전 질문'으로도 갱신한다.

        update_question=False: 개입 대사와 마무리 멘트는 질문이 아니다.
        여기서 덮어쓰면 이탈 판정(judge_off_topic)과 답변 채점의 기준 질문이
        개입 대사로 바뀌어, 실제 질문과 무관하게 판정된다.
        """
        self.turns.append({"role": role, "stage": self.stage.value, "text": text})
        if role == "interviewer" and text and update_question:
            self.current_question_text = text

    def _history_text(self) -> str:
        """지금까지의 대화를 LLM 프롬프트에 넣을 한 덩어리 텍스트로 만든다."""
        lines = []
        for t in self.turns:
            who = "면접관" if t["role"] == "interviewer" else "지원자"
            lines.append(f"{who}({t['stage']}): {t['text']}")
        return "\n".join(lines)

    def _advance_stage(self):
        """다음 단계로 한 칸 이동한다. 마지막 단계에서는 더 이상 움직이지 않는다."""
        if self.stage == Stage.INIT:
            self.stage = STAGE_ORDER[0]
            return
        idx = STAGE_ORDER.index(self.stage)
        self.stage = STAGE_ORDER[min(idx + 1, len(STAGE_ORDER) - 1)]

    def _to_packet(self, turn: LLMTurn, is_final: bool,
                   bargein_type: str = "") -> BehaviorPacket:
        """LLMTurn -> Unity 로 보낼 BehaviorPacket 으로 변환한다(감정 강도 계산 포함)."""
        pv = persona_value_from_score(turn.score, self.consecutive_low, self.condition)
        self.last_persona_value = pv
        return BehaviorPacket(
            session_id=self.session_id,
            stage=self.stage.value,
            persona=self.persona.value,
            persona_value=pv,
            dialogue=turn.dialogue,
            expression_id=turn.expression_id,
            gesture_id=turn.gesture_id,
            score=turn.score,
            is_final=is_final,
            bargein_type=bargein_type,
        )

    def _ignored_packet(self, stage_value: str, bargein_type: str = "") -> BehaviorPacket:
        """Unity 로 대사를 보내지 않아야 할 때 쓰는 빈 패킷. 호출자가 발송을 건너뛴다."""
        return BehaviorPacket(
            type="ignored", session_id=self.session_id, stage=stage_value,
            persona=self.persona.value, persona_value=self.last_persona_value,
            dialogue="", expression_id=ExpressionID.NEUTRAL.value,
            gesture_id=GestureID.IDLE.value, score=-1, is_final=False,
            bargein_type=bargein_type,
        )

    # =======================================================================
    # 3. 면접관 발화 생성
    # =======================================================================
    FIRST_QUESTION_TEMPLATE = (
        "안녕하세요. {company} {job} 직무 면접을 시작하겠습니다. "
        "먼저 지원 동기를 포함해 자기소개를 부탁드립니다."
    )

    def template_first_question(self) -> BehaviorPacket:
        """첫 질문을 LLM 없이 템플릿으로 만든다. 프리웜 단계에서 지연을 0으로 만드는 용도."""
        if self.stage == Stage.INIT:
            self._advance_stage()

        company = (self.company or "").strip().splitlines()[0] if self.company else ""
        job = (self.job_title or "").strip().splitlines()[0] if self.job_title else ""
        dialogue = self.FIRST_QUESTION_TEMPLATE.format(
            company=company or "저희 회사",
            job=job or "지원하신",
        )

        self._record("interviewer", dialogue)
        self.last_persona_value = 0.0
        return BehaviorPacket(
            type="interviewer_turn",
            session_id=self.session_id,
            stage=self.stage.value,
            persona=self.persona.value,
            persona_value=0.0,
            dialogue=dialogue,
            expression_id=ExpressionID.WARM_SMILE.value,
            gesture_id=GestureID.WELCOME.value,
            score=-1,
            is_final=False,
        )

    async def first_question(self) -> BehaviorPacket:
        """면접 시작 발화. 설정에 따라 템플릿 또는 LLM 생성을 고른다."""
        if settings.template_first_question:
            return self.template_first_question()

        self._advance_stage()
        turn = await llm.generate_turn(
            stage=self.stage, persona=self.persona,
            info=self.info, resume=self.resume,
            history="", user_answer="",
        )
        self._record("interviewer", turn.dialogue)
        return self._to_packet(turn, is_final=False)

    async def on_user_answer(self, text: str, features: dict, truncated: bool = False) -> BehaviorPacket:
        """사용자 답변을 받아 채점하고 다음 단계 발화를 만든다. 세션의 메인 루프.

        반환 패킷의 type 이 "ignored" 면 호출자는 Unity 로 아무것도 보내지 않아야 한다.
        """
        # (0) 면접 종료 후 들어온 발화는 버린다 (결과 화면에서의 혼잣말 등).
        if self.stage == Stage.DONE:
            print(f"[{self.session_id}] 면접 종료 후 발화 수신 - 무시: {text[:30]}")
            return self._ignored_packet(Stage.DONE.value)

        # (0-b) 워치독이 이미 잘린 답변 자리를 채운 뒤 도착한 늦은 전사.
        #       재답변으로 오인하면 단계가 두 칸 전진한다.
        if self.awaiting_reanswer and self.truncated_captured and truncated:
            print(f"[개입] 늦게 도착한 잘린 전사 - 폐기: {text[:30]}")
            return self._ignored_packet(self.stage.value, bargein_type="REDIRECT")
        
        # (1) Type A 잘린 답변이면 채점만 하고 단계를 전진시키지 않는다.
        if self.awaiting_reanswer and not self.truncated_captured:
            if truncated:
                return await self._absorb_truncated_answer(text, features)
        
            await self._absorb_missing_truncated()

        # (2) 답변을 기록하고 음성 피쳐를 집계한다.
        #     이 세 줄은 항상 함께 실행되어야 리스트 인덱스 정합이 유지된다.
        self._record("user", text)
        self.turn_stages.append(self.stage.value)
        self._collect_features(features, text)

        # (3) 자기소개 답변이면 동적 페르소나 정보를 1회 추출한다.
        if self.stage == Stage.SELF_INTRO and not self._info_extracted:
            await self._extract_persona_info(text)

        # (4) 단계 전진. 마무리 답변이었다면 여기서 종료 멘트로 끝낸다.
        was_closing = self.stage == Stage.CLOSING
        self._advance_stage()
        if was_closing:
            return self._closing_packet()

        # (5) Type B 개입으로 잘린 답변인지 확인하고, LLM 에 줄 전제를 만든다.
        cutoff = self.pending_cutoff
        self.pending_cutoff = None
        bargein_note = self._build_cutoff_instruction(cutoff)

        # (6) 직전 답변 채점 + 다음 질문 생성 (동적 페르소나 정보 주입).
        turn = await llm.generate_turn(
            stage=self.stage, persona=self.persona,
            info=self.info, resume=self.resume,
            history=self._history_text(), user_answer=text,
            extra_instruction=bargein_note,
        )

        # (7) 점수를 반영해 페르소나를 갱신한다 (코드가 최종 결정권).
        turn = self._apply_score(turn)

        # (8) 개입 후속 처리 — 태그와 비언어 ID 확정.
        bargein_type = self._finalize_bargein(turn, cutoff)

        self._record("interviewer", turn.dialogue)
        return self._to_packet(turn, is_final=False, bargein_type=bargein_type)

    async def _extract_persona_info(self, self_intro: str):
        """자기소개 텍스트에서 회사/직무/경력/기술/강점을 뽑아 동적 페르소나를 활성화한다."""
        extracted = await llm.extract_info(
            self_intro, fallback_company=self.company, fallback_job=self.job_title
        )
        self.info = extracted
        if not self.info.company_name:
            self.info.company_name = self.company
        if not self.info.job_role:
            self.info.job_role = self.job_title
        self._info_extracted = True

    def _closing_packet(self) -> BehaviorPacket:
        """마무리 답변까지 끝났을 때의 종료 멘트. 여기서 is_final=True 를 세운다."""
        self.stage = Stage.DONE
        closing = BehaviorPacket(
            session_id=self.session_id, stage=Stage.DONE.value,
            persona=self.persona.value,
            persona_value=self.last_persona_value,
            dialogue="면접에 응해 주셔서 감사합니다. 잠시 후 결과를 안내해 드리겠습니다.",
            expression_id=ExpressionID.WARM_SMILE.value,
            gesture_id=GestureID.DEEP_NOD.value, score=-1, is_final=True,
        )
        self._record("interviewer", closing.dialogue, update_question=False)
        return closing

    def _apply_score(self, turn: LLMTurn) -> LLMTurn:
        """LLM 이 매긴 점수를 세션 상태에 반영하고 페르소나를 전환한다.

        점수 귀속에 turn_stages[-1] 을 쓰는 이유: turns[-2] 로 역산하면 개입이
        레코드를 끼워 넣을 때 다른 단계 점수로 조용히 오귀속된다.
        """
        if turn.score < 0:
            return turn

        prev_stage_name = self.turn_stages[-1] if self.turn_stages else self.stage.value
        self.stage_scores.append((prev_stage_name, turn.score))
        self._consecutive_low_prev = self.consecutive_low  
        self.consecutive_low = self.consecutive_low + 1 if turn.score < 40 else 0
        self.persona = persona_from_score(turn.score, self.consecutive_low, self.condition)
        turn = llm._clamp_to_set(turn, self.persona)

        # 압박 고착: 연속 2회 이상 저점이면 강한 부정 비언어를 강제한다.
        if self.persona == Persona.NEGATIVE and self.consecutive_low >= 2:
            turn.expression_id = ExpressionID.SLIGHT_FROWN.value
            turn.gesture_id = GestureID.ARMS_CROSSED.value
        return turn

    # =======================================================================
    # 4. 개입(끼어들기)
    # =======================================================================
    def commit_bargein(self, decision) -> None:
        """개입 허가 직후 세션 상태를 확정한다. 컷인 메시지 발송과 동시에 호출된다."""
        from .domain import BargeInType

        self.bargein_used_stages.add(self.stage.value)
        self.bargein_total += 1

        # 실험 분석용 이벤트 레코드. 이후 단계에서 필드가 하나씩 채워진다.
        self.bargein_log.append({
            "stage": self.stage.value,                                   # 개입이 일어난 단계
            "type": decision.bargein_type,                               # REDIRECT | CUTOFF
            "reason": decision.reason,                                   # 발동 원인
            "triggered_at": round(time.time() - self.session_started_at, 2),  # 세션 시작 기준 경과초
            "persona_value_at_trigger": self.last_persona_value,         # 개입 시점의 감정 강도
            "utterance_elapsed": decision.meta.get("utterance_elapsed", 0.0),  # 잘린 발화 길이
            "partial_text": decision.meta.get("partial_text", ""),       # (Type A) 판정에 쓴 부분 전사
            "latency_judge_ms": decision.judge_latency_ms,               # (Type A) LLM 이탈 판정 소요
            "yield_time": None,                                          # 개입 후 사용자가 멈추기까지
            "score_truncated": -1,                                       # 잘린 답변 점수
            "score_reanswer": -1,                                        # (Type A) 재답변 점수
            "score_final": -1,                                           # 최종 반영 점수
            "latency_to_speech_ms": None,                                # 개입 확정 -> 첫 발성 (논문 지표)
            "latency_full_ms": None,                                     # 개입 확정 -> 합성 완료 (참고용)
        })

        if decision.bargein_type == BargeInType.REDIRECT.value:
            self.pending_redirect = decision
            self.awaiting_reanswer = True
            self.truncated_captured = False
            self.pending_partial_text = decision.meta.get("partial_text", "")
        else:
            self.pending_cutoff = decision

    def note_yield_time(self, yield_time: float):
        """Unity 가 측정한 '개입 후 사용자가 입을 다물기까지의 시간'을 기록한다."""
        if self.bargein_log:
            self.bargein_log[-1]["yield_time"] = round(yield_time, 2)

    def note_speech_latency(self, ms: int):
        """개입 확정부터 첫 발성까지의 지연을 기록한다 (설계서 9.6 기준값)."""
        if self.bargein_log:
            self.bargein_log[-1]["latency_to_speech_ms"] = ms

    def _build_cutoff_instruction(self, cutoff) -> str:
        """Type B 개입 후 다음 질문을 만들 때 LLM 에 붙일 전제 문장. 개입이 없으면 빈 문자열."""
        if cutoff is None:
            return ""
        cause = "지나치게 길어져" if cutoff.reason == "LONG_ANSWER" else "응답이 없어"
        return (
            f"[개입 전제] 직전 지원자 답변은 {cause} 면접관이 도중에 말을 끊고 종료시킨 것이다. "
            "대사는 (1) 끊는 신호, (2) 간결한 지적, (3) 다음 질문 순서로 구성하고 "
            "원본 질문보다 짧게, 2문장 이내로 말하라. "
            "개입 자체를 사과하거나 변명하지 말고 단호하게 진행하라."
        )

    def _finalize_bargein(self, turn: LLMTurn, cutoff) -> str:
        """개입 후속 처리: 패킷 태그를 정하고 필요하면 점수를 혼합/비언어를 덮어쓴다."""
        # Type B — 발화 2(다음 질문). 잘린 답변 점수를 그대로 최종 점수로 쓴다.
        if cutoff is not None:
            if self.bargein_log and turn.score >= 0:
                self.bargein_log[-1]["score_truncated"] = turn.score
                self.bargein_log[-1]["score_final"] = turn.score
            # 개입 직후이므로 표정이 중립으로 풀리지 않게 부정 비언어를 강제한다.
            turn.expression_id = ExpressionID.SLIGHT_FROWN.value
            turn.gesture_id = GestureID.ARMS_CROSSED.value
            return "CUTOFF_QUESTION"

        # Type A — 재답변 도착. 잘린 답변 점수와 가중 혼합한다.
        if self.awaiting_reanswer and self.truncated_captured:
            self._blend_reanswer_score(turn)
            return "REDIRECT_REANSWER"

        return ""

    async def _absorb_truncated_answer(self, text: str, features: dict) -> BehaviorPacket:
        """Type A: 잘린 답변을 채점해 보관만 하고 단계는 전진시키지 않는다.

        주의: turn_stages.append() 와 _collect_features() 를 호출하지 않는다.
              turn_stages / speaking_times / vision_turns 의 길이 정합이 유지되어야
              최종 채점의 인덱스 정렬이 깨지지 않는다. 잘린 답변의 음성 피쳐는
              정의상 짧아 지표를 왜곡하므로 버리고 재답변의 것만 쓴다.
        """
        self.truncated_captured = True
        self.pending_truncated_text = text
        self._record("user", text)

        score = await llm.score_answer(question=self.current_question_text, answer=text)
        self.pending_truncated_score = score
        if self.bargein_log:
            self.bargein_log[-1]["score_truncated"] = score

        print(f"[개입] 잘린 답변 채점 완료 score={score} "
              f"stage={self.stage.value} (단계 유지, 재답변 대기)")

        # 개입 대사는 이미 별도로 발송되었으므로 여기서는 아무 대사도 보내지 않는다.
        return self._ignored_packet(self.stage.value, bargein_type="REDIRECT")

    async def _absorb_missing_truncated(self) -> None:
        """잘린 전사가 끝내 오지 않았을 때 부분 전사로 대체 채점한다.

        Type A 의 유일한 치명적 실패 모드는 '재답변이 잘린 답변으로 흡수되어
        단계가 전진하지 않는 것'이다. 여기서 truncated_captured 를 반드시 세워
        그 경로를 끊는다. 부분 전사조차 없으면 점수는 -1 로 두고
        _blend_reanswer_score 가 재답변 점수만 쓰도록 맡긴다.
        """
        self.truncated_captured = True
        partial = (self.pending_partial_text or "").strip()

        if not partial:
            self.pending_truncated_score = -1
            print("[개입] 잘린 전사·부분 전사 모두 없음 - 재답변 점수만 사용")
            return

        self.pending_truncated_text = partial
        score = await llm.score_answer(
            question=self.current_question_text, answer=partial)
        self.pending_truncated_score = score
        if self.bargein_log:
            self.bargein_log[-1]["score_truncated"] = score
            self.bargein_log[-1]["truncated_source"] = "partial_fallback"

        print(f"[개입] 잘린 전사 유실 - 부분 전사로 대체 채점 score={score} "
              f"'{partial[:30]}'")

    def _blend_reanswer_score(self, turn: LLMTurn) -> LLMTurn:
        """잘린 답변과 재답변 점수를 가중 혼합하고 Type A 대기 상태를 해제한다."""
        from .config import BargeInConfig as B

        s_re = turn.score
        s_tr = self.pending_truncated_score
        if s_re >= 0 and s_tr >= 0:
            blended = round(B.W_TRUNCATED * s_tr + B.W_REANSWER * s_re)
        else:
            blended = max(s_re, s_tr)

        if self.bargein_log:
            self.bargein_log[-1]["score_reanswer"] = s_re
            self.bargein_log[-1]["score_final"] = blended

        # stage_scores 에는 혼합값만 남기고, 원본 두 점수는 bargein_log 에 보존한다.
        if self.stage_scores and s_re >= 0:
            self.stage_scores[-1] = (self.stage_scores[-1][0], blended)
        elif s_tr >= 0:
            # 재답변 채점이 실패(-1)하면 _apply_score 가 조기 반환해
            # stage_scores 에 그 턴이 아예 안 남는다. 잘린 답변 점수로라도 채운다.
            self.stage_scores.append((self.turn_stages[-1] if self.turn_stages
                                      else self.stage.value, blended))
            print(f"[개입] 재답변 채점 실패 - 잘린 답변 점수({blended})로 대체 기록")

        if s_re >= 0 and s_tr >= 0:
            prev_low = getattr(self, "_consecutive_low_prev", 0)
            self.consecutive_low = prev_low + 1 if blended < 40 else 0
            self.persona = persona_from_score(blended, self.consecutive_low, self.condition)
            turn = llm._clamp_to_set(turn, self.persona)
            print(f"[개입] 혼합 점수 기준 페르소나 재산정 -> {self.persona.value}")

        print(f"[개입] 점수 혼합 truncated={s_tr} reanswer={s_re} -> {blended}")

        self.awaiting_reanswer = False
        self.truncated_captured = False
        self.pending_truncated_score = -1
        self.pending_truncated_text = ""
        self.pending_partial_text = ""
        self.pending_redirect = None

        turn.score = blended
        return turn

    # =======================================================================
    # 5. 피쳐 수집
    # =======================================================================
    def _collect_features(self, features: dict, text: str = ""):
        """Unity/STT 가 보낸 턴별 음성 피쳐를 각 리스트에 한 칸씩 쌓는다.

        계약: 이 함수가 한 번 호출되면 모든 리스트가 정확히 한 칸씩 늘어난다.
        예전에는 speakingTime<=0 이나 features 부재 시 일부 리스트만 건너뛰어,
        그 이후 턴부터 turn_stages 와 인덱스가 어긋났다. 그러면 반응 속도가
        다른 단계의 기대값으로 채점된다.
        무효한 턴은 여기서 0 으로 채우고 _score_voice 가 걸러낸다.
        """
        st = features.get("speakingTime") if features else None
        valid_st = isinstance(st, (int, float)) and st > 0

        # 로그 전용 행
        self.turn_features.append({
            "stage": self.stage.value,
            "text_len": len(text),
            "speakingTime": round(float(st), 3) if valid_st else None,
            "cps": round(len(text) / float(st), 2) if valid_st else None,
            "meaningfulPauseCount": int(features.get(
                "meaningfulPauseCount", features.get("pauseCount", 0))) if features else None,
            "volumeVariance": features.get("volumeVariance") if features else None,
            "lowVolumeRatio": features.get("lowVolumeRatio") if features else None,
            "responseTime": features.get("responseTime") if features else None,
            "averageVolume": features.get("averageVolume") if features else None,
        })

        # 채점용 리스트 — 조건 없이 전부 한 칸씩. 무효 턴은 0(또는 기준값)으로 채운다.
        self.speaking_times.append(float(st) if valid_st else 0.0)
        self.cps_list.append(len(text) / float(st) if valid_st else 0.0)
        self.meaningful_pauses.append(int(features.get(
            "meaningfulPauseCount", features.get("pauseCount", 0))) if features else 0)
        self.volume_variances.append(
            float(features.get("volumeVariance", 0.0)) if features else 0.0)
        self.low_volume_ratios.append(
            float(features.get("lowVolumeRatio", 0.0)) if features else 0.0)
        self.response_times.append(
            float(features.get("responseTime", 0.0)) if features else 0.0)
        self.average_volumes.append(
            float(features.get("averageVolume", 0.1)) if features else 0.1)

    def collect_vision_features(self, features: dict):
        """Vision 워커가 턴 종료 시 POST 한 웹캠 피쳐를 보관한다."""
        if features:
            self.vision_turns.append(features)

    # =======================================================================
    # 6. 최종 채점
    # =======================================================================
    async def build_feedback(self) -> FeedbackReport:
        """면접 종료 시 10항목 점수 + LLM 총평을 묶어 결과 리포트를 만든다."""
        valid_times = [t for t in self.speaking_times if t > 0]
        turn_count = len(valid_times)
        avg_speak = (sum(valid_times) / turn_count) if turn_count > 0 else 0.0
        total_pauses = sum(self.meaningful_pauses)

        # (1) LLM 총평 — 강점/개선점/총평 + filler_score, density_score
        data = await llm.generate_feedback(
            company=self.info.company_name or self.company,
            job_title=self.info.job_role or self.job_title,
            transcript=self._history_text(), stage_scores=self.stage_scores,
            avg_speaking_time=avg_speak, total_pauses=total_pauses,
        )

        # (2) 답변 품질 — 단계별 점수(0~100)의 평균을 10점으로 환산
        scored = [v for _, v in self.stage_scores if v >= 0]
        accuracy10 = round((sum(scored) / len(scored)) / 10) if scored else 0

        # (3) 음성 4항목 — 턴별 점수의 평균
        vol_score, speed_score, time_score_avg, rt_score = self._score_voice()

        # (4) 답변 길이 = 물리적 시간 점수와 LLM 내용 밀도 점수의 50:50
        density_score = data.get("density_score", 5)
        length_score = max(0, min(10, (time_score_avg + density_score) // 2))

        # (5) 추임새 — LLM 정성 평가
        filler_score = data.get("filler_score", 5)

        # (6) 시각 4항목 — 웹캠
        vision_ok, gaze_s, gesture_s, posture_s, expr_s = self._score_vision()

        scores = InterviewScore(
            accuracy=accuracy10,
            voiceVolume=vol_score,
            voiceSpeed=speed_score,
            answerLength=length_score,
            fillerWords=filler_score,
            responseTime=rt_score,
            gaze=gaze_s,
            gesture=gesture_s,
            posture=posture_s,
            expression=expr_s,
        )

        # (7) 종합 점수 — 웹캠을 못 쓴 세션은 음성 6항목(60점)을 100점으로 환산
        if vision_ok:
            overall = sum(scores.model_dump().values())
        else:
            voice_sum = (accuracy10 + vol_score + speed_score
                         + length_score + filler_score + rt_score)
            overall = round(voice_sum / 60 * 100)

        return FeedbackReport(
            session_id=self.session_id,
            scores=scores,
            overall_score=overall,
            stage_scores=[StageScore(stage=s, score=v) for s, v in self.stage_scores],
            strengths=data.get("strengths", ""),
            improvements=data.get("improvements", ""),
            summary=data.get("summary", ""),
            avg_speaking_time=round(avg_speak, 1),
            total_pauses=total_pauses,
        )

    def _score_voice(self) -> tuple[int, int, int, int]:
        """음성 4항목을 턴별로 채점해 평균을 낸다.

        반환: (목소리 크기, 발화 속도, 답변 길이-시간분, 반응 속도)
        답변 길이는 여기서 '물리적 시간 점수'만 내고, LLM 밀도 점수와의 합산은 호출자가 한다.

        모든 리스트가 turn_stages 와 인덱스 정합을 이루므로, 발화가 실제로
        있었던 턴(speakingTime>0)만 골라 그 인덱스로 전부를 조회한다.
        """
        from .config import ScoringConfig as S

        idxs = [i for i, st in enumerate(self.speaking_times) if st > 0]
        if not idxs:
            return (10, 10, 10, 10)   # 유효한 턴이 없으면 감점할 근거가 없으므로 만점

        total_vol = total_speed = total_length = total_rt = 0

        for i in idxs:
            # ── 1. 목소리 크기 ──────────────────────────────────────
            vol = self.average_volumes[i]
            var = self.volume_variances[i]
            low = self.low_volume_ratios[i]

            s_vol = 10 - int(abs(vol - S.VOICE_VOLUME_MEAN) / S.VOICE_VOLUME_TOLERANCE)
            if var > S.VOLUME_VARIANCE_THRESHOLD:
                s_vol -= S.VOLUME_VARIANCE_PENALTY
            if low > S.LOW_VOLUME_RATIO_THRESHOLD:
                s_vol -= S.LOW_VOLUME_RATIO_PENALTY
            total_vol += max(0, min(10, s_vol))

            # ── 2. 발화 속도 ────────────────────────────────────────
            cps = self.cps_list[i]
            pauses = self.meaningful_pauses[i]

            s_speed = 10 - int(abs(cps - S.VOICE_SPEED_MEAN) / S.VOICE_SPEED_TOLERANCE)
            s_speed -= int(max(0, pauses - S.PAUSE_ALLOWANCE))
            total_speed += max(0, min(10, s_speed))

            # ── 3. 답변 길이 (물리적 시간) ──────────────────────────
            st = self.speaking_times[i]
            if st < S.ANSWER_LENGTH_MIN:
                s_len = int((st / S.ANSWER_LENGTH_MIN) * 10)
            elif st > S.ANSWER_LENGTH_MAX:
                s_len = 10 - int((st - S.ANSWER_LENGTH_MAX) / 10)
            else:
                s_len = 10
            total_length += max(0, min(10, s_len))

            # ── 4. 반응 속도 (단계별 기대값이 다르다) ───────────────
            rt = self.response_times[i]
            stage = self.turn_stages[i] if i < len(self.turn_stages) else "UNKNOWN"
            target_rt = (S.RESPONSE_TIME_INTRO_MEAN if stage == "SELF_INTRO"
                         else S.RESPONSE_TIME_FOLLOWUP_MEAN)

            s_rt = 10 - int(abs(rt - target_rt) / S.RESPONSE_TIME_TOLERANCE)
            total_rt += max(0, min(10, s_rt))

        n = len(idxs)
        return (round(total_vol / n), round(total_speed / n),
                round(total_length / n), round(total_rt / n))

    def _score_vision(self) -> tuple[bool, int, int, int, int]:
        """웹캠 4항목을 턴별로 채점해 평균을 낸다.

        반환: (사용가능여부, gaze, gesture, posture, expression)
        사용가능여부가 False 면 결과 UI 에서 시각 항목을 제외하고 100점으로 환산한다.
        """
        from .config import VisionScoringConfig as V

        # (1) 얼굴 검출이 부실한 턴은 신뢰할 수 없으므로 제외한다.
        turns = [t for t in self.vision_turns
                 if t.get("faceDetectedRatio", 0.0) >= V.MIN_FACE_RATIO]
        if not turns:
            print("[채점] 유효한 시각 피쳐 없음 -> 시각 4항목 제외")
            return (False, 0, 0, 0, 0)

        # (2) (stage, phase) 중복 제거.
        #     개입이 일어나면 같은 stage 에 위상이 다른 턴이 여러 개 생긴다.
        #     stage 단독을 키로 쓰면 그중 하나가 소리 없이 사라지고,
        #     어느 쪽이 남을지도 frameCount 에 따라 달라져 비결정적이다.
        best: dict[tuple, dict] = {}
        for t in turns:
            key = (t.get("stage", ""), t.get("phase", "NORMAL"))
            if key not in best or t.get("frameCount", 0) > best[key].get("frameCount", 0):
                best[key] = t
        if len(best) < len(turns):
            print(f"[채점] 시각 턴 중복 제거: {len(turns)} -> {len(best)}")

        # (3) 채점 대상 위상만 남긴다. TRUNCATED / REACTION 은 로그 전용이다.
        #     구간이 짧아 지표가 불안정하고, 포함하면 개입이 일어난 단계의 가중치가
        #     부당하게 커진다.
        SCORED_PHASES = {"NORMAL", "REANSWER"}
        turns = [t for t in best.values() if t.get("phase", "NORMAL") in SCORED_PHASES]
        if not turns:
            print("[채점] 채점 대상 위상의 시각 턴 없음 -> 시각 4항목 제외")
            return (False, 0, 0, 0, 0)

        # (4) 턴별 감점 후 평균.
        g = ge = p = e = 0
        for t in turns:
            # 자세 미검출 턴은 손짓/자세를 측정할 수 없다.
            # 감점하지 않고 중립 처리한다(측정 실패를 사용자 탓으로 돌리지 않는다).
            pose_ok = t.get("poseDetectedRatio", 0.0) >= V.MIN_POSE_RATIO

            g += self._score_gaze(t, V)
            p += self._score_posture(t, V, pose_ok)
            ge += self._score_gesture(t, V, pose_ok)
            e += self._score_expression(t, V)

        n = len(turns)
        return (True, round(g / n), round(ge / n), round(p / n), round(e / n))

    @staticmethod
    def _score_gaze(t: dict, V) -> int:
        """시선: 정면 응시 비율을 선형 매핑하고 두부 흔들림으로 추가 감점."""
        r = t.get("gazeOnTargetRatio", 0.0)
        if r >= V.GAZE_RATIO_FULL:
            s = 10
        elif r <= V.GAZE_RATIO_ZERO:
            s = 0
        else:
            s = round(10 * (r - V.GAZE_RATIO_ZERO)
                      / (V.GAZE_RATIO_FULL - V.GAZE_RATIO_ZERO))

        jitter = math.hypot(t.get("headYawStd", 0.0), t.get("headPitchStd", 0.0))
        s -= int(max(0.0, jitter - V.GAZE_JITTER_TOLERANCE) / V.GAZE_JITTER_STEP)
        return max(0, min(10, s))

    @staticmethod
    def _score_posture(t: dict, V, pose_ok: bool) -> int:
        """자세: 어깨 기울기와 상체 흔들림으로 감점."""
        if not pose_ok:
            return V.UNMEASURABLE_SCORE

        s = 10
        s -= int(max(0.0, t.get("shoulderTiltMean", 0.0)
                     - V.POSTURE_TILT_TOLERANCE) / V.POSTURE_TILT_STEP)
        s -= int(max(0.0, t.get("bodySwayStd", 0.0)
                     - V.POSTURE_SWAY_TOLERANCE) / V.POSTURE_SWAY_STEP)
        # torsoDrift 는 턴이 길수록 커지는 시간 경과 편향이 있어 기본 비활성.
        # (CSV 에는 계속 기록되므로 나중에 분석할 수 있다.)
        if V.POSTURE_DRIFT_ENABLED and t.get("calibrated"):
            s -= int(max(0.0, t.get("torsoDriftMean", 0.0)
                         - V.POSTURE_DRIFT_TOLERANCE) / V.POSTURE_DRIFT_STEP)
        return max(0, min(10, s))

    @staticmethod
    def _score_gesture(t: dict, V, pose_ok: bool) -> int:
        """손짓: 손 사용 빈도가 적정 구간에 있는지 + 굳음/얼굴 만지기 감점."""
        if not pose_ok:
            return V.UNMEASURABLE_SCORE

        u = t.get("handUsageRatio", 0.0)
        if u <= V.GESTURE_USAGE_HARD_ZERO:
            s = V.GESTURE_ZERO_SCORE
        elif u < V.GESTURE_USAGE_MIN:
            s = 10 - int((V.GESTURE_USAGE_MIN - u) / V.GESTURE_UNDER_STEP)
        elif u > V.GESTURE_USAGE_MAX:
            s = 10 - int((u - V.GESTURE_USAGE_MAX) / V.GESTURE_OVER_STEP)
        else:
            s = 10

        # 손이 보이지만 한 자리에 굳어 있으면 감점
        if u >= V.GESTURE_USAGE_MIN and t.get("handExtent", 0.0) < V.GESTURE_EXTENT_MIN:
            s -= V.GESTURE_EXTENT_PENALTY
        s -= max(0, t.get("faceTouchCount", 0) - V.GESTURE_FACE_TOUCH_ALLOWANCE)
        return max(0, min(10, s))

    @staticmethod
    def _score_expression(t: dict, V) -> int:
        """표정: 변화량이 없으면 '굳음' 감점, 찌푸림은 비율만큼 감점, 미소는 보너스."""
        s = 10
        if t.get("expressionVariance", 0.0) < V.EXPRESSION_RIGID_THRESHOLD:
            s -= V.EXPRESSION_RIGID_PENALTY
        s -= int(max(0.0, t.get("frownRatio", 0.0)
                     - V.EXPRESSION_FROWN_TOLERANCE) / V.EXPRESSION_FROWN_STEP)

        # 눈깜빡임: 10fps 에서는 감은 구간(100~150ms)을 놓쳐 신뢰할 수 없어 기본 비활성.
        if V.EXPRESSION_BLINK_ENABLED:
            bpm = t.get("blinkPerMinute", 15.0)
            if bpm > V.EXPRESSION_BLINK_MAX:
                s -= int((bpm - V.EXPRESSION_BLINK_MAX) / V.EXPRESSION_BLINK_STEP)
            elif bpm < V.EXPRESSION_BLINK_MIN:
                s -= 1

        if t.get("smileRatio", 0.0) >= V.EXPRESSION_SMILE_BONUS_RATIO:
            s += 1
        return max(0, min(10, s))