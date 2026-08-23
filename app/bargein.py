"""개입(끼어들기) 판정 계층.

설계 원칙:
  - 이 모듈은 '허가/불허'만 판단한다. 상태 변경과 대사 생성은 session.py 가 한다.
  - 비용이 싼 게이팅부터 순서대로 검사하고, LLM 호출(G7)은 마지막이다.
  - 개입은 희소한 사건이므로 대부분 G5(부정 페르소나)에서 걸러진다.
    이 게이팅이 비용과 지연을 동시에 통제한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import BargeInConfig as B
from .config import settings
from .domain import BargeInReason, BargeInType, Condition, Persona


@dataclass
class BargeInDecision:
    """게이팅 결과 한 건. 호출자는 granted 만 보고 분기하면 된다."""
    granted: bool = False              # 개입을 허가했는가
    bargein_type: str = ""             # 허가 시 유형 "REDIRECT" | "CUTOFF"
    reason: str = ""                   # 허가 시 원인 OFF_TOPIC | LONG_ANSWER | LONG_SILENCE
    advance_stage: bool = False        # 이 개입이 단계를 전진시키는가 (Type B만 True)
    denied_by: str = ""                # 거부한 게이트 이름 (로그/디버그용)
    judge_latency_ms: int = 0          # (Type A) LLM 이탈 판정에 걸린 시간
    meta: dict = field(default_factory=dict)   # 부가 정보 (잘린 발화 길이, 부분 전사 등)


def _deny(gate: str) -> BargeInDecision:
    """어느 게이트에서 막혔는지 이름을 붙여 거부 결정을 만든다."""
    return BargeInDecision(granted=False, denied_by=gate)


# ---------------------------------------------------------------------------
# 공통 게이팅 G1~G5
# ---------------------------------------------------------------------------
def _check_common_gates(session) -> BargeInDecision | None:
    """통과하면 None, 막히면 거부 Decision 을 반환한다."""

    # G1: 실험 조건이 개입 활성(C)인가
    if session.condition != Condition.C.value:
        return _deny("G1_CONDITION")

    # G2: 현재 단계가 개입 대상인가
    if session.stage.value not in B.TARGET_STAGES:
        return _deny("G2_STAGE")

    # G3: 이 단계에서 아직 개입하지 않았는가 (한 답변당·질문당 최대 1회)
    if session.stage.value in session.bargein_used_stages:
        return _deny("G3_STAGE_USED")

    # G4: 세션 총 개입 횟수 < 상한
    if session.bargein_total >= B.MAX_PER_SESSION:
        return _deny("G4_TOTAL_CAP")

    # G4-b: Type A 재답변 중에는 다시 개입하지 않는다
    if session.awaiting_reanswer:
        return _deny("G4B_REANSWER")

    # G5: 현재 페르소나가 부정인가
    if session.persona != Persona.NEGATIVE:
        if not settings.bargein_force_negative:
            return _deny("G5_PERSONA")

    return None


# ---------------------------------------------------------------------------
# Type B — 길이/침묵 (Unity 가 이미 판정했으므로 G1~G5 만)
# ---------------------------------------------------------------------------
async def evaluate_signal(session, reason: str, elapsed: float) -> BargeInDecision:
    denied = _check_common_gates(session)
    if denied:
        print(f"[개입] CUTOFF 거부 ({denied.denied_by}) "
              f"stage={session.stage.value} persona={session.persona.value}")
        return denied

    # G6 은 Unity 로컬 타이머가 곧 유예 검사다(임계 시간 자체가 유예).
    # 다만 LONG_SILENCE 는 유예 개념이 필요하므로 방어적으로 한 번 더 본다.
    if reason == BargeInReason.LONG_SILENCE.value and elapsed < B.GRACE_SEC:
        return _deny("G6_GRACE")

    print(f"[개입] CUTOFF 허가 stage={session.stage.value} "
          f"reason={reason} elapsed={elapsed:.1f}s")
    return BargeInDecision(
        granted=True,
        bargein_type=BargeInType.CUTOFF.value,
        reason=reason,
        advance_stage=True,
        meta={"utterance_elapsed": round(elapsed, 2)},
    )


# ---------------------------------------------------------------------------
# Type A — 주제 이탈 (G1~G6 통과 후 G7 LLM 판정)
# ---------------------------------------------------------------------------
async def evaluate_partial(session, cumulative: str) -> BargeInDecision:
    denied = _check_common_gates(session)
    if denied:
        return denied

    # G6: 유예 — 답변 초반은 거의 항상 주제에서 벗어나 보인다.
    #     "제가 대학교 때 축구 동아리에서..." 는 협업 경험 답변의 도입일 수 있다.
    elapsed = time.time() - (session.utterance_started_at or time.time())
    if elapsed < B.GRACE_SEC:
        return _deny("G6_GRACE_TIME")
    if len(cumulative.replace(" ", "")) < B.MIN_PARTIAL_CHARS:
        return _deny("G6_GRACE_CHARS")

    # G7: LLM 이탈 판정 (AI 담당 L1)
    from . import llm
    t0 = time.perf_counter()
    try:
        off_topic = await llm.judge_off_topic(
            question=session.current_question_text,
            partial_answer=cumulative,
        )
    except Exception as e:
        # L1 이 아직 없어도 백엔드가 죽지 않게 한다. 개입만 보류된다.
        print(f"[개입] 이탈 판정 실패(개입 보류): {e}")
        return _deny("G7_JUDGE_ERROR")
    latency = int((time.perf_counter() - t0) * 1000)

    if not off_topic:
        d = _deny("G7_ON_TOPIC")
        d.judge_latency_ms = latency
        return d

    print(f"[개입] REDIRECT 허가 stage={session.stage.value} "
          f"judge={latency}ms partial='{cumulative[:40]}'")
    return BargeInDecision(
        granted=True,
        bargein_type=BargeInType.REDIRECT.value,
        reason=BargeInReason.OFF_TOPIC.value,
        advance_stage=False,
        judge_latency_ms=latency,
        meta={"partial_text": cumulative, "utterance_elapsed": round(elapsed, 2)},
    )


# ---------------------------------------------------------------------------
# 컷인 반사 명령 (대사 없음. 즉시 발송이 유일한 목적)
# ---------------------------------------------------------------------------
def build_cutin_message(session, decision: BargeInDecision) -> dict:
    """Unity 로 즉시 보낼 컷인 명령. 대사가 없고 '상태 전이 + 표정'만 담는다.

    이 메시지가 실시간 왕복의 전부다. 대사 생성을 기다리지 않고 먼저 나가야
    표정이 음성보다 앞서 바뀐다.
    """
    return {
        "type": "bargein_cutin",
        "session_id": session.session_id,
        "bargein_type": decision.bargein_type,
        "reason": decision.reason,
        "expression_id": B.EXPRESSION_FIRM_STOP,
        "gesture_id": B.GESTURE_BARGEIN,
    }