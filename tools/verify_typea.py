"""Type A(REDIRECT) 개입 로직 오프라인 검증기.

verify_p5.py 가 조건 스위치(G1)와 Type B 게이팅을 검증한다면,
이 파일은 **Type A 의 유일한 치명적 실패 모드**를 검증한다.

    잘린 답변과 재답변의 분리 실패 -> 단계가 전진하지 않거나 두 칸 전진

LLM / TTS / Unity / STT 없이 순수 파이썬으로 돌린다. 네트워크를 타지 않으므로
실기 테스트 전에 몇 초 만에 회귀를 잡을 수 있다.

    python -m tools.verify_typea        (VRoom_Backend 루트에서)
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm as llm_module  # noqa: E402
from app.config import BargeInConfig as B  # noqa: E402
from app.domain import LLMTurn, Persona, Stage  # noqa: E402
from app.session import InterviewSession  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((PASS if ok else FAIL, name, detail))


# ---------------------------------------------------------------------------
# LLM 스텁 — 네트워크를 타지 않고 점수만 결정적으로 돌려준다
# ---------------------------------------------------------------------------
class LLMStub:
    """호출 횟수와 인자를 기록해 '무엇을 기준으로 채점했는가'까지 검증한다."""

    def __init__(self):
        self.score_answer_calls: list[dict] = []
        self.generate_turn_calls: list[dict] = []
        self.next_answer_score = 30      # 잘린 답변 점수
        self.next_turn_score = 70        # 재답변 점수

    async def score_answer(self, *, question: str, answer: str) -> int:
        self.score_answer_calls.append({"question": question, "answer": answer})
        return self.next_answer_score

    async def generate_turn(self, **kw) -> LLMTurn:
        self.generate_turn_calls.append(kw)
        return LLMTurn(dialogue="다음 질문입니다.", score=self.next_turn_score,
                       expression_id=0, gesture_id=0)

    async def extract_info(self, *a, **kw):
        from app.domain import ExtractedInfo
        return ExtractedInfo(company_name="테스트", job_role="백엔드")

    @staticmethod
    def _clamp_to_set(turn, persona):
        return turn


def install_stub() -> LLMStub:
    stub = LLMStub()
    for fn in ("score_answer", "generate_turn", "extract_info"):
        setattr(llm_module, fn, getattr(stub, fn))
    llm_module._clamp_to_set = LLMStub._clamp_to_set
    return stub


# ---------------------------------------------------------------------------
# 개입 결정 더미
# ---------------------------------------------------------------------------
def redirect_decision(partial_text: str = "그런데 어제 축구 경기 보셨어요?"):
    d = types.SimpleNamespace()
    d.granted = True
    d.bargein_type = "REDIRECT"
    d.reason = "OFF_TOPIC"
    d.advance_stage = False
    d.judge_latency_ms = 420
    d.meta = {"partial_text": partial_text, "utterance_elapsed": 12.3}
    return d


def make_session(stage: Stage = Stage.FOLLOWUP_1) -> InterviewSession:
    s = InterviewSession("verify", company="테스트", job_title="백엔드", condition="C")
    s.stage = stage
    s.persona = Persona.NEGATIVE
    s.current_question_text = "팀 프로젝트에서 갈등을 해결한 경험을 말씀해 주세요."
    s._info_extracted = True
    return s


FEATURES = {"speakingTime": 9.0, "pauseCount": 1, "averageVolume": 0.02,
            "volumeVariance": 0.001, "lowVolumeRatio": 0.1, "responseTime": 1.2}


# ---------------------------------------------------------------------------
# T1. 정상 경로 — 잘린 답변(truncated=True) 다음 재답변(truncated=False)
# ---------------------------------------------------------------------------
async def t1_normal_flow(stub: LLMStub):
    s = make_session()
    start_stage = s.stage
    s.commit_bargein(redirect_decision())

    p1 = await s.on_user_answer("팀 프로젝트라고 하면 저는", FEATURES, truncated=True)
    check("T1-1 잘린 답변은 대사를 만들지 않는다", p1.type == "ignored", p1.type)
    check("T1-2 잘린 답변은 단계를 전진시키지 않는다", s.stage == start_stage, s.stage.value)
    check("T1-3 잘린 답변이 채점되어 보관된다", s.pending_truncated_score == 30,
          str(s.pending_truncated_score))
    check("T1-4 채점 기준 질문이 개입 대사가 아닌 원본 질문이다",
          stub.score_answer_calls[-1]["question"].startswith("팀 프로젝트"),
          stub.score_answer_calls[-1]["question"][:20])
    check("T1-5 잘린 답변은 음성 피쳐를 쌓지 않는다", len(s.turn_stages) == 0,
          str(len(s.turn_stages)))

    p2 = await s.on_user_answer("갈등이 있었을 때 저는 회의를 열어", FEATURES, truncated=False)
    check("T1-6 재답변은 REDIRECT_REANSWER 로 태깅된다",
          p2.bargein_type == "REDIRECT_REANSWER", p2.bargein_type)
    check("T1-7 재답변에서 단계가 정확히 한 칸 전진한다",
          s.stage == Stage.FOLLOWUP_2, s.stage.value)
    expected = round(B.W_TRUNCATED * 30 + B.W_REANSWER * 70)
    check(f"T1-8 점수가 {B.W_TRUNCATED}:{B.W_REANSWER} 로 혼합된다({expected})",
          p2.score == expected, str(p2.score))
    check("T1-9 개입 로그에 두 점수가 각각 보존된다",
          s.bargein_log[-1]["score_truncated"] == 30
          and s.bargein_log[-1]["score_reanswer"] == 70,
          str(s.bargein_log[-1]))
    check("T1-10 재답변 후 대기 상태가 해제된다",
          not s.awaiting_reanswer and not s.truncated_captured)


# ---------------------------------------------------------------------------
# T2. 잘린 전사 유실 — stt_skip 으로 truncated 전사가 아예 안 온 경우
#     (Type A 의 유일한 치명적 실패 모드)
# ---------------------------------------------------------------------------
async def t2_missing_truncated(stub: LLMStub):
    s = make_session()
    s.commit_bargein(redirect_decision("그런데 어제 축구 경기 보셨어요? 후반전에"))

    before = len(stub.score_answer_calls)
    p = await s.on_user_answer("다시 말씀드리면 저희 팀은", FEATURES, truncated=False)

    check("T2-1 재답변이 잘린 답변으로 흡수되지 않는다", p.type != "ignored", p.type)
    check("T2-2 단계가 한 칸 전진한다", s.stage == Stage.FOLLOWUP_2, s.stage.value)
    check("T2-3 부분 전사로 대체 채점이 일어난다",
          len(stub.score_answer_calls) == before + 1,
          f"{before} -> {len(stub.score_answer_calls)}")
    check("T2-4 대체 채점 대상이 부분 전사다",
          "축구" in stub.score_answer_calls[-1]["answer"],
          stub.score_answer_calls[-1]["answer"][:30])
    check("T2-5 로그에 대체 채점 출처가 남는다",
          s.bargein_log[-1].get("truncated_source") == "partial_fallback",
          str(s.bargein_log[-1].get("truncated_source")))


# ---------------------------------------------------------------------------
# T3. 부분 전사조차 없는 최악의 경우 — 재답변 점수만 쓴다
# ---------------------------------------------------------------------------
async def t3_no_partial(stub: LLMStub):
    s = make_session()
    s.commit_bargein(redirect_decision(""))

    p = await s.on_user_answer("다시 답변드리면", FEATURES, truncated=False)
    check("T3-1 단계가 멈추지 않는다", s.stage == Stage.FOLLOWUP_2, s.stage.value)
    check("T3-2 재답변 점수를 그대로 쓴다", p.score == 70, str(p.score))


# ---------------------------------------------------------------------------
# T4. 워치독 — 재답변이 오기 전에 타임아웃이 먼저 돈 경우
# ---------------------------------------------------------------------------
async def t4_watchdog(stub: LLMStub):
    s = make_session()
    s.commit_bargein(redirect_decision("어제 축구 경기 후반전에"))

    await s._absorb_missing_truncated()          # 워치독이 부르는 그 함수
    check("T4-1 워치독 후 truncated_captured 가 선다", s.truncated_captured)

    p = await s.on_user_answer("다시 답변드리면", FEATURES, truncated=False)
    check("T4-2 워치독 후 도착한 답변은 재답변으로 처리된다",
          p.bargein_type == "REDIRECT_REANSWER", p.bargein_type)
    check("T4-3 이중 대체 채점이 일어나지 않는다",
          len([c for c in stub.score_answer_calls
               if "축구" in c["answer"]]) == 1)


# ---------------------------------------------------------------------------
# T5. 인덱스 정합 — 잘린 답변이 turn_stages 를 오염시키지 않는가
# ---------------------------------------------------------------------------
async def t5_index_alignment(stub: LLMStub):
    s = make_session(Stage.TECH_Q1)
    await s.on_user_answer("일반 답변입니다.", FEATURES)          # 개입 없는 턴
    s.stage = Stage.FOLLOWUP_1
    s.commit_bargein(redirect_decision())
    await s.on_user_answer("잘린 답변", FEATURES, truncated=True)
    await s.on_user_answer("재답변입니다.", FEATURES, truncated=False)

    lens = {
        "turn_stages": len(s.turn_stages),
        "speaking_times": len(s.speaking_times),
        "cps_list": len(s.cps_list),
        "meaningful_pauses": len(s.meaningful_pauses),
        "turn_features": len(s.turn_features),
    }
    check("T5-1 모든 피쳐 리스트 길이가 같다", len(set(lens.values())) == 1, str(lens))
    check("T5-2 채점 턴 수 = 일반 1 + 재답변 1", lens["turn_stages"] == 2, str(lens))


# ---------------------------------------------------------------------------
# T6. 게이팅 — 재답변 중 재개입 금지(G4-b), 같은 단계 재개입 금지(G3)
# ---------------------------------------------------------------------------
async def t6_gates(stub: LLMStub):
    from app import bargein

    import time as _t

    s = make_session()
    s.utterance_started_at = _t.time() - 60           # 유예는 이미 지난 것으로
    s.commit_bargein(redirect_decision())

    # G3 를 비워 G4-b 만 남긴다. 실제로는 G3 가 먼저 잡지만, 재답변 중
    # 다른 단계로 넘어간 뒤에도 막히는지가 여기서 확인할 지점이다.
    used = set(s.bargein_used_stages)
    s.bargein_used_stages.clear()
    d = await bargein.evaluate_partial(s, "가" * 50)
    check("T6-1 재답변 대기 중에는 재개입하지 않는다(G4-b)",
          not d.granted and d.denied_by == "G4B_REANSWER", d.denied_by)
    s.bargein_used_stages.update(used)

    s.awaiting_reanswer = False
    d = await bargein.evaluate_partial(s, "가" * 50)
    check("T6-2 같은 단계에서 두 번 개입하지 않는다(G3)",
          not d.granted and d.denied_by == "G3_STAGE_USED", d.denied_by)

    s2 = make_session()
    s2.utterance_started_at = _t.time() - 60
    s2.condition = "B"
    d = await bargein.evaluate_partial(s2, "가" * 50)
    check("T6-3 조건 B 는 G1 에서 전부 차단된다",
          not d.granted and d.denied_by == "G1_CONDITION", d.denied_by)

    s3 = make_session(Stage.CLOSING)
    s3.utterance_started_at = _t.time() - 60
    d = await bargein.evaluate_partial(s3, "가" * 50)
    check("T6-4 마무리 단계는 개입 대상이 아니다(G2)",
          not d.granted and d.denied_by == "G2_STAGE", d.denied_by)

    s4 = make_session()
    s4.utterance_started_at = _t.time() - 60
    s4.persona = Persona.NEUTRAL
    d = await bargein.evaluate_partial(s4, "가" * 50)
    check("T6-5 중립 페르소나는 개입하지 않는다(G5)",
          not d.granted and d.denied_by == "G5_PERSONA", d.denied_by)

    s5 = make_session()
    s5.utterance_started_at = _t.time() - 60
    d = await bargein.evaluate_partial(s5, "짧음")
    check("T6-6 부분 전사가 짧으면 판정하지 않는다(G6)",
          not d.granted and d.denied_by == "G6_GRACE_CHARS", d.denied_by)


async def main() -> int:
    stub = install_stub()
    from app.config import settings
    settings.bargein_force_negative = False

    for fn in (t1_normal_flow, t2_missing_truncated, t3_no_partial,
               t4_watchdog, t5_index_alignment, t6_gates):
        stub.score_answer_calls.clear()
        stub.generate_turn_calls.clear()
        await fn(stub)

    print("\n" + "=" * 78)
    print("Type A (REDIRECT) 오프라인 검증")
    print("=" * 78)
    failed = 0
    for status, name, detail in _results:
        mark = "  OK  " if status == PASS else " FAIL "
        line = f"[{mark}] {name}"
        if status == FAIL and detail:
            line += f"   <- {detail}"
        print(line)
        failed += status == FAIL
    print("-" * 78)
    print(f"총 {len(_results)}건 / 실패 {failed}건")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
