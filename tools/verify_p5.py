"""A / B / F / G 통합 검증 — 완전 오프라인.

서버, Unity, 마이크, TTS 워커, OpenAI 키가 전혀 필요 없다.
InterviewSession 객체를 직접 조립해 네 가지를 확인한다.

  A  세션 로그 JSON    : 스키마 / 파일 생성 / 중복 방지 / 인덱스 정합
  B  실험 조건 스위치  : 조건 A/B/C 의 페르소나 전이와 개입 게이팅
  F  기준 질문 오염    : 개입 대사가 current_question_text 를 덮지 않는가
  G  피쳐 인덱스 정합  : 무효 턴이 껴도 리스트 길이와 점수가 어긋나지 않는가

사용법:
    cd VRoom_Backend
    python -m tools.verify_p5
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile

from app import bargein, session_log
from app.config import settings
from app.domain import Persona, Stage, persona_from_score, persona_value_from_score
from app.session import InterviewSession

# ---------------------------------------------------------------------------
# 결과 집계
# ---------------------------------------------------------------------------
_results: list[tuple[bool, str, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    _results.append((ok, label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   ({detail})" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * 64)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------
def new_session(condition: str = "C") -> InterviewSession:
    return InterviewSession(session_id="verify", company="넥슨",
                            job_title="게임 클라이언트 프로그래머",
                            resume="Photon 기반 멀티플레이 경험", condition=condition)


VALID = {"speakingTime": 12.0, "meaningfulPauseCount": 2, "averageVolume": 0.11,
         "volumeVariance": 0.0009, "lowVolumeRatio": 0.02, "responseTime": 1.1}
ZERO_ST = dict(VALID, speakingTime=0.0)


def add_turn(s: InterviewSession, features: dict, text: str) -> None:
    """on_user_answer 의 (2) 블록과 동일한 순서로 한 턴을 쌓는다."""
    s.turn_stages.append(s.stage.value)
    s._collect_features(features, text)


FEATURE_LISTS = ("speaking_times", "cps_list", "meaningful_pauses", "volume_variances",
                 "low_volume_ratios", "response_times", "average_volumes")


# ---------------------------------------------------------------------------
# F. 기준 질문 오염
# ---------------------------------------------------------------------------
def verify_f() -> None:
    section("F. current_question_text 오염 수정")

    s = new_session()
    try:
        s._record("interviewer", "Photon 을 선택한 이유를 설명해 주시겠습니까?")
    except TypeError as e:
        check("F-0 _record 호출", False, str(e))
        return

    q = s.current_question_text
    try:
        s._record("interviewer", "네, 거기까지 듣겠습니다.", update_question=False)
    except TypeError as e:
        check("F-0 _record 에 update_question 인자 존재", False, str(e))
        return
    check("F-0 _record 에 update_question 인자 존재", True)
    check("F-1 개입 대사가 기준 질문을 덮지 않는다", s.current_question_text == q,
          f"현재값={s.current_question_text[:24]!r}")

    s._record("interviewer", "그렇다면 병목은 어떻게 처리했습니까?")
    check("F-2 일반 질문은 기준 질문을 갱신한다",
          s.current_question_text == "그렇다면 병목은 어떻게 처리했습니까?")

    s._record("user", "오브젝트 풀링을 썼습니다.")
    check("F-3 사용자 발화는 기준 질문을 건드리지 않는다",
          s.current_question_text == "그렇다면 병목은 어떻게 처리했습니까?")


# ---------------------------------------------------------------------------
# G. 피쳐 인덱스 정합
# ---------------------------------------------------------------------------
def verify_g() -> None:
    section("G. _collect_features 인덱스 정합")

    s = new_session()
    add_turn(s, VALID, "정상 답변입니다. " * 6)
    add_turn(s, {}, "")                      # 워치독 강제 진행 (피쳐 없음)
    add_turn(s, ZERO_ST, "무음")             # speakingTime = 0
    add_turn(s, VALID, "또 정상 답변입니다. " * 6)

    lens = {name: len(getattr(s, name)) for name in FEATURE_LISTS}
    lens["turn_stages"] = len(s.turn_stages)
    if hasattr(s, "turn_features"):
        lens["turn_features"] = len(s.turn_features)
    else:
        check("G-0 turn_features 배열 존재", False, "A 단계 Step 1 누락")

    check("G-1 모든 리스트 길이가 턴 수와 같다", set(lens.values()) == {4}, str(lens))

    # 무효 턴이 채점 결과에 영향을 주지 않아야 한다 (이번 수정의 핵심 성질).
    clean = new_session()
    add_turn(clean, VALID, "정상 답변입니다. " * 6)
    add_turn(clean, VALID, "또 정상 답변입니다. " * 6)

    dirty = new_session()
    add_turn(dirty, VALID, "정상 답변입니다. " * 6)
    add_turn(dirty, {}, "")
    add_turn(dirty, ZERO_ST, "무음")
    add_turn(dirty, VALID, "또 정상 답변입니다. " * 6)

    try:
        a, b = clean._score_voice(), dirty._score_voice()
    except TypeError as e:
        check("G-2 _score_voice() 무인자 시그니처", False, str(e))
        return
    check("G-2 _score_voice() 무인자 시그니처", True)
    check("G-3 무효 턴이 음성 점수를 바꾸지 않는다", a == b, f"clean={a} dirty={b}")

    # 단계별 반응속도 기대값이 올바른 턴에 매칭되는지 (오정합의 대표 증상)
    stages = new_session()
    stages.stage = Stage.SELF_INTRO
    add_turn(stages, VALID, "자기소개입니다. " * 6)
    stages.stage = Stage.TECH_Q1
    add_turn(stages, {}, "")
    stages.stage = Stage.FOLLOWUP_1
    add_turn(stages, VALID, "기술 답변입니다. " * 6)
    check("G-4 turn_stages 가 턴과 1:1 로 대응한다",
          stages.turn_stages == ["SELF_INTRO", "TECH_Q1", "FOLLOWUP_1"],
          str(stages.turn_stages))


# ---------------------------------------------------------------------------
# B. 실험 조건 스위치 (백엔드 절반)
# ---------------------------------------------------------------------------
def verify_b() -> None:
    section("B. 실험 조건 스위치 — 백엔드 게이팅")

    if settings.bargein_force_negative:
        check("B-0 BARGEIN_FORCE_NEGATIVE=false", False,
              "true 이면 G5 가 우회되어 조건 C 가 성립하지 않는다")
    else:
        check("B-0 BARGEIN_FORCE_NEGATIVE=false", True)

    cases = [("A", 20, "NEUTRAL"), ("B", 20, "NEGATIVE"), ("C", 20, "NEGATIVE"),
             ("A", 90, "NEUTRAL"), ("B", 90, "POSITIVE"), ("C", 90, "POSITIVE")]
    ok = all(persona_from_score(sc, 0, c).value == exp for c, sc, exp in cases)
    check("B-1 조건 A 는 점수와 무관하게 NEUTRAL 고정", ok,
          "; ".join(f"{c}/{sc}->{persona_from_score(sc, 0, c).value}" for c, sc, exp in cases))

    check("B-2 조건 A 는 감정 강도 0.0 고정",
          persona_value_from_score(20, 2, "A") == 0.0 and persona_value_from_score(90, 0, "A") == 0.0)
    check("B-3 조건 B/C 는 감정 강도가 움직인다",
          persona_value_from_score(20, 0, "B") < 0 < persona_value_from_score(90, 0, "C"))

    expect = {"A": ("G1_CONDITION", False), "B": ("G1_CONDITION", False), "C": (None, True)}
    for cond, (gate, granted) in expect.items():
        s = new_session(condition=cond)
        s.stage = Stage.FOLLOWUP_1          # G2 통과
        s.persona = Persona.NEGATIVE        # G5 통과
        d = asyncio.run(bargein.evaluate_signal(s, "LONG_ANSWER", 15.0))
        check(f"B-4 조건 {cond} 개입 게이팅",
              d.granted == granted and (granted or d.denied_by == gate),
              f"granted={d.granted} denied_by={d.denied_by}")

    s = new_session(condition="C")
    s.stage = Stage.FOLLOWUP_1
    s.persona = Persona.NEUTRAL
    d = asyncio.run(bargein.evaluate_signal(s, "LONG_ANSWER", 15.0))
    check("B-5 조건 C 라도 페르소나가 중립이면 거부(G5)",
          not d.granted and d.denied_by == "G5_PERSONA", f"denied_by={d.denied_by}")


# ---------------------------------------------------------------------------
# A. 세션 로그 JSON
# ---------------------------------------------------------------------------
def verify_a() -> None:
    section("A. 세션 로그 JSON")

    tmp = tempfile.mkdtemp(prefix="vroom_log_")
    original_dir = settings.session_log_dir
    settings.session_log_dir = tmp
    try:
        s = new_session(condition="B")
        s._record("interviewer", "자기소개를 부탁드립니다.")
        add_turn(s, VALID, "유재명입니다. " * 6)
        s._record("user", "유재명입니다.")
        s.stage_scores.append(("TECH_Q1", 40))
        s.vision_turns.append({"stage": "TECH_Q1", "phase": "NORMAL", "gazeRatio": 0.8})
        s.stage = Stage.DONE

        path = session_log.dump(s, None, exit_reason="normal")
        check("A-1 로그 파일이 생성된다", bool(path) and os.path.exists(path), str(path))
        if not path:
            return

        check("A-2 파일명이 조건으로 시작한다",
              os.path.basename(path).startswith("B_"), os.path.basename(path))

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        check("A-3 JSON 으로 파싱된다", True)

        for key in ("schema_version", "meta", "gate_config", "turns",
                    "turn_features", "stage_scores", "vision_turns", "bargein"):
            check(f"A-4 최상위 키 '{key}'", key in data)

        meta = data.get("meta", {})
        for key in ("session_id", "condition", "exit_reason", "force_negative",
                    "llm_model", "started_at", "ended_at", "duration_sec", "final_stage"):
            check(f"A-5 meta.{key}", key in meta)
        check("A-6 meta.condition 이 세션 조건과 일치", meta.get("condition") == "B",
              str(meta.get("condition")))

        check("A-7 로그 안에서도 인덱스가 정합",
              len(data.get("turn_features", [])) == len(data.get("turn_stages", [])),
              f"features={len(data.get('turn_features', []))} "
              f"stages={len(data.get('turn_stages', []))}")

        again = session_log.dump(s, None, exit_reason="normal")
        check("A-8 중복 저장이 방지된다", again is None)

        before = len(os.listdir(tmp))
        s2 = new_session(condition="C")
        add_turn(s2, VALID, "중간에 끊긴 세션")
        session_log.dump(s2, None, exit_reason="disconnect")
        after = os.listdir(tmp)
        check("A-9 비정상 종료도 저장된다", len(after) == before + 1)
        disc = [n for n in after if n.startswith("C_")]
        if disc:
            with open(os.path.join(tmp, disc[0]), encoding="utf-8") as f:
                d2 = json.load(f)
            check("A-10 exit_reason=disconnect 기록",
                  d2["meta"]["exit_reason"] == "disconnect")
            check("A-11 비정상 종료 시 feedback 은 null", d2.get("feedback") is None)

        print(f"\n  (임시 로그 디렉터리: {tmp})")
    finally:
        settings.session_log_dir = original_dir


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 64)
    print("VRoom P5 오프라인 검증 (A / B / F / G)")
    print("=" * 64)

    verify_f()
    verify_g()
    verify_b()
    verify_a()

    failed = [r for r in _results if not r[0]]
    print("\n" + "=" * 64)
    print(f"총 {len(_results)}건 중 {len(_results) - len(failed)}건 통과, {len(failed)}건 실패")
    for _, label, detail in failed:
        print(f"  - {label}" + (f"   ({detail})" if detail else ""))
    print("=" * 64)


if __name__ == "__main__":
    main()
