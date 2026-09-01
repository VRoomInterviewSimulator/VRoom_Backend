"""본실험 로그를 읽어 최종 보고서용 표를 만든다.

summarize_logs.py 가 '수집이 잘 됐는지'를 보는 도구라면, 이 스크립트는
'무엇을 주장할 수 있는지'를 만드는 도구다. 표는 그대로 보고서에 옮길 수 있는
형태로 출력하고, 같은 내용을 CSV 로도 떨어뜨린다.

  표 1  세션 목록 및 유효성
  표 2  조작 확인 (manipulation check) — 조건이 의도대로 작동했는가
  표 3  개입 성능 — 지연 / 양보 / 잘린 길이
  표 4  단계별 · 조건 간 웹캠 지표          ← 핵심 표
  표 5  개입 직후 반응 (REACTION vs 무개입 기준선)
  표 6  채점 및 종합 점수

표 4·5 를 조건 C 내부(NORMAL vs TRUNCATED)로 만들면 안 된다.
개입이 일어난 단계에는 NORMAL 턴이 정의상 존재하지 않아, 단계 차이를
위상 차이로 오인하게 된다. 반드시 '같은 단계를 조건 간에' 비교한다.

사용법:
    cd VRoom_Backend
    python -m tools.analyze_experiment                # logs/ 를 읽는다
    python -m tools.analyze_experiment data/final     # 다른 폴더
    python -m tools.analyze_experiment --csv          # CSV 도 생성
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys

DEFAULT_DIR = "logs"
CONDITIONS = ("A", "B", "C")
COND_LABEL = {"A": "A 정적", "B": "B 가변", "C": "C 개입"}

# 면접 단계. 조건 간 비교는 반드시 같은 단계끼리 한다.
STAGES = ("SELF_INTRO", "TECH_Q1", "FOLLOWUP_1", "FOLLOWUP_2", "BEHAVIORAL", "CLOSING")
# 개입 대상 단계 (BargeInConfig.TARGET_STAGES 와 동일)
TARGET_STAGES = ("TECH_Q1", "FOLLOWUP_1", "FOLLOWUP_2", "BEHAVIORAL")

PHASES = ("NORMAL", "TRUNCATED", "REACTION", "REANSWER")

# vision_process/aggregator.py:191-213 의 end_turn() 반환 키와 1:1
VISION_CORE = (
    "gazeOnTargetRatio",   # 시선이 면접관을 향한 프레임 비율
    "headYawStd",          # 좌우 두리번거림
    "bodySwayStd",         # 상체 흔들림
    "handUsageRatio",      # 손을 쓴 프레임 비율
    "faceTouchCount",      # 얼굴 만짐 횟수
    "expressionVariance",  # 표정 변화량
)
VISION_ALL = VISION_CORE + ("shoulderTiltMean", "headPitchStd", "handExtent",
                            "smileRatio", "frownRatio")
VISION_META = ("durationSec", "frameCount", "faceDetectedRatio",
               "poseDetectedRatio", "calibrated")
# 폐기 지표: torsoDriftMean(경과시간 편향) / blinkPerMinute(10fps 한계) / handMotionEnergy(대체됨)

MIN_FACE_RATIO = 0.5   # 얼굴 검출이 이보다 낮은 턴은 지표를 신뢰할 수 없다
LINE = "=" * 108
THIN = "-" * 108


# ---------------------------------------------------------------------------
# 로딩 · 유효성
# ---------------------------------------------------------------------------
def load_all(directory: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.basename(p).startswith(("summary_", "analysis_")):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            d["_file"] = os.path.basename(p)
            out.append(d)
        except Exception as e:
            print(f"  ! 읽기 실패 {os.path.basename(p)}: {e}")
    return out


def flags(s: dict) -> list[str]:
    """분석에서 빼야 할 이유. 비어 있으면 유효한 세션이다."""
    f = []
    m = s.get("meta", {})
    if m.get("force_negative"):
        f.append("FORCE_NEG")        # G5 우회 — 조건 C 가 성립하지 않음
    if m.get("condition") not in CONDITIONS:
        f.append("NO_COND")
    if m.get("exit_reason") != "normal":
        f.append("ABORTED")
    if s.get("feedback") is None:
        f.append("NO_REPORT")
    if len(s.get("turn_features", [])) != len(s.get("turn_stages", [])):
        f.append("MISALIGNED")      # 인덱스 정합 붕괴
    return f


def valid(sessions):
    return [s for s in sessions if not flags(s)]


def vturns(s: dict) -> list[dict]:
    """얼굴 검출이 충분한 웹캠 턴만."""
    return [t for t in s.get("vision_turns", [])
            if t.get("faceDetectedRatio", 0.0) >= MIN_FACE_RATIO]


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.fmean(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.stdev(xs) if len(xs) > 1 else None


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def ms(v, sdv=None, nd=3):
    """평균(표준편차) 형태. 보고서 표에 그대로 옮길 수 있다."""
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if sdv is None else f"{v:.{nd}f} ({sdv:.{nd}f})"


def short(k, w=18):
    return k if len(k) <= w else k[:w - 1] + "."


def by_cond(sessions):
    return {c: [s for s in sessions if s.get("meta", {}).get("condition") == c]
            for c in CONDITIONS}


# ---------------------------------------------------------------------------
# 표 1 — 세션 목록
# ---------------------------------------------------------------------------
def t1_sessions(sessions):
    print("\n[표 1] 세션 목록 및 유효성")
    print(THIN)
    print(f"{'파일':<38}{'조건':>4}{'길이(s)':>9}{'턴':>4}{'개입':>5}{'웹캠턴':>7}{'종합':>6}   비고")
    print(THIN)
    for s in sessions:
        m, fb = s.get("meta", {}), (s.get("feedback") or {})
        fl = flags(s)
        print(f"{s['_file']:<38}{m.get('condition','?'):>4}"
              f"{fmt(m.get('duration_sec'),1):>9}{len(s.get('turn_stages',[])):>4}"
              f"{s.get('bargein',{}).get('total',0):>5}{len(vturns(s)):>7}"
              f"{fmt(fb.get('overall_score'),0):>6}   {','.join(fl) if fl else 'OK'}")
    print(THIN)
    bad = [s for s in sessions if flags(s)]
    if bad:
        print(f"  유효 {len(sessions)-len(bad)} / 전체 {len(sessions)} — 무효 세션은 이후 표에서 제외한다.")


# ---------------------------------------------------------------------------
# 표 2 — 조작 확인
# ---------------------------------------------------------------------------
def t2_manipulation(sessions):
    print("\n[표 2] 조작 확인 (manipulation check) — 조건이 의도대로 작동했는가")
    print(THIN)
    print(f"{'조건':<10}{'N':>4}{'개입/세션':>11}{'개입 세션 비율':>15}"
          f"{'페르소나 변동':>15}{'최장 답변(s)':>14}{'answerLength':>14}")
    print(THIN)

    g = by_cond(valid(sessions))
    for c in CONDITIONS:
        grp = g[c]
        if not grp:
            print(f"{COND_LABEL[c]:<10}{0:>4}   (세션 없음)")
            continue

        totals = [s.get("bargein", {}).get("total", 0) for s in grp]
        with_bi = sum(1 for t in totals if t > 0)

        # 페르소나가 실제로 움직였는가 — 개입 시점 강도의 절대값 평균으로 근사
        moved = []
        for s in grp:
            vals = [abs(e.get("persona_value_at_trigger", 0.0) or 0.0)
                    for e in s.get("bargein", {}).get("events", [])]
            moved.extend(vals)

        longest = [max([tf.get("speakingTime") or 0 for tf in s.get("turn_features", [])] or [0])
                   for s in grp]
        alen = [(s.get("feedback") or {}).get("scores", {}).get("answerLength") for s in grp]

        print(f"{COND_LABEL[c]:<10}{len(grp):>4}{fmt(mean(totals)):>11}"
              f"{f'{with_bi}/{len(grp)}':>15}"
              f"{(fmt(mean(moved)) if moved else '해당 없음'):>15}"
              f"{fmt(mean(longest),1):>14}{fmt(mean(alen),1):>14}")
    print(THIN)
    print("  * 조건 C 의 '개입 세션 비율'이 1.00 이 아니면 그 세션은 조건 B 와 구분되지 않는다.")
    print("  * answerLength 는 평가 지표가 아니라 조작의 부수 효과다. 조건 간 비교에 쓰지 않는다.")

    types, reasons = {}, {}
    for s in valid(sessions):
        for e in s.get("bargein", {}).get("events", []):
            types[e.get("type", "?")] = types.get(e.get("type", "?"), 0) + 1
            reasons[e.get("reason", "?")] = reasons.get(e.get("reason", "?"), 0) + 1
    if types:
        print(f"  개입 유형: {types}    개입 사유: {reasons}")


# ---------------------------------------------------------------------------
# 표 3 — 개입 성능
# ---------------------------------------------------------------------------
def t3_performance(sessions):
    ev = [e for s in valid(sessions) if s.get("meta", {}).get("condition") == "C"
          for e in s.get("bargein", {}).get("events", [])]
    print("\n[표 3] 개입 성능 지표 (조건 C)")
    print(THIN)
    if not ev:
        print("  조건 C 의 개입 이벤트가 없다.")
        print(THIN)
        return

    rows = [
        ("첫 발성까지 지연 (ms)", [e.get("latency_to_speech_ms") for e in ev], 0),
        ("개입 대사 완료 (ms)", [e.get("latency_full_ms") for e in ev], 0),
        ("양보 시간 (s)", [e.get("yield_time") for e in ev], 2),
        ("잘린 발화 길이 (s)", [e.get("utterance_elapsed") for e in ev], 1),
        ("개입 시각 (s, 면접 시작 기준)", [e.get("triggered_at") for e in ev], 1),
    ]
    print(f"{'지표':<32}{'N':>5}{'평균':>12}{'표준편차':>12}{'최소':>10}{'최대':>10}")
    print(THIN)
    for name, xs, nd in rows:
        v = [x for x in xs if isinstance(x, (int, float))]
        if not v:
            print(f"{name:<32}{0:>5}{'-':>12}{'-':>12}{'-':>10}{'-':>10}")
            continue
        print(f"{name:<32}{len(v):>5}{fmt(mean(v),nd):>12}{fmt(sd(v),nd):>12}"
              f"{fmt(min(v),nd):>10}{fmt(max(v),nd):>10}")
    print(THIN)
    print("  * 양보 시간은 LONG_SILENCE 개입에서는 정의되지 않아 N 이 더 작다.")


# ---------------------------------------------------------------------------
# 표 4 — 단계별 · 조건 간 웹캠 지표  (핵심)
# ---------------------------------------------------------------------------
def _stage_rows(sessions, stage):
    """(조건, 위상) 별로 해당 단계의 웹캠 턴을 모은다."""
    buckets: dict[tuple, list[dict]] = {}
    for s in valid(sessions):
        c = s.get("meta", {}).get("condition")
        for t in vturns(s):
            if t.get("stage") != stage:
                continue
            buckets.setdefault((c, t.get("phase", "NORMAL")), []).append(t)
    return buckets


def t4_stage_by_condition(sessions, metric="gazeOnTargetRatio"):
    print(f"\n[표 4] 단계별 · 조건 간 웹캠 지표 — {metric}")
    print("   같은 단계를 조건 간에 비교한다. 조건 C 내부(NORMAL vs TRUNCATED) 비교는")
    print("   개입 단계에 NORMAL 턴이 없어 단계 차이를 위상 차이로 오인하게 된다.")
    print(THIN)
    print(f"{'단계':<14}{'A / NORMAL':>16}{'B / NORMAL':>16}"
          f"{'C / TRUNCATED':>16}{'C / REACTION':>16}{'C / NORMAL':>16}")
    print(THIN)
    for stage in STAGES:
        b = _stage_rows(sessions, stage)
        def cellv(c, ph):
            rows = b.get((c, ph), [])
            v = mean([r.get(metric) for r in rows])
            return "-" if v is None else f"{v:.3f} (n={len(rows)})"
        print(f"{stage:<14}{cellv('A','NORMAL'):>16}{cellv('B','NORMAL'):>16}"
              f"{cellv('C','TRUNCATED'):>16}{cellv('C','REACTION'):>16}{cellv('C','NORMAL'):>16}")
    print(THIN)
    print("  * 개입 대상 단계: " + ", ".join(TARGET_STAGES))
    print("  * 조건 C 의 개입 단계에는 NORMAL 턴이 없는 것이 정상이다.")


# ---------------------------------------------------------------------------
# 표 5 — 개입 직후 반응
# ---------------------------------------------------------------------------
def t5_reaction(sessions):
    """개입 대상 단계에 한정해, 무개입 기준선(A·B의 NORMAL)과 조건 C 위상을 비교."""
    print("\n[표 5] 개입 직후 반응 — 무개입 기준선 대비 (개입 대상 단계에 한정)")
    print(THIN)

    base, trunc, react = {k: [] for k in VISION_CORE}, {k: [] for k in VISION_CORE}, {k: [] for k in VISION_CORE}
    nb = nt = nr = 0
    for s in valid(sessions):
        c = s.get("meta", {}).get("condition")
        for t in vturns(s):
            if t.get("stage") not in TARGET_STAGES:
                continue
            ph = t.get("phase", "NORMAL")
            tgt = None
            if c in ("A", "B") and ph == "NORMAL":
                tgt, _ = base, None
                nb += 1
            elif c == "C" and ph == "TRUNCATED":
                tgt = trunc
                nt += 1
            elif c == "C" and ph == "REACTION":
                tgt = react
                nr += 1
            if tgt is None:
                continue
            for k in VISION_CORE:
                if isinstance(t.get(k), (int, float)):
                    tgt[k].append(t[k])

    if nb == 0:
        print("  무개입 기준선(A·B) 턴이 없어 비교할 수 없다.")
        print(THIN)
        return

    print(f"{'지표':<22}{'기준선 A·B':>16}{'C TRUNCATED':>16}{'C REACTION':>16}"
          f"{'Δ TRUNCATED':>15}{'Δ REACTION':>15}")
    print(f"{'':<22}{f'(n={nb})':>16}{f'(n={nt})':>16}{f'(n={nr})':>16}{'':>15}{'':>15}")
    print(THIN)
    for k in VISION_CORE:
        b, t, r = mean(base[k]), mean(trunc[k]), mean(react[k])
        dt = f"{t-b:+.3f}" if (t is not None and b is not None) else "-"
        dr = f"{r-b:+.3f}" if (r is not None and b is not None) else "-"
        print(f"{short(k,22):<22}{ms(b, sd(base[k])):>16}{ms(t, sd(trunc[k])):>16}"
              f"{ms(r, sd(react[k])):>16}{dt:>15}{dr:>15}")
    print(THIN)
    print("  * 기준선은 조건 A·B 의 같은 단계 NORMAL 턴이다. 개입이 없었을 때의 행동을 뜻한다.")
    print("  * Δ REACTION 이 이 연구가 보려는 '개입의 즉각 효과'다.")
    print("  * 괄호 안은 표준편차. 표본이 작으면 평균만으로 결론 내지 말 것.")


# ---------------------------------------------------------------------------
# 표 6 — 채점 및 종합 점수
# ---------------------------------------------------------------------------
def t6_scores(sessions):
    print("\n[표 6] 채점 및 종합 점수")
    print(THIN)
    g = by_cond(valid(sessions))
    keys = ("gaze", "gesture", "posture", "expression", "voiceVolume",
            "voiceSpeed", "fillerWords", "accuracy", "responseTime")
    print(f"{'조건':<10}{'N':>4}{'단계 평균':>11}" + "".join(f"{k[:9]:>11}" for k in keys) + f"{'종합':>8}")
    print(THIN)
    for c in CONDITIONS:
        grp = g[c]
        if not grp:
            continue
        stage_scores = [r.get("score") for s in grp for r in s.get("stage_scores", [])
                        if isinstance(r.get("score"), int) and r["score"] >= 0]
        row = f"{COND_LABEL[c]:<10}{len(grp):>4}{fmt(mean(stage_scores),1):>11}"
        for k in keys:
            row += f"{fmt(mean([(s.get('feedback') or {}).get('scores',{}).get(k) for s in grp]),1):>11}"
        row += f"{fmt(mean([(s.get('feedback') or {}).get('overall_score') for s in grp]),1):>8}"
        print(row)
    print(THIN)
    print("  * answerLength 는 조작의 부수 효과이므로 이 표에서 제외했다 (표 2 참조).")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def export(sessions, directory):
    # (1) 세션 단위
    p1 = os.path.join(directory, "analysis_sessions.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "condition", "valid", "flags", "duration_sec", "turns",
                    "bargein_total", "mean_latency_ms", "mean_yield_s",
                    "mean_stage_score", "overall_score"])
        for s in sessions:
            m, fb = s.get("meta", {}), (s.get("feedback") or {})
            ev = s.get("bargein", {}).get("events", [])
            sc = [r.get("score") for r in s.get("stage_scores", [])
                  if isinstance(r.get("score"), int) and r["score"] >= 0]
            w.writerow([s["_file"], m.get("condition"), int(not flags(s)),
                        "|".join(flags(s)), m.get("duration_sec"),
                        len(s.get("turn_stages", [])),
                        s.get("bargein", {}).get("total", 0),
                        fmt(mean([e.get("latency_to_speech_ms") for e in ev]), 0),
                        fmt(mean([e.get("yield_time") for e in ev]), 2),
                        fmt(mean(sc), 1), fb.get("overall_score")])

    # (2) 웹캠 턴 단위 — 롱 포맷. 통계 도구에서 바로 돌릴 수 있다.
    p2 = os.path.join(directory, "analysis_vision_turns.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "condition", "stage", "phase", "is_target_stage",
                    *VISION_META, *VISION_ALL])
        for s in valid(sessions):
            c = s.get("meta", {}).get("condition")
            for t in vturns(s):
                w.writerow([s["_file"], c, t.get("stage"), t.get("phase"),
                            int(t.get("stage") in TARGET_STAGES),
                            *[t.get(k) for k in VISION_META],
                            *[t.get(k) for k in VISION_ALL]])

    # (3) 개입 이벤트 단위
    p3 = os.path.join(directory, "analysis_bargein_events.csv")
    with open(p3, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        cols = ["stage", "type", "reason", "triggered_at", "persona_value_at_trigger",
                "utterance_elapsed", "yield_time", "score_truncated", "score_reanswer",
                "score_final", "latency_to_speech_ms", "latency_full_ms"]
        w.writerow(["file", "condition", *cols])
        for s in valid(sessions):
            c = s.get("meta", {}).get("condition")
            for e in s.get("bargein", {}).get("events", []):
                w.writerow([s["_file"], c, *[e.get(k) for k in cols]])

    print(f"\nCSV 생성:\n  {p1}\n  {p2}\n  {p3}")


# ---------------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_csv = "--csv" in sys.argv
    directory = args[0] if args else DEFAULT_DIR
    metric = "gazeOnTargetRatio"
    for a in sys.argv[1:]:
        if a.startswith("--metric="):
            metric = a.split("=", 1)[1]

    if not os.path.isdir(directory):
        print(f"디렉터리를 찾을 수 없다: {directory}")
        return

    sessions = load_all(directory)
    print(LINE)
    print(f"VRoom 본실험 분석 — {directory} ({len(sessions)}개 세션)")
    print(LINE)
    if not sessions:
        print("JSON 파일이 없다.")
        return

    seen = {k for s in sessions for t in s.get("vision_turns", []) for k in t}
    missing = [k for k in VISION_CORE if seen and k not in seen]
    if missing:
        print(f"! 로그에 없는 웹캠 키: {missing}")
        print("  vision_process/aggregator.py 의 end_turn() 반환 키와 대조할 것")

    t1_sessions(sessions)
    t2_manipulation(sessions)
    t3_performance(sessions)
    t4_stage_by_condition(sessions, metric)
    t5_reaction(sessions)
    t6_scores(sessions)

    if want_csv:
        export(sessions, directory)


if __name__ == "__main__":
    main()
