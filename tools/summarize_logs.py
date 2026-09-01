"""세션 로그(JSON)를 읽어 조건별·위상별로 집계한다.

session_log.py 가 떨어뜨린 logs/*.json 을 전부 읽어 네 개의 표를 낸다.

  1) 세션 목록      — 파일별 유효성 점검
  2) 조건별 집계    — 개입 지표 / 채점 / 종합 점수
  3) 위상별 웹캠    — 조건 C 의 NORMAL vs TRUNCATED vs REACTION  ← 핵심 표
  4) 조건별 기저선  — 세 조건의 NORMAL 턴만 비교

--csv 를 주면 통계 도구용 CSV 두 개를 만든다.

웹캠 키 이름은 vision_process/aggregator.py 의 end_turn() 반환 dict 와
1:1로 맞춰져 있다. 워커 쪽 키를 바꾸면 여기도 같이 바꿔야 한다.

사용법:
    cd VRoom_Backend
    python -m tools.summarize_logs                 # logs/ 를 읽는다
    python -m tools.summarize_logs data/pilot      # 다른 폴더
    python -m tools.summarize_logs --csv           # CSV 도 함께 생성
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

# Unity BehaviorCollector 가 붙이는 턴 위상.
PHASES = ("NORMAL", "TRUNCATED", "REACTION", "REANSWER")

# ── 웹캠 지표 ───────────────────────────────────────────────────────────
# 출처: vision_process/aggregator.py:191-213 (end_turn 반환값)
# 콘솔 표에 쓸 핵심 6개. 폭이 좁아 전부는 못 넣으므로 CSV 에 나머지를 담는다.
VISION_CORE = (
    "gazeOnTargetRatio",    # 시선이 면접관을 향한 프레임 비율
    "headYawStd",           # 좌우 두리번거림
    "bodySwayStd",          # 상체 흔들림
    "handUsageRatio",       # 손을 쓴 프레임 비율
    "faceTouchCount",       # 얼굴 만짐 횟수
    "expressionVariance",   # 표정 변화량
)

# CSV 에 담을 전체 지표. 폐기된 3개는 제외했다.
#   torsoDriftMean   — 경과 시간에 단조 증가하는 편향이 있어 비활성
#   blinkPerMinute   — 10fps 라 눈 깜빡임 구간을 놓침
#   handMotionEnergy — handUsageRatio 로 대체됨
VISION_ALL = VISION_CORE + (
    "shoulderTiltMean", "headPitchStd", "handExtent", "smileRatio", "frownRatio",
)
VISION_META = ("durationSec", "frameCount", "faceDetectedRatio", "poseDetectedRatio",
               "calibrated")
DEPRECATED = ("torsoDriftMean", "blinkPerMinute", "handMotionEnergy")

# 얼굴 검출이 이보다 낮은 턴은 지표를 신뢰할 수 없다 (VisionScoringConfig 와 동일 취지).
MIN_FACE_RATIO = 0.5


# ---------------------------------------------------------------------------
# 로딩 / 유효성
# ---------------------------------------------------------------------------
def load_all(directory: str) -> list[dict]:
    sessions = []
    for p in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.basename(p).startswith("summary_"):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = os.path.basename(p)
            sessions.append(data)
        except Exception as e:
            print(f"  ! 읽기 실패 {os.path.basename(p)}: {e}")
    return sessions


def validity_flags(s: dict) -> list[str]:
    """분석에서 빼야 할 이유를 모은다. 비어 있으면 유효한 세션이다."""
    flags = []
    meta = s.get("meta", {})
    if meta.get("force_negative"):
        flags.append("FORCE_NEG")       # G5 우회 — 조건 C 가 성립하지 않는다
    if meta.get("condition") not in CONDITIONS:
        flags.append("NO_COND")
    if meta.get("exit_reason") != "normal":
        flags.append("ABORTED")
    if s.get("feedback") is None:
        flags.append("NO_REPORT")
    if len(s.get("turn_features", [])) != len(s.get("turn_stages", [])):
        flags.append("MISALIGNED")      # 인덱스 정합 붕괴
    if not s.get("vision_turns"):
        flags.append("NO_VISION")
    return flags


def usable_vision(s: dict) -> list[dict]:
    """얼굴 검출이 충분한 웹캠 턴만 돌려준다."""
    return [t for t in s.get("vision_turns", [])
            if t.get("faceDetectedRatio", 0.0) >= MIN_FACE_RATIO]


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.fmean(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.stdev(xs) if len(xs) > 1 else None


def fmt(v, nd: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def short(key: str, width: int = 18) -> str:
    """긴 지표 이름을 표 폭에 맞게 줄인다."""
    return key if len(key) <= width else key[:width - 1] + "."


# ---------------------------------------------------------------------------
# 1. 세션 목록
# ---------------------------------------------------------------------------
def print_sessions(sessions: list[dict]) -> None:
    print("\n1. 세션 목록")
    print("-" * 125)
    print(f"{'파일':<40}{'조건':>4}{'길이(s)':>9}{'턴':>4}{'개입':>5}"
          f"{'웹캠턴':>7}{'종합':>6}   비고")
    print("-" * 125)
    for s in sessions:
        meta = s.get("meta", {})
        fb = s.get("feedback") or {}
        flags = validity_flags(s)
        print(f"{s['_file']:<40}"
              f"{meta.get('condition', '?'):>4}"
              f"{fmt(meta.get('duration_sec'), 1):>9}"
              f"{len(s.get('turn_stages', [])):>4}"
              f"{s.get('bargein', {}).get('total', 0):>5}"
              f"{len(usable_vision(s)):>7}"
              f"{fmt(fb.get('overall_score'), 0):>6}   "
              f"{','.join(flags) if flags else 'OK'}")
    print("-" * 125)


# ---------------------------------------------------------------------------
# 2. 조건별 집계
# ---------------------------------------------------------------------------
def print_by_condition(sessions: list[dict]) -> None:
    print("\n2. 조건별 집계 (유효 세션만)")
    print("-" * 125)
    print(f"{'조건':<6}{'N':>4}{'개입/세션':>11}{'양보(s)':>10}{'첫발성(ms)':>12}"
          f"{'잘린발화(s)':>12}{'평균점수':>10}{'종합점수':>10}")
    print("-" * 125)

    for cond in CONDITIONS:
        group = [s for s in sessions
                 if s.get("meta", {}).get("condition") == cond and not validity_flags(s)]
        if not group:
            print(f"{cond:<6}{0:>4}   (유효 세션 없음)")
            continue

        events = [e for s in group for e in s.get("bargein", {}).get("events", [])]
        stage_scores = [r.get("score") for s in group for r in s.get("stage_scores", [])
                        if isinstance(r.get("score"), int) and r["score"] >= 0]
        overalls = [(s.get("feedback") or {}).get("overall_score") for s in group]

        print(f"{cond:<6}{len(group):>4}"
              f"{fmt(mean([s.get('bargein', {}).get('total', 0) for s in group]), 2):>11}"
              f"{fmt(mean([e.get('yield_time') for e in events]), 2):>10}"
              f"{fmt(mean([e.get('latency_to_speech_ms') for e in events]), 0):>12}"
              f"{fmt(mean([e.get('utterance_elapsed') for e in events]), 1):>12}"
              f"{fmt(mean(stage_scores), 1):>10}"
              f"{fmt(mean(overalls), 1):>10}")
    print("-" * 125)

    types: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for s in sessions:
        if validity_flags(s):
            continue
        for e in s.get("bargein", {}).get("events", []):
            types[e.get("type", "?")] = types.get(e.get("type", "?"), 0) + 1
            reasons[e.get("reason", "?")] = reasons.get(e.get("reason", "?"), 0) + 1
    if types:
        print(f"  개입 유형: {types}")
        print(f"  개입 사유: {reasons}")

# ---------------------------------------------------------------------------
# 3. 위상별 웹캠 지표 — 핵심 표
# ---------------------------------------------------------------------------
def print_by_phase(sessions: list[dict]) -> None:
    print("\n3. 위상별 웹캠 지표 (조건 C, 유효 세션)")
    print("   NORMAL=평상 답변 / TRUNCATED=끊긴 답변 / REACTION=개입 직후 5초 / REANSWER=재답변")
    print("-" * 125)

    buckets = {p: {k: [] for k in VISION_CORE} for p in PHASES}
    counts = {p: 0 for p in PHASES}

    for s in sessions:
        if validity_flags(s) or s.get("meta", {}).get("condition") != "C":
            continue
        for t in usable_vision(s):
            phase = t.get("phase", "NORMAL")
            if phase not in buckets:
                continue
            counts[phase] += 1
            for k in VISION_CORE:
                if isinstance(t.get(k), (int, float)):
                    buckets[phase][k].append(t[k])

    if not any(counts.values()):
        print("  (조건 C 의 유효 세션이 없거나 웹캠 턴이 비어 있다)")
        print("-" * 125)
        return

    print(f"{'위상':<12}{'N':>5}" + "".join(f"{short(k):>18}" for k in VISION_CORE))
    print("-" * 125)
    for p in PHASES:
        if counts[p] == 0:
            continue
        row = f"{p:<12}{counts[p]:>5}"
        row += "".join(f"{fmt(mean(buckets[p][k]), 3):>18}" for k in VISION_CORE)
        print(row)
    print("-" * 125)

    # NORMAL 대비 변화량 — 개입의 즉각 효과를 한 줄로 보여준다.
    base = {k: mean(buckets["NORMAL"][k]) for k in VISION_CORE}
    for p in ("TRUNCATED", "REACTION", "REANSWER"):
        if counts[p] == 0 or counts["NORMAL"] == 0:
            continue
        row = f"{'Δ ' + p:<12}{'':>5}"
        for k in VISION_CORE:
            m, b = mean(buckets[p][k]), base[k]
            row += f"{(f'{m - b:+.3f}' if (m is not None and b is not None) else '-'):>18}"
        print(row)
    print("-" * 125)
    print("  * NORMAL 대비 REACTION 의 변화가 개입의 즉각 효과다.")
    print("  * 표본이 조건당 10 미만이면 평균만 보고 결론 내지 말 것.")

def print_typea_scores(sessions):
    """Type A 개입 전후 답변 품질 비교. 논문의 핵심 지표."""
    rows = []
    for s in sessions:
        for ev in s.get("bargein", {}).get("events", []):
            if ev.get("type") != "REDIRECT":
                continue
            rows.append((
                ev.get("stage", ""),
                ev.get("score_truncated", -1),
                ev.get("score_reanswer", -1),
                ev.get("score_final", -1),
                ev.get("latency_judge_ms"),
                ev.get("latency_to_speech_ms"),
                ev.get("truncated_source", "final"),
            ))
    if not rows:
        print("\n5. Type A 개입 전후 점수 — 해당 이벤트 없음")
        return

    print("\n5. Type A (REDIRECT) 개입 전후 답변 품질")
    print("-" * 125)
    print(f"{'단계':<14}{'잘린답변':>10}{'재답변':>10}{'최종':>8}"
          f"{'판정(ms)':>10}{'첫발성(ms)':>12}{'채점출처':>16}")
    print("-" * 125)
    valid = [(t, r) for _, t, r, _, _, _, _ in rows if t >= 0 and r >= 0]
    for stage, tr, re_, fin, lj, ls, src in rows:
        print(f"{stage:<14}{tr:>10}{re_:>10}{fin:>8}"
              f"{lj if lj else '-':>10}{ls if ls else '-':>12}{src:>16}")
    print("-" * 125)
    if valid:
        d = sum(r - t for t, r in valid) / len(valid)
        print(f"  재답변 - 잘린답변 평균 차이: {d:+.1f}점 (N={len(valid)})")
    print(f"  * 이 값이 양수이면 개입이 답변 품질을 끌어올렸다는 방향이다.")

# ---------------------------------------------------------------------------
# 4. 조건별 기저선 — NORMAL 턴만
# ---------------------------------------------------------------------------
def print_baseline(sessions: list[dict]) -> None:
    print("\n4. 조건별 기저선 비교 (NORMAL 턴만)")
    print("-" * 125)
    print(f"{'조건':<12}{'N':>5}" + "".join(f"{short(k):>18}" for k in VISION_CORE))
    print("-" * 125)
    for cond in CONDITIONS:
        vals = {k: [] for k in VISION_CORE}
        n = 0
        for s in sessions:
            if validity_flags(s) or s.get("meta", {}).get("condition") != cond:
                continue
            for t in usable_vision(s):
                if t.get("phase", "NORMAL") != "NORMAL":
                    continue
                n += 1
                for k in VISION_CORE:
                    if isinstance(t.get(k), (int, float)):
                        vals[k].append(t[k])
        if n == 0:
            continue
        row = f"{cond:<12}{n:>5}"
        row += "".join(f"{fmt(mean(vals[k]), 3):>18}" for k in VISION_CORE)
        print(row)
    print("-" * 125)
    print("  * 개입이 없는 평상 구간의 조건 간 차이. 여기서 큰 차이가 나면")
    print("    개입 효과가 아니라 참가자/회차 차이를 보고 있는 것이다.")


# ---------------------------------------------------------------------------
# 5. CSV
# ---------------------------------------------------------------------------
def export_csv(sessions: list[dict], directory: str) -> None:
    sess_path = os.path.join(directory, "summary_sessions.csv")
    with open(sess_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "condition", "exit_reason", "valid", "flags",
                    "duration_sec", "turns", "bargein_total",
                    "mean_yield_time", "mean_latency_to_speech_ms",
                    "mean_stage_score", "overall_score"])
        for s in sessions:
            meta = s.get("meta", {})
            fb = s.get("feedback") or {}
            flags = validity_flags(s)
            ev = s.get("bargein", {}).get("events", [])
            scores = [r.get("score") for r in s.get("stage_scores", [])
                      if isinstance(r.get("score"), int) and r["score"] >= 0]
            w.writerow([s["_file"], meta.get("condition"), meta.get("exit_reason"),
                        int(not flags), "|".join(flags),
                        meta.get("duration_sec"), len(s.get("turn_stages", [])),
                        s.get("bargein", {}).get("total", 0),
                        fmt(mean([e.get("yield_time") for e in ev]), 2),
                        fmt(mean([e.get("latency_to_speech_ms") for e in ev]), 0),
                        fmt(mean(scores), 1), fb.get("overall_score")])

    # 턴 단위 롱 포맷 — 위상별 비교를 R/SPSS 에서 바로 돌릴 수 있다.
    turn_path = os.path.join(directory, "summary_vision_turns.csv")
    with open(turn_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "condition", "stage", "phase", *VISION_META, *VISION_ALL])
        for s in sessions:
            if validity_flags(s):
                continue
            cond = s.get("meta", {}).get("condition")
            for t in usable_vision(s):
                w.writerow([s["_file"], cond, t.get("stage"), t.get("phase"),
                            *[t.get(k) for k in VISION_META],
                            *[t.get(k) for k in VISION_ALL]])

    print(f"\nCSV 생성:\n  {sess_path}\n  {turn_path}")


# ---------------------------------------------------------------------------
def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--csv"]
    want_csv = "--csv" in sys.argv
    directory = args[0] if args else DEFAULT_DIR

    if not os.path.isdir(directory):
        print(f"디렉터리를 찾을 수 없다: {directory}")
        return

    sessions = load_all(directory)
    print("=" * 125)
    print(f"VRoom 세션 로그 집계 — {directory} ({len(sessions)}개)")
    print("=" * 125)
    if not sessions:
        print("JSON 파일이 없다. 세션을 한 번 완주해야 로그가 생성된다.")
        return

    versions = {s.get("schema_version") for s in sessions}
    if len(versions) > 1:
        print(f"! 스키마 버전이 섞여 있다: {versions} — 비교 전에 확인할 것")

    # 워커가 키 이름을 바꾸면 조용히 '-' 로 나오므로 미리 경고한다.
    seen = {k for s in sessions for t in s.get("vision_turns", []) for k in t}
    missing = [k for k in VISION_CORE if seen and k not in seen]
    if missing:
        print(f"! 로그에 없는 웹캠 키: {missing}")
        print(f"  vision_process/aggregator.py 의 end_turn() 반환 키와 대조할 것")
    stale = [k for k in DEPRECATED if k in seen]
    if stale:
        print(f"  (참고) 폐기 지표가 로그에 남아 있다: {stale} — 분석에서 제외했다")

    print_sessions(sessions)
    print_by_condition(sessions)
    print_by_phase(sessions)
    print_baseline(sessions)

    invalid = [s for s in sessions if validity_flags(s)]
    if invalid:
        print(f"\n! 유효하지 않은 세션 {len(invalid)}건은 2~4번 집계에서 제외했다.")
        print("  FORCE_NEG=G5 우회 / ABORTED=중도 종료 / MISALIGNED=인덱스 붕괴 /")
        print("  NO_VISION=웹캠 턴 없음 / NO_REPORT=피드백 미생성")

    if want_csv:
        export_csv(sessions, directory)


if __name__ == "__main__":
    main()
