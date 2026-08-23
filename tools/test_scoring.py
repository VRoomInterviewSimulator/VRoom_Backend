"""
채점 로직 검증용 백엔드 단독 테스트.

Unity / STT 워커 / TTS 워커 없이 백엔드만 띄운 채로
면접 6턴을 자동으로 진행하며 단계별 score / persona 를 표로 출력한다.

사용법:
    # .env 에 SKIP_TTS=true 설정 후 백엔드 실행
    pip install websockets
    python tools/test_scoring.py            # 충실한 답변 프로필
    python tools/test_scoring.py bad        # 부실한 답변 프로필
    python tools/test_scoring.py mixed      # 좋음/나쁨 교차

기대 결과:
    SELF_INTRO, TECH_Q1        -> score = -1  (설계상 채점 대상 아님)
    FOLLOWUP_1, FOLLOWUP_2,
    BEHAVIORAL, CLOSING        -> score = 0~100
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import urllib.request

import websockets

BASE_HTTP = "http://127.0.0.1:8080"
BASE_WS = "ws://127.0.0.1:8080/ws/control"
SESSION_ID = "test_scoring"

COMPANY = "넥슨"
JOB_TITLE = "게임 클라이언트 프로그래머"
RESUME = (
    "유니티 2년차 게임 클라이언트 프로그래머입니다. "
    "1인 개발 프로젝트로 기획, 프로그래밍, 아트를 모두 담당한 경험이 있으며, "
    "Photon을 통해 멀티플레이 게임을 개발했습니다."
)

# ---------------------------------------------------------------------------
# 답변 프로필: 6턴 (자기소개 / 기술 / 꼬리1 / 꼬리2 / 인성 / 마무리)
# ---------------------------------------------------------------------------
GOOD = [
    "안녕하세요. 유니티 기반 게임 클라이언트를 2년간 개발해 온 유재명입니다. "
    "1인 개발로 8번 출구 모작인 '지하 10층'을 출시했고, Photon 기반 멀티플레이 게임 "
    "'풍림화산 전쟁'도 개발했습니다. 넥슨의 라이브 서비스 규모에서 클라이언트 성능 "
    "최적화를 깊이 있게 다뤄보고 싶어 지원했습니다.",

    "가장 중요한 최적화는 드로우콜 절감이라고 생각합니다. '풍림화산 전쟁'에서는 유닛이 "
    "동시에 200개까지 스폰되는데, 초기에는 유닛마다 개별 머티리얼을 써서 드로우콜이 "
    "300회를 넘겼습니다. 이를 텍스처 아틀라스로 묶고 GPU 인스턴싱을 적용해 40회 수준으로 "
    "줄였고, 모바일 기준 프레임이 32에서 58로 올랐습니다.",

    "GC 스파이크도 큰 문제였습니다. 매 프레임 유닛 탐색에 LINQ와 new List를 쓰고 있어서 "
    "1초에 2MB 가까이 할당되고 있었습니다. 오브젝트 풀링을 도입하고 탐색 로직을 "
    "사전 할당된 배열 순회로 바꿔서 프레임당 할당을 0에 가깝게 만들었습니다. "
    "Profiler로 측정했을 때 GC.Alloc 스파이크가 사라졌습니다.",

    "상황은 출시 2주 전 QA에서 저사양 기기 프레임 드랍 리포트가 올라온 것이었습니다. "
    "제 과제는 릴리즈 일정을 지키면서 저사양 대응을 하는 것이었고, 저는 우선 Profiler로 "
    "병목이 렌더링인지 스크립트인지부터 나눴습니다. 렌더링이 70%를 차지해 LOD와 "
    "인스턴싱을 먼저 적용했고, 결과적으로 일정 안에 목표 프레임 30을 달성했습니다.",

    "팀 프로젝트에서 아트 담당과 텍스처 해상도로 의견이 갈린 적이 있습니다. 아트는 품질을, "
    "저는 메모리를 우선했습니다. 감으로 논쟁하는 대신 실제 기기에서 2048과 1024를 각각 "
    "빌드해 메모리 사용량과 스크린샷을 나란히 놓고 비교했습니다. 육안 차이가 크지 않다는 "
    "데 합의해서 1024로 정했고, 이후에도 수치로 이야기하는 문화가 자리 잡았습니다.",

    "라이브 서비스 환경에서 성능 회귀를 어떻게 사전에 잡는지 궁금합니다. "
    "CI에 자동 프로파일링을 붙이는 방식인지, 아니면 QA 단계에서 잡는 구조인지 알고 싶습니다.",
]

BAD = [
    "저는 유재명입니다. 게임 좋아합니다.",
    "음... 잘 모르겠습니다. 그냥 최적화 잘 하면 되는 것 같습니다.",
    "그냥 만들었습니다.",
    "특별히 기억나는 건 없습니다.",
    "혼자 했어서 갈등은 없었습니다.",
    "없습니다.",
]

MIXED = [GOOD[0], GOOD[1], BAD[2], GOOD[3], BAD[4], GOOD[5]]

PROFILES = {"good": GOOD, "bad": BAD, "mixed": MIXED}

SCORING_STAGES = {"FOLLOWUP_1", "FOLLOWUP_2", "BEHAVIORAL", "CLOSING"}


def http_post(path: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_HTTP + path, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


async def wait_turn(ws, timeout: float = 90.0) -> dict | None:
    """thinking / audio_end 를 흘려보내고 실제 면접관 발화 패킷만 돌려준다."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
        except asyncio.TimeoutError:
            return None
        if isinstance(raw, bytes):
            continue                      # PCM 음성은 무시
        msg = json.loads(raw)
        mtype = msg.get("type")
        if mtype in ("thinking", "audio_end"):
            continue
        if mtype == "feedback_report":
            return msg
        if msg.get("dialogue"):
            return msg
    return None


async def main():
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "good"
    answers = PROFILES.get(profile_name)
    if answers is None:
        print(f"프로필은 {list(PROFILES)} 중 하나여야 합니다.")
        return

    print(f"=== 프로필: {profile_name} / session={SESSION_ID} ===\n")

    # 1) 세션 준비
    res = http_post("/session/prepare", {
        "session_id": SESSION_ID, "company": COMPANY,
        "job_title": JOB_TITLE, "resume": RESUME,
    })
    print(f"[prepare] {res.get('dialogue', '')[:60]}...")
    print(f"[prepare] audio_bytes={res.get('audio_bytes')} "
          f"(SKIP_TTS=true 면 0 이 정상)\n")

    rows: list[tuple[str, int, str, float]] = []

    async with websockets.connect(BASE_WS, max_size=None) as ws:
        # 2) init -> 첫 질문
        await ws.send(json.dumps({
            "type": "init", "session_id": SESSION_ID,
            "company": COMPANY, "job_title": JOB_TITLE, "resume": RESUME,
        }))
        first = await wait_turn(ws)
        if first:
            print(f"[{first['stage']}] {first['dialogue']}")
            rows.append((first["stage"], first["score"],
                         first["persona"], first.get("persona_value", 0.0)))

        # 3) 6턴 진행
        for i, answer in enumerate(answers, start=1):
            print(f"\n  >> 답변 {i}: {answer[:50]}...")
            t0 = time.perf_counter()
            http_post("/process", {
                "session_id": SESSION_ID,
                "text": answer,
                "features": {"speakingTime": 25.0, "pauseCount": 2,
                             "averageVolume": 0.12},
            })
            packet = await wait_turn(ws)
            elapsed = time.perf_counter() - t0
            if packet is None:
                print("  !! 응답 없음 (타임아웃)")
                break

            stage = packet["stage"]
            score = packet["score"]
            mark = ""
            if stage in SCORING_STAGES and score < 0:
                mark = "   <-- 버그: 채점 단계인데 -1"
            print(f"[{stage}/{packet['persona']}] score={score} "
                  f"emo={packet.get('persona_value', 0.0):+.2f} ({elapsed:.1f}s){mark}")
            print(f"    {packet['dialogue']}")
            rows.append((stage, score, packet["persona"],
                         packet.get("persona_value", 0.0)))

            if packet.get("is_final"):
                break

        # 4) 최종 피드백
        await ws.send(json.dumps({
            "type": "request_feedback", "session_id": SESSION_ID,
        }))
        report = await wait_turn(ws, timeout=120)

    # 5) 요약표
    print("\n" + "=" * 68)
    print(f"{'단계':<14}{'점수':>6}{'페르소나':>12}{'감정강도':>10}   판정")
    print("-" * 68)
    bug_count = 0
    for stage, score, persona, emo in rows:
        expect_score = stage in SCORING_STAGES
        if expect_score and score < 0:
            verdict = "BUG (채점 누락)"
            bug_count += 1
        elif expect_score:
            verdict = "OK"
        elif score >= 0:
            verdict = "BUG (미채점 단계인데 점수)"
            bug_count += 1
        else:
            verdict = "OK (미채점 단계)"
        print(f"{stage:<14}{score:>6}{persona:>12}{emo:>+10.2f}   {verdict}")
    print("=" * 68)
    print(f"결과: {'통과' if bug_count == 0 else f'{bug_count}건 문제'}\n")

    if report and report.get("type") == "feedback_report":
        print(f"[피드백] overall_score={report.get('overall_score')}")
        print(f"[피드백] stage_scores={report.get('stage_scores')}")
        print(f"[피드백] accuracy={report.get('scores', {}).get('accuracy')}")


if __name__ == "__main__":
    asyncio.run(main())
