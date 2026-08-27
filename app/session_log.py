"""세션 종료 시 실험 원자료를 JSON 파일 하나로 떨어뜨린다.

목적은 결과 리포트가 아니라 재분석이다. 화면에 보여줄 요약이 아니라
실험 조건 / 게이트 설정 / 턴별 원자료 / 개입 이벤트를 그대로 남겨,
데이터 수집이 끝난 뒤에도 조건 간 비교를 다시 돌릴 수 있게 한다.

파일명: {조건}_{세션ID}_{YYYYmmdd_HHMMSS}.json
쓰기는 임시 파일 -> os.replace 로 원자적으로 처리한다. 세션 종료와
프로세스 종료가 겹쳐도 반쯤 쓰인 JSON 이 남지 않는다.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

from .config import BargeInConfig, settings

SCHEMA_VERSION = 1


def _safe(value) -> str:
    """파일명에 쓸 수 없는 문자를 걷어낸다."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))
    return cleaned[:40] or "unknown"


def build(session, report=None, exit_reason: str = "normal") -> dict:
    """세션 객체를 직렬화 가능한 dict 로 변환한다. 파일 IO 는 하지 않는다."""
    ended = time.time()
    started = session.session_started_at

    return {
        "schema_version": SCHEMA_VERSION,

        # ── 이 세션을 분석에 쓸 수 있는지 판정하는 헤더 ──────────────
        "meta": {
            "session_id": session.session_id,
            "condition": session.condition,          # A | B | C
            "exit_reason": exit_reason,              # normal | disconnect
            # True 면 G5 가 우회되어 조건 C 가 성립하지 않는다. 반드시 확인할 것.
            "force_negative": settings.bargein_force_negative,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.openai_model,
            "company": session.company,
            "job_title": session.job_title,
            "final_stage": session.stage.value,
            "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "ended_at": datetime.fromtimestamp(ended).isoformat(timespec="seconds"),
            "duration_sec": round(ended - started, 1),
        },

        # ── 게이트 상수 스냅샷 ──────────────────────────────────────
        # 실험 중간에 상수를 조정하면 세션마다 조건이 달라진다.
        # 값을 같이 남겨야 나중에 "이 세션은 어떤 설정이었나"를 답할 수 있다.
        "gate_config": {
            "target_stages": sorted(BargeInConfig.TARGET_STAGES),
            "max_per_session": BargeInConfig.MAX_PER_SESSION,
            "grace_sec": BargeInConfig.GRACE_SEC,
            "min_partial_chars": BargeInConfig.MIN_PARTIAL_CHARS,
            "final_wait_timeout": BargeInConfig.FINAL_WAIT_TIMEOUT,
        },

        # ── 원자료 ──────────────────────────────────────────────────
        "turns": session.turns,                                  # {role, stage, text}
        "turn_stages": session.turn_stages,                      # turn_features 와 1:1
        "turn_features": session.turn_features,                  # 턴별 음성 피쳐
        "stage_scores": [{"stage": s, "score": v} for s, v in session.stage_scores],
        "vision_turns": session.vision_turns,                    # stage/phase 포함
        "bargein": {
            "total": session.bargein_total,
            "used_stages": sorted(session.bargein_used_stages),
            "events": session.bargein_log,                       # 논문 지표 12개 필드
        },

        # ── 최종 리포트 (비정상 종료면 None) ────────────────────────
        "feedback": report.model_dump() if report is not None else None,
    }


def dump(session, report=None, exit_reason: str = "normal") -> str | None:
    """세션 로그를 파일로 쓴다. 이미 썼으면 아무것도 하지 않는다.

    로그 실패가 면접 진행을 막아서는 안 되므로 모든 예외를 삼킨다.
    다만 조용히 사라지면 안 되니 콘솔에는 반드시 남긴다.
    """
    if getattr(session, "log_written", False):
        return None

    try:
        os.makedirs(settings.session_log_dir, exist_ok=True)
        name = (f"{_safe(session.condition)}_{_safe(session.session_id)}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        path = os.path.join(settings.session_log_dir, name)
        tmp = path + ".tmp"

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(build(session, report, exit_reason), f,
                      ensure_ascii=False, indent=2)
        os.replace(tmp, path)

        session.log_written = True
        print(f"[{session.session_id}] 세션 로그 저장 — {path}")
        return path

    except Exception as e:
        print(f"[{session.session_id}] 세션 로그 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        return None