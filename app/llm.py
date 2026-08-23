"""
LLM 엔진. OpenAI 호환 SDK로 면접관 대사 + 채점 + 비언어 메타데이터를
단일 JSON으로 강제 출력받는다.

명세 반영:
  - 자기소개 답변에서 핵심 정보를 Function Calling으로
    추출(extract_info)해 '동적 페르소나'를 구성한다. == example.py 의 extract_interview_info
  - 꼬리 질문 단계에서 직전 답변을 구체적으로 인용하고, STAR(상황/과제/
    행동/결과) 중 부족한 부분을 콕 집으며, 경력 수준에 맞춰 난이도를 조정한다.

설계 원칙: 시나리오 진행과 페르소나 결정은 코드(session.py)가 통제하고,
LLM은 '주어진 단계/페르소나/추출정보에 맞는 대사·점수·제스처'만 생성한다.

통합 지점: generate_turn / generate_feedback / extract_info 의
입출력 계약(LLMTurn, dict)만 유지하면 내부 구현을 자유롭게 교체할 수 있다.
"""

from __future__ import annotations

import json
from typing import List

from openai import AsyncOpenAI

from .config import settings
from .domain import (
    ExpressionID,
    ExtractedInfo,
    GestureID,
    LLMTurn,
    Persona,
    PERSONA_BEHAVIOR_SET,
    Stage,
)

_base_url, _api_key, _model = settings.resolve_llm()
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """클라이언트를 첫 사용 시점에 생성(지연 초기화). 키가 없어도 임포트는 성공한다."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=_api_key or "missing-key", base_url=_base_url)
    return _client


# ---------------------------------------------------------------------------
# (1) 자기소개 정보 추출 — Function Calling
#     example.py 의 extract_interview_info 와 동일한 스키마/강제호출 패턴.
# ---------------------------------------------------------------------------
_EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_interview_info",
        "description": "면접 자기소개에서 핵심 정보를 추출합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "지원 기업명"},
                "job_role": {"type": "string", "description": "지원 직무"},
                "experience_level": {
                    "type": "string",
                    "enum": ["신입", "주니어", "중급", "시니어"],
                },
                "mentioned_skills": {
                    "type": "array", "items": {"type": "string"},
                    "description": "언급된 기술 스택",
                },
                "key_strengths": {
                    "type": "array", "items": {"type": "string"},
                    "description": "지원자가 강조한 강점",
                },
            },
            "required": ["company_name", "job_role", "experience_level"],
        },
    },
}


async def extract_info(self_intro: str, *, fallback_company: str = "",
                       fallback_job: str = "") -> ExtractedInfo:
    """자기소개 답변에서 동적 페르소나 슬롯을 추출한다.

    명세 [1단계]: '자기소개 답변이 동적 페르소나 프롬프트를 활성화'.
    추출 실패 시 Unity init 으로 받은 회사/직무로 폴백한다.
    """
    try:
        resp = await _get_client().chat.completions.create(
            model=_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "사용자의 자기소개에서 면접에 필요한 정보를 정확히 추출하세요."},
                {"role": "user", "content": self_intro},
            ],
            tools=[_EXTRACTION_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_interview_info"}},
        )
        args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        info = ExtractedInfo(**args)
    except Exception:
        info = ExtractedInfo()

    # 비어 있으면 Unity가 준 값으로 보강
    if not info.company_name:
        info.company_name = fallback_company
    if not info.job_role:
        info.job_role = fallback_job
    return info


# ---------------------------------------------------------------------------
# (2) 턴 생성 — 대사 + 채점 + 제스처/표정 (JSON 강제)
# ---------------------------------------------------------------------------
async def _ask_json(system: str, user: str) -> dict:
    resp = await _get_client().chat.completions.create(
        model=_model,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def _clamp_to_set(turn: LLMTurn, persona: Persona) -> LLMTurn:
    """LLM이 허용 세트를 벗어난 ID를 내놓아도 안전하게 보정."""
    sets = PERSONA_BEHAVIOR_SET[persona]
    expr_ids = [e.value for e in sets["expressions"]]
    gest_ids = [g.value for g in sets["gestures"]]
    if turn.expression_id not in expr_ids:
        turn.expression_id = expr_ids[0]
    if turn.gesture_id not in gest_ids:
        turn.gesture_id = gest_ids[0]
    return turn


# 단계별 '이번 턴에 해야 할 일' 지시문. (시나리오 설계 문서의 6단계 흐름)
STAGE_TASK = {
    Stage.SELF_INTRO: "면접 시작 인사를 건네고, 타겟 기업/직무를 포함한 자기소개를 요청하라. 아직 채점 대상이 아니므로 score=-1.",
    Stage.TECH_Q1: "지원 직무의 핵심 역량을 검증하는 기술 질문 1개를 하라. 직전 답변(자기소개)은 채점하지 말고 score=-1.",
    Stage.FOLLOWUP_1: "직전 기술 답변의 정확도/논리성을 0~100으로 채점하라. 직전 답변의 핵심 표현을 한 구절 그대로 인용하며, 점수가 높으면 심화 질문, 낮거나 모호하면 반박/압박 꼬리질문을 하라.",
    Stage.FOLLOWUP_2: "직전 답변을 STAR(상황/과제/행동/결과) 관점에서 재채점하라. STAR 중 빠졌거나 모호한 요소를 정확히 한 가지 콕 집어 그 부분을 캐묻는 2차 꼬리질문을 하라.",
    Stage.BEHAVIORAL: "직전 답변을 채점한 뒤, 협업/갈등해결 등 인성·조직적합성 질문으로 분위기를 자연스럽게 전환하라.",
    Stage.CLOSING: "직전 답변을 채점한 뒤, '마지막으로 하고 싶은 말이나 궁금한 점'을 묻는 마무리 질문을 하고 정중히 마친다.",
}


def _system_prompt(info: ExtractedInfo, resume: str) -> str:
    """동적 페르소나 System Prompt. example.py 의 persona_prompt 가이드라인을 그대로 반영."""
    skills = ", ".join(info.mentioned_skills) or "미언급"
    strengths = ", ".join(info.key_strengths) or "미언급"
    return (
        f"당신은 {info.company_name or '지원 기업'}의 {info.job_role or '해당 직무'} 채용 면접관이다.\n"
        "지원자 정보:\n"
        f"- 경력 수준: {info.experience_level}\n"
        f"- 보유 기술: {skills}\n"
        f"- 강조한 강점: {strengths}\n"
        f"- 이력서 요약: {resume or '미언급'}\n\n"
        "면접관 가이드라인:\n"
        "- 지원자가 언급한 기술과 경험에 대해 깊이 있게 질문한다.\n"
        "- 지원자의 직전 답변 내용을 구체적으로 인용하며 꼬리 질문을 만든다.\n"
        "- STAR(상황-과제-행동-결과) 중 부족하거나 모호한 부분이 있으면 콕 집어 물어본다.\n"
        f"- {info.experience_level} 수준에 적합한 난이도로 조정한다.\n"
        "- 전문적이지만 위협적이지 않은 톤을 유지한다.\n"
        "- 질문은 한 번에 하나만, 2~3문장 이내로 짧게 한다.\n"
        "규칙:\n"
        "1) 모든 대사는 자연스러운 한국어 존댓말.\n"
        "2) 대사는 음성합성(TTS)으로 출력되므로 마크다운/이모지/괄호설명 없이 말로만 작성.\n"
        "3) 반드시 지정된 JSON 스키마로만 응답한다. 그 외 텍스트 금지.\n"
    )


def _turn_instruction(stage: Stage, persona: Persona, history: str, user_answer: str) -> str:
    sets = PERSONA_BEHAVIOR_SET[persona]
    allowed_expr = ", ".join(f"{e.value}={e.name}" for e in sets["expressions"])
    allowed_gest = ", ".join(f"{g.value}={g.name}" for g in sets["gestures"])
    return (
        f"[현재 단계] {stage.value}\n"
        f"[이번 턴 과제] {STAGE_TASK[stage]}\n"
        f"[현재 페르소나] {persona.value} "
        "(POSITIVE=안정감/미소, NEUTRAL=경청, NEGATIVE=압박/냉정)\n"
        f"[직전 지원자 답변] {user_answer or '(아직 없음 - 면접관이 먼저 말함)'}\n"
        f"[지금까지의 대화 요약]\n{history or '(없음)'}\n\n"
        "반드시 아래의 순수 JSON 형식으로만 응답하라. 설명이나 주석 기호 없이 JSON 괄호만 출력할 것:\n"
        "{\n"
        '  "dialogue": "면접관 대사(한국어)",\n'
        '  "score": 0,\n'
        '  "score_reason": "채점 근거 한 문장",\n'
        f'  "expression_id": 1\n'
        f'  "gesture_id": 1\n'
        "}\n\n"
        "[데이터 작성 엄격 규칙]\n"
        "- score: 채점 대상이 아닌 단계(인사 등)면 무조건 -1. 채점 대상이면 지원자 답변의 정확도를 평가하여 0~100 사이의 정수(int)만 기입할 것.\n"
        f"- expression_id: 반드시 다음 정수 중 하나만 기입할 것 ({allowed_expr})\n"
        f"- gesture_id: 반드시 다음 정수 중 하나만 기입할 것 ({allowed_gest})\n"
        #"expression_id/gesture_id는 반드시 위 허용값 중에서만 고른다."
    )


def _truncation_premise(reason: str) -> str:
    """설계서 6.2/6.3: CUTOFF 개입 직후 다음 질문 생성 시 주입하는 전제."""
    label = {
        "LONG_ANSWER": "답변이 지나치게 길어져 시스템이 시간 초과로 중단시켰다",
        "LONG_SILENCE": "질문 후 지원자가 오래 응답하지 않아 시스템이 중단시켰다",
    }.get(reason, "시스템이 답변을 중단시켰다")
    return (
        f"\n**[전제]** 직전 답변은 지원자가 스스로 마친 것이 아니라 {label}. "
        "이 사실을 지원자 탓으로 언급하거나 캐묻지 말고 자연스럽게 다음 질문으로 넘어간다.\n"
    )

async def generate_turn(
    *, stage: Stage, persona: Persona, info: ExtractedInfo, resume: str,
    history: str, user_answer: str,
    truncated_reason: str | None = None,
) -> LLMTurn:
    """한 턴의 면접관 발화를 생성한다. JSON 파싱 실패 시 1회 재시도."""
    system = _system_prompt(info, resume, stage)
    if truncated_reason:
        system += _truncation_premise(truncated_reason)
    user = _turn_instruction(stage, persona, history, user_answer)
    for attempt in range(2):
        try:
            data = await _ask_json(system, user)

            #LLM이 점수를 "80점", "80" 같은 문자열로 줬을 경우 숫자만 추출하는 방어 로직
            if "score" in data and isinstance(data["score"], str):
                import re
                numbers = re.findall(r'-?\d+', data["score"])
                if numbers:
                    data["score"] = int(numbers[0])
                else:
                    data["score"] = -1

            turn = LLMTurn(**data)
            return _clamp_to_set(turn, persona)
        except Exception as e:
            print(f"[에러 발생] LLM 응답 파싱 실패: {e}, 받은 데이터: {data}")
            if attempt == 1:
                return LLMTurn(
                    dialogue="네, 잘 들었습니다. 다음 질문으로 넘어가겠습니다.",
                    score=-1,
                    expression_id=ExpressionID.NEUTRAL.value,
                    gesture_id=GestureID.LISTENING_NOD.value,
                )
    raise RuntimeError("unreachable")

# ---------------------------------------------------------------------------
# (2.5) 개입(바지인) - 주제 이탈 판정 
# ---------------------------------------------------------------------------
_OFF_TOPIC_TOOL = {
    "type": "function",
    "function" : {
        "name" : "judge_off_topic",
        "description": "지원자의 답변이 질문 주제에서 벗어났는지 판정합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "is_off_topic": {
                    "type": "boolean", 
                    "description": "명백히 주제에서 벗어났으면 true, 애매하거나 사례 도입/설명 중일 가능성이 있으면 false",
                    },
            },
            "required": ["is_off_topic"]
        },
    },
}

async def judge_off_topic(question:str, partial_answer: str) -> bool:
    """직전 질문과 지금까지의 부분 전사만으로 주제 이탈 여부를 판정한다.
    설계서 3.5: 보수적으로 판정한다 - 애매하면 False(이탈 아님)."""
    system = (
        "당신은 면접 답변이 질문 주제에서 벗어났는지 판정하는 판정기다.\n"
        "반드시 보수적으로 판정하라. 애매하면 무조건 이탈 아님(false)으로 판정한다.\n"
        "답변 초반은 사례를 도입하는 중일 수 있으므로, 겉보기에 다른 화제 같아도 질문과 연결될 가능성이 있으면 이탈로 보지 않는다.\n"
        "예시:\n"
        "- 이탈 아님: 질문 '협업 경험'에 답변이 '대학교 때 축구 동아리에서...'로 시작 -> 사례 도입일 수 있음.\n"
        "- 이탈 맞음: 질문 '협업 경험'에 답변이 '어제 축구 경기 보셨어요? 후반전에...' -> 질문과 무관한 화제로 완전히 벗어남.\n"
        "판정 이유는 생성하지 말고 boolean 값만 반환한다."
    )
    user = f"[질문] {question}\n[지금까지의 답변] {partial_answer}"
    try:
        resp = await _get_client().chat.completions.create(
            model=_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[_OFF_TOPIC_TOOL],
            tool_choice={"type": "function", "function": {"name": "judge_off_topic"}},
        )
        args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        return bool(args.get("is_off_topic", False))
    except Exception as e:
        print(f"[에러 발생] 주제 이탈 판정 실패: {e}")
        return False

INTERVENTION_EXPRESSION_ID = 5
INTERVENTION_GESTURE_ID = 8

def _intervention_system_prompt(info: ExtractedInfo) -> str:
    skills = ", ".join(info.mentioned_skills) or "미언급"
    return (
        f"당신은 {info.company_name or '지원 기업'}의 {info.job_role or '해당 직무'} 채용 면접관이다.\n"
        f"지원자 보유 기술: {skills}\n\n"
        "지금 당신은 부정적인 태도 상태이며, 지원자의 답변 도중 말을 끊고 개입하려 한다.\n"
        "규칙:\n"
        "1) 대사는 원본 질문보다 짧게, 2문장 이내로 작성한다.\n"
        "2) 지금까지의 답변을 인용하되, 부분 전사는 정확도가 낮을 수 있으므로 핵심 명사 한두 개만 짧게 인용한다. 문장 전체를 그대로 옮기지 않는다.\n"
        "3) 단호한 어조를 쓰되 모욕적이지 않게 한다.\n"
        "4) 자연스러운 한국어 존댓말, TTS로 출력되므로 마크다운/이모지/괄호설명 없이 말로만.\n"
        "5) 반드시 지정된 JSON 스키마로만 응답한다.\n"
    )

def _intervention_instruction(bargein_type: str, reason: str, question: str, partial_answer: str) -> str:
    if bargein_type == "REDIRECT":
        shape = (
            "[대사 구성] [끊는 신호] + [지금까지 답변의 어떤 부분이 주제를 벗어났는지 짧게 지적] + [원본 질문을 그대로 다시 제시]\n"
            f"[원본 질문] {question}\n"
        ) 
    else:
        shape = (
            "[대사 구성] [끊는 신호] + [간결한 마무리 지적] + [다음으로 넘어간다는 짧은 전환 문구]\n"
            "다음 질문 본문은 별도로 이어붙으므로 여기서는 전환 문구까지만 작성한다.\n"
        )
    return (
        f"[개입 유형] {bargein_type} (사유: {reason})\n"
        f"[지금까지 지원자가 한 말] {partial_answer}\n"
        f"{shape}\n"
        "반드시 아래 순수 JSON으로만 응답하라:\n"
        "{\n"
        '  "dialogue": "면접관 개입 대사(한국어)",\n'
        f'  "expression_id": {INTERVENTION_EXPRESSION_ID},\n'
        f'  "gesture_id": {INTERVENTION_GESTURE_ID}\n'
        "}\n"
        "score, score_reason 필드는 포함하지 마라. 이 대사는 채점 대상이 아니다."
    )

async def generate_intervention(
        *, stage: Stage, persona: Persona, bargein_type: str, reason: str, question: str, partial_answer: str, info: ExtractedInfo
) -> LLMTurn:
    """면접관이 답변 도중 끼어들 때의 개입 대사를 생성한다. score는 항상 -1로 고정."""
    system = _intervention_system_prompt(info)
    user = _intervention_instruction(bargein_type, reason, question, partial_answer)
    try:
        data = await _ask_json(system, user)
        data["score"] = -1
        data.setdefault("score_reason", "")
        turn = LLMTurn(**data)
        return _clamp_to_set(turn, persona)
    except Exception as e:
        print(f"[에러 발생] genereate_intervention 실패: {e}")
        fallback = (
            "잠시만요, 질문으로 다시 돌아가죠." if bargein_type == "REDIRECT" else "네, 거기까지 듣겠습니다. 다음 질문으로 넘어가죠."
        )
        return LLMTurn(
            dialogue=fallback, 
            score=-1,
            expression_id=ExpressionID.NEUTRAL.value,
            gesture_id=GestureID.LISTENING_NOD.value
        )



# ---------------------------------------------------------------------------
# (3) 종료 피드백
# ---------------------------------------------------------------------------
async def generate_feedback(*, company: str, job_title: str, transcript: str,
                            stage_scores: list[tuple[str, int]],
                            avg_speaking_time: float, total_pauses: int) -> dict:
    """면접 전체를 종합한 피드백(강점/개선점/총평)을 JSON으로 생성."""
    scored = [s for _, s in stage_scores if s >= 0]
    overall = round(sum(scored) / len(scored)) if scored else 0
    system = "당신은 면접 코치다. 아래 면접 기록을 바탕으로 건설적 피드백을 한국어로 작성한다. JSON으로만 응답한다."
    user = (
        f"기업/직무: {company} / {job_title}\n"
        f"평균 발화시간(초): {avg_speaking_time:.1f}, 총 침묵 횟수: {total_pauses}\n"
        f"단계별 점수: {stage_scores}\n"
        f"전체 대화 기록:\n{transcript}\n\n"
        "추가 요구사항:\n"
        "1) 전체 답변 텍스트를 분석하여 무의미한 필러 단어(어, 그, 뭐랄까 등)의 남용이나 말더듬 현상을 평가하여 'filler_score' (0~10점 정수)를 매기세요. (필러가 적을수록 10점)\n"
        "2) 답변 내용의 충실도와 밀도(불필요하게 장황하지 않은지)를 평가하여 'density_score' (0~10점 정수)를 매기세요. (충실할수록 10점)\n\n"
        "다음 JSON으로만 응답:\n"
        '{ "strengths": "강점 2~3문장", "improvements": "개선점 2~3문장", "summary": "총평 2문장", "filler_score": 10, "density_score": 10 }'
    )
    try:
        data = await _ask_json(system, user)
    except Exception:
        data = {"strengths": "", "improvements": "", "summary": "", "filler_score": 5, "density_score": 5}
    data["overall_score"] = overall
    return data

# ---------------------------------------------------------------------------
# (4) 예열 — 첫 실제 호출의 커넥션 수립 비용을 Setup 단계에서 미리 지불
# ---------------------------------------------------------------------------
async def warmup() -> None:
    try:
        await _get_client().chat.completions.create(
            model=_model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        print("[llm] warmup 완료")
    except Exception as e:
        print(f"[llm] warmup 실패(무시 가능): {e}")