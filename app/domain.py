"""
면접 도메인 정의: 단계(Stage), 페르소나(Persona), 그리고 Unity로 넘길
Expression_ID / Gesture_ID 코드 테이블. 이 ID 값들은 Unity Animator의
Expression_ID / Gesture_ID 파라미터와 1:1로 매핑되도록 약속한 값이다.
값을 바꾸고 싶으면 Unity 쪽 매핑과 함께 여기만 수정하면 된다.
"""
from __future__ import annotations

from enum import Enum, IntEnum
from typing import List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 면접 시나리오 단계 (면접관 페르소나 및 면접 시나리오 설계 PDF 기준)
# 자기소개 -> 기술1 -> 꼬리1(분기점) -> 꼬리2 -> 인성 -> 마무리
# ---------------------------------------------------------------------------
class Stage(str, Enum):
    INIT = "INIT"
    SELF_INTRO = "SELF_INTRO"      # 1. 자기소개 요구
    TECH_Q1 = "TECH_Q1"            # 2. 직무 맞춤 기술 질문 1
    FOLLOWUP_1 = "FOLLOWUP_1"      # 3. 채점 기반 1차 꼬리질문 (페르소나 가변 핵심 분기점)
    FOLLOWUP_2 = "FOLLOWUP_2"      # 4. 2차 꼬리질문
    BEHAVIORAL = "BEHAVIORAL"      # 5. 인성/조직 적합성 질문
    CLOSING = "CLOSING"            # 6. 마무리 질문
    DONE = "DONE"                  # 종료 -> 피드백 리포트 산출


# 사용자가 답변을 마칠 때마다 다음 단계로 넘어가는 고정 순서.
STAGE_ORDER: List[Stage] = [
    Stage.SELF_INTRO,
    Stage.TECH_Q1,
    Stage.FOLLOWUP_1,
    Stage.FOLLOWUP_2,
    Stage.BEHAVIORAL,
    Stage.CLOSING,
    Stage.DONE,
]


class Persona(str, Enum):
    POSITIVE = "POSITIVE"    # 긍정형: 안정감 부여
    NEUTRAL = "NEUTRAL"      # 중립/경청
    NEGATIVE = "NEGATIVE"    # 부정/압박형


class ExpressionID(IntEnum):
    NEUTRAL = 0
    WARM_SMILE = 1     # 온화한 미소 (긍정)
    SLIGHT_FROWN = 2   # 미간 찌푸림 (부정)
    ATTENTIVE = 3      # 관심/경청
    THINKING = 4       # 생각 중(검토)


class GestureID(IntEnum):
    IDLE = 0
    DEEP_NOD = 1        # 깊게 끄덕임 (긍정)
    HEAD_TILT = 2       # 고개 갸우뚱 (약한 부정)
    ARMS_CROSSED = 3    # 팔짱 (강한 부정)
    PEN_FIDGET = 4      # 펜 만지작 (강한 부정)
    WELCOME = 5         # 시작 안내 제스처
    REVIEW_RESUME = 6   # 이력서 검토 = '생각 중' 더미 모션
    LISTENING_NOD = 7   # 경청 끄덕임 (중립)


# 페르소나별로 LLM이 고를 수 있는 '미리 정의된 행동 세트'.
# (요구사항: 가변 페르소나 상태에 맞춰 정의된 세트 안에서 제스처/표정을 동적 선택)
PERSONA_BEHAVIOR_SET = {
    Persona.POSITIVE: {
        "expressions": [ExpressionID.WARM_SMILE, ExpressionID.ATTENTIVE],
        "gestures": [GestureID.DEEP_NOD, GestureID.LISTENING_NOD],
    },
    Persona.NEUTRAL: {
        "expressions": [ExpressionID.NEUTRAL, ExpressionID.ATTENTIVE],
        "gestures": [GestureID.LISTENING_NOD, GestureID.IDLE, GestureID.HEAD_TILT],
    },
    Persona.NEGATIVE: {
        "expressions": [ExpressionID.SLIGHT_FROWN, ExpressionID.NEUTRAL],
        "gestures": [GestureID.HEAD_TILT, GestureID.ARMS_CROSSED, GestureID.PEN_FIDGET],
    },
}


def persona_from_score(score: int, consecutive_low: int) -> Persona:
    """답변 점수 -> 페르소나 결정 (코드가 결정권을 가져 안정적으로 제어).

    score < 0 은 채점 대상이 아닌 단계(자기소개 등)를 의미하므로 중립 유지.
    연속 저점(consecutive_low)이 누적되면 압박을 고착(강한 부정)시킨다.
    """
    if score < 0:
        return Persona.NEUTRAL
    if score >= 70:
        return Persona.POSITIVE
    if score >= 40:
        return Persona.NEUTRAL
    return Persona.NEGATIVE

def persona_value_from_score(score: int, consecutive_low: int) -> float:
    """답변 점수(0~100) -> 연속 감정 강도(-1.0 ~ +1.0).
    55점을 중립(0)으로 두고 선형 매핑. score < 0(채점 대상 아님)은 0.0.
    연속 저점이 누적되면 음의 방향으로 압박을 고착.
    """
    if score < 0:
        return 0.0
    value = (score - 50) / 50.0;
    value -= 0.2 * min(consecutive_low, 3) 
    return max(-1.0, min(1.0, value))

# ---------------------------------------------------------------------------
# Pydantic 스키마
# ---------------------------------------------------------------------------
class LLMTurn(BaseModel):
    """LLM이 단일 JSON으로 강제 출력해야 하는 구조 (Structured Output)."""
    dialogue: str = Field(description="면접관이 말할 한국어 대사")
    score: int = Field(default=-1, ge=-1, le=100, description="직전 사용자 답변 점수 0~100, 채점 대상 아니면 -1")
    score_reason: str = Field(default="", description="채점 근거(피드백 리포트용, 음성으로 출력되지 않음)")
    expression_id: int = Field(default=ExpressionID.NEUTRAL.value)
    gesture_id: int = Field(default=GestureID.IDLE.value)


class InitRequest(BaseModel):
    """Unity가 WebSocket 연결 직후 보내는 면접 세션 초기화 정보."""
    session_id: str
    company: str = ""
    job_title: str = ""
    resume: str = ""


class AnswerRequest(BaseModel):
    """STT 워커(Node A)가 전사 텍스트와 음성 피쳐를 백엔드로 POST 할 때의 본문."""
    session_id: str = ""
    text: str
    features: dict = Field(default_factory=dict)  # {speakingTime, pauseCount, averageVolume ...}


class BehaviorPacket(BaseModel):
    """백엔드 -> Unity 로 보내는 '동적 행동 지시 패킷'(JSON).

    Unity의 JsonUtility가 그대로 역직렬화할 수 있도록 모든 필드를 1차원
    primitive 로만 구성한다(중첩 dict / 임의 key map 금지).
    """
    type: str = "interviewer_turn"
    session_id: str = ""
    stage: str = ""
    persona: str = Persona.NEUTRAL.value
    persona_value: float = 0.0 
    dialogue: str = ""
    expression_id: int = ExpressionID.NEUTRAL.value
    gesture_id: int = GestureID.IDLE.value
    score: int = -1
    is_final: bool = False


class StageScore(BaseModel):
    stage: str
    score: int

class InterviewScore(BaseModel):
    """Unity 결과 UI의 10개 셀과 1:1. 필드명은 프론트와 동일해야 한다."""
    gaze: int = 0          # 시선 처리
    gesture: int = 0       # 손짓 처리
    posture: int = 0       # 몸짓 처리
    expression: int = 0    # 표정 처리
    voiceVolume: int = 0   # 음성 크기
    voiceSpeed: int = 0    # 음성 속도
    answerLength: int = 0  # 음성 길이
    fillerWords: int = 0   # 음성 필러 분석
    accuracy: int = 0      # 음성 LLM 채점 평균
    responseTime: int = 0  # 반응 시간

class FeedbackReport(BaseModel):
    """면접 종료 후 Unity의 3D 결과 UI에 시각화할 종합 피드백."""
    type: str = "feedback_report"
    session_id: str = ""
    scores: InterviewScore = Field(default_factory=InterviewScore)
    overall_score: int = 0
    stage_scores: List[StageScore] = Field(default_factory=list)
    strengths: str = ""
    improvements: str = ""
    summary: str = ""
    avg_speaking_time: float = 0.0
    total_pauses: int = 0

class ExtractedInfo(BaseModel):
    """사용자의 자기소개 답변에서 추출한 동적 페르소나 슬롯.
 
    명세(요구사항/시나리오 [1단계])의 '자기소개 답변이 동적 페르소나를 활성화'를 구현.
    example.py의 extract_interview_info Function Calling 출력 형식과 동일.
    """
    company_name: str = ""                       # 지원 기업명 (없으면 Unity init 값 사용)
    job_role: str = ""                           # 지원 직무
    experience_level: str = "신입"               # 신입 | 주니어 | 중급 | 시니어
    mentioned_skills: List[str] = Field(default_factory=list)   # 언급된 기술 스택
    key_strengths: List[str] = Field(default_factory=list)      # 강조한 강점
