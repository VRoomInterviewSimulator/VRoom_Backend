"""도메인 정의: 열거형, 비언어 ID 코드표, 그리고 모든 통신 스키마.

이 파일은 '계약서'다. 여기 정의된 필드명은 Unity 쪽 C# 클래스와 1:1로 맞춰야 한다.
Unity 의 JsonUtility 는 클래스에 없는 필드를 **조용히 무시**하므로,
백엔드만 고치면 값이 사라지고 에러도 나지 않는다. 반드시 양쪽을 함께 수정할 것.

  BehaviorPacket   <-> Assets/Scripts/Backend/BehaviorPacket.cs
  FeedbackReport   <-> 같은 파일의 FeedbackReport
  ExpressionID     <-> InterviewerExpression.Apply(int) 의 switch 분기
"""
from __future__ import annotations

from enum import Enum, IntEnum
from typing import List

from pydantic import BaseModel, Field


# ===========================================================================
# 1. 면접 진행 (단계 / 페르소나)
# ===========================================================================
class Stage(str, Enum):
    """면접 시나리오 단계. 사용자가 답변을 마칠 때마다 한 칸씩 전진한다."""
    INIT = "INIT"                  # 세션 생성 직후(아직 첫 질문 전)
    SELF_INTRO = "SELF_INTRO"      # 1. 자기소개 요구 — 동적 페르소나 정보 추출원
    TECH_Q1 = "TECH_Q1"            # 2. 직무 맞춤 기술 질문
    FOLLOWUP_1 = "FOLLOWUP_1"      # 3. 채점 기반 1차 꼬리질문 (페르소나 가변의 핵심 분기점)
    FOLLOWUP_2 = "FOLLOWUP_2"      # 4. 2차 꼬리질문 (화제 전환)
    BEHAVIORAL = "BEHAVIORAL"      # 5. 인성/조직 적합성 질문
    CLOSING = "CLOSING"            # 6. 마무리 질문
    DONE = "DONE"                  # 종료 — 피드백 리포트 산출


# 단계 전진 순서. INIT 은 시작점이라 여기 포함하지 않는다.
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
    """면접관 태도. 직전 답변 점수로 코드가 결정한다(LLM 이 정하지 않는다)."""
    POSITIVE = "POSITIVE"    # 긍정형 — 안정감 부여
    NEUTRAL = "NEUTRAL"      # 중립 — 경청
    NEGATIVE = "NEGATIVE"    # 부정 — 압박. 개입(G5)이 열리는 유일한 상태


class Condition(str, Enum):
    """실험 조건. 세션 생성 시 Unity 가 init 으로 알려준다."""
    A = "A"   # 정적 턴제 (페르소나 고정, 개입 없음)
    B = "B"   # 가변 페르소나 (개입 없음)
    C = "C"   # 가변 페르소나 + 능동 개입


def persona_from_score(score: int, consecutive_low: int, condition: str = "C") -> Persona:
    """답변 점수 -> 페르소나. 코드가 결정권을 가져 전이를 안정적으로 통제한다.

    score < 0 은 채점 대상이 아닌 단계(자기소개 등)를 뜻하므로 중립을 유지한다.
    조건 A 는 정적 턴제이므로 항상 중립 [실험 게이팅 지점 1/2].
    """
    if condition == Condition.A.value:
        return Persona.NEUTRAL
    if score < 0:
        return Persona.NEUTRAL
    if score >= 70:
        return Persona.POSITIVE
    if score >= 40:
        return Persona.NEUTRAL
    return Persona.NEGATIVE


def persona_value_from_score(score: int, consecutive_low: int, condition: str = "C") -> float:
    """답변 점수(0~100) -> 연속 감정 강도(-1.0 ~ +1.0). Unity Animator 의 Emotion 축 값.

    50점을 중립(0)으로 두고 선형 매핑하며, 연속 저점이 쌓이면 음의 방향으로 고착시킨다.
    조건 A 는 항상 0.0 [실험 게이팅 지점 2/2].
    """
    if condition == Condition.A.value:
        return 0.0
    if score < 0:
        return 0.0
    value = (score - 50) / 50.0
    value -= 0.2 * min(consecutive_low, 3)
    return max(-1.0, min(1.0, value))


# ===========================================================================
# 2. 개입(끼어들기) 분류
# ===========================================================================
class BargeInType(str, Enum):
    """개입 유형. 단계 전진 여부가 갈린다."""
    REDIRECT = "REDIRECT"   # Type A: 주제 이탈 -> 같은 질문 재요청, 단계 유지
    CUTOFF = "CUTOFF"       # Type B: 길이/침묵 -> 답변 조기 종료, 단계 전진


class BargeInReason(str, Enum):
    """개입 발동 원인. 어느 트리거가 잡았는지 구분한다."""
    OFF_TOPIC = "OFF_TOPIC"          # 부분 전사 기반 LLM 이탈 판정 (Type A)
    LONG_ANSWER = "LONG_ANSWER"      # Unity 로컬 타이머: 발화가 너무 길다 (Type B)
    LONG_SILENCE = "LONG_SILENCE"    # Unity 로컬 타이머: 질문 후 무응답 (Type B)


class TurnPhase(str, Enum):
    """웹캠 턴 위상. vision_process 및 Unity TurnPhase 상수와 문자열이 일치해야 한다."""
    NORMAL = "NORMAL"          # 개입 없는 일반 답변 구간 — 채점 대상
    TRUNCATED = "TRUNCATED"    # 개입으로 잘린 답변 구간 — 로그 전용
    REACTION = "REACTION"      # 개입 직후 반응 구간 — 로그 전용, 논문 핵심 종속변인
    REANSWER = "REANSWER"      # Type A 재답변 구간 — 채점 대상


# ===========================================================================
# 3. 비언어 ID 코드표 (Unity 와 공유하는 약속)
# ===========================================================================
class ExpressionID(IntEnum):
    """얼굴 표정 ID. InterviewerExpression.Apply(int) 의 분기와 1:1."""
    NEUTRAL = 0        # 무표정
    WARM_SMILE = 1     # 온화한 미소 (긍정)
    SLIGHT_FROWN = 2   # 미간 찌푸림 (부정)
    ATTENTIVE = 3      # 관심/경청
    THINKING = 4       # 생각 중
    FIRM_STOP = 5      # [개입] 단호/제지 — 블렌드셰이프 프리셋으로 구현됨


class GestureID(IntEnum):
    """제스처 ID. 현재 Unity 는 이 값을 사용하지 않는다(애니메이션 리소스 미확보).

    enum 을 유지하는 이유는 백엔드 로그와 세션 로그의 가독성 때문이다.
    """
    IDLE = 0            # 정지
    DEEP_NOD = 1        # 깊게 끄덕임 (긍정)
    HEAD_TILT = 2       # 고개 갸우뚱 (약한 부정)
    ARMS_CROSSED = 3    # 팔짱 (강한 부정) — 개입 시 실제로 발신하는 값
    PEN_FIDGET = 4      # 펜 만지작 (강한 부정)
    WELCOME = 5         # 시작 안내 제스처
    REVIEW_RESUME = 6   # 이력서 검토 = '생각 중' 더미 모션
    LISTENING_NOD = 7   # 경청 끄덕임 (중립)
    # ── 아래는 예약. 애니메이션 리소스 미확보로 미구현. 백엔드가 발신하지 않는다. ──
    PALM_STOP = 8       # [예약] 손바닥 들어 제지
    LEAN_FORWARD = 9    # [예약] 상체 앞으로 기울임


# 페르소나별로 LLM 이 고를 수 있는 행동 세트.
# llm._clamp_to_set() 이 이 범위를 벗어난 ID 를 강제 보정한다.
# 주의: 개입 전용 ID(FIRM_STOP 등)를 여기 넣으면 안 된다.
#       평범한 꼬리질문에서도 LLM 이 제지 표정을 고를 수 있게 되어버린다.
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


# ===========================================================================
# 4. LLM 입출력 스키마
# ===========================================================================
class LLMTurn(BaseModel):
    """LLM 이 단일 JSON 으로 강제 출력해야 하는 구조 (Structured Output)."""
    dialogue: str = Field(description="면접관이 말할 한국어 대사")
    score: int = Field(default=-1, ge=-1, le=100,
                       description="직전 사용자 답변 점수 0~100, 채점 대상이 아니면 -1")
    score_reason: str = Field(default="",
                              description="채점 근거 한 문장. 리포트용이며 음성으로 나가지 않는다")
    expression_id: int = Field(default=ExpressionID.NEUTRAL.value)   # 이번 턴의 표정 ID
    gesture_id: int = Field(default=GestureID.IDLE.value)            # 이번 턴의 제스처 ID


class ExtractedInfo(BaseModel):
    """자기소개 답변에서 뽑아낸 동적 페르소나 슬롯.

    이후 모든 질문 생성의 System Prompt 에 주입되어 '이 지원자 전용 면접관'을 만든다.
    """
    company_name: str = ""                                        # 지원 기업명 (없으면 Unity init 값)
    job_role: str = ""                                            # 지원 직무
    experience_level: str = "신입"                                # 신입 | 주니어 | 중급 | 시니어
    mentioned_skills: List[str] = Field(default_factory=list)     # 언급된 기술 스택
    key_strengths: List[str] = Field(default_factory=list)        # 지원자가 강조한 강점


# ===========================================================================
# 5. 요청 스키마 (외부 -> 백엔드)
# ===========================================================================
class InitRequest(BaseModel):
    """Unity 가 /ws/control 연결 직후 보내는 세션 초기화 정보."""
    session_id: str                 # 세션 식별자. 실험 시 "P01_C" 형태 권장
    company: str = ""               # 지원 기업
    job_title: str = ""             # 지원 직무
    resume: str = ""                # 이력서 원문
    condition: str = "C"            # 실험 조건 A/B/C


class AnswerRequest(BaseModel):
    """STT 워커가 /process 로 전사 텍스트를 POST 할 때의 본문 (레거시 HTTP 경로)."""
    session_id: str = ""
    text: str                                       # 전사된 사용자 답변
    features: dict = Field(default_factory=dict)    # {speakingTime, pauseCount, averageVolume, ...}


# ===========================================================================
# 6. 응답 스키마 (백엔드 -> Unity)
# ===========================================================================
class BehaviorPacket(BaseModel):
    """백엔드 -> Unity '동적 행동 지시 패킷'.

    Unity 의 JsonUtility 가 그대로 역직렬화할 수 있도록 모든 필드를
    1차원 primitive 로만 구성한다(중첩 dict / 임의 key map 금지).
    """
    type: str = "interviewer_turn"   # "interviewer_turn" | "thinking" | "ignored"
    session_id: str = ""
    stage: str = ""                  # 이 패킷이 속한 단계 (Stage 값)
    persona: str = Persona.NEUTRAL.value
    persona_value: float = 0.0       # 연속 감정 강도 -1.0~+1.0 -> Animator Emotion 축
    dialogue: str = ""               # 면접관 대사 (자막 + TTS 입력)
    expression_id: int = ExpressionID.NEUTRAL.value
    gesture_id: int = GestureID.IDLE.value
    score: int = -1                  # 직전 답변 점수 (-1 = 해당 없음)
    is_final: bool = False           # True 면 Unity 가 피드백 리포트를 요청한다
    bargein_type: str = ""           # "" | "CUTOFF"(발화1) | "CUTOFF_QUESTION"(발화2) | "REDIRECT"


class StageScore(BaseModel):
    """단계별 답변 점수 한 건."""
    stage: str
    score: int


class InterviewScore(BaseModel):
    """결과 UI 의 10개 셀과 1:1. 필드명을 프론트와 동일하게 유지해야 한다."""
    # ── 시각 4항목 (웹캠) ───────────────────────────────
    gaze: int = 0          # 시선 처리
    gesture: int = 0       # 손짓 처리
    posture: int = 0       # 몸짓 처리
    expression: int = 0    # 표정 처리
    # ── 음성 6항목 ─────────────────────────────────────
    voiceVolume: int = 0   # 음성 크기
    voiceSpeed: int = 0    # 음성 속도
    answerLength: int = 0  # 답변 길이 (물리적 시간 + LLM 내용 밀도)
    fillerWords: int = 0   # 추임새 남용
    accuracy: int = 0      # 답변 품질 (LLM 채점 평균)
    responseTime: int = 0  # 반응 시간


class FeedbackReport(BaseModel):
    """면접 종료 후 Unity 결과 화면에 시각화할 종합 피드백."""
    type: str = "feedback_report"
    session_id: str = ""
    scores: InterviewScore = Field(default_factory=InterviewScore)   # 10항목 세부 점수
    overall_score: int = 0                                           # 종합 점수(100점 만점)
    stage_scores: List[StageScore] = Field(default_factory=list)     # 단계별 답변 점수
    strengths: str = ""                                              # 강점 (LLM 총평)
    improvements: str = ""                                           # 개선점 (LLM 총평)
    summary: str = ""                                                # 총평 2문장
    avg_speaking_time: float = 0.0                                   # 평균 발화 시간(초)
    total_pauses: int = 0                                            # 총 의미 있는 침묵 횟수