"""환경 변수(.env) 로딩. 모든 설정은 여기 한곳에서 관리한다."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: str = "openai"          # "openai" | "groq"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # TTS 워커(Node B)
    tts_worker_url: str = "http://host.docker.internal:8001/process"
    tts_ws_url: str = "ws://host.docker.internal:8001/ws/tts"
    
    # 서버
    host: str = "0.0.0.0"
    port: int = 8080
    proxy_audio_to_stt: bool = False
    skip_tts: bool = False
    
    # 첫 질문 정책 / 프리웜
    template_first_question: bool = True   # True면 첫 질문은 LLM 없이 템플릿으로 생성
    warmup_llm_on_prepare: bool = True     # prepare 시 OpenAI 커넥션 예열(1토큰 요청)

    def resolve_llm(self) -> tuple[str | None, str, str]:
        if self.llm_provider == "groq":
            return ("https://api.groq.com/openai/v1", self.groq_api_key, self.groq_model)
        return (None, self.openai_api_key, self.openai_model)  # openai 기본 base_url


settings = Settings()

class ScoringConfig:
    """
    비언어적 평가 항목(0~10점) 채점을 위한 조절 가능한 상수 모음
    
    [기본 점수 계산 원리]
    각 평가 항목(Voice Volume, Speed, Answer Length, Response Time)은 전체 평균이 아닌
    "매 턴(Turn)마다 개별적으로 점수(10점 만점 기준)를 산출한 뒤, 
    이 턴별 점수들을 모두 더해 평균을 내는 방식"으로 최종 점수를 확정합니다.
    턴별 기본 감점 공식: 10 - int(abs(해당 턴의 데이터 - MEAN) / TOLERANCE)
    
    [참고: Unity 클라이언트 설정(VoiceActivityDetector.cs)과의 관계]
    백엔드는 유니티에서 1차 가공되어 넘어온 피쳐(Feature) 데이터를 기반으로 최종 감점만 수행합니다.
    피쳐 데이터를 추출하는 원천 기준값 2개는 백엔드가 아닌 유니티 인스펙터에서 직접 조절해야 합니다.
    - meaningfulPauseThreshold (기본 0.4초): 침묵이 몇 초 이상 지속되어야 '의미 있는 퍼즈(Pause)' 1회로 카운트할 것인가?
    - lowVolumeRatioThreshold (기본 0.3): 특정 프레임의 볼륨이 '전체 평균 볼륨의 30% 이하'로 떨어질 때만 '작은 목소리'로 카운트.
    
    [기타 평가 항목 (LLM 정성 평가)]
    아래 항목들은 수식(상수)에 의한 기계적 계산이 아닌, LLM의 정성 평가에 의해 0~10점 척도로 매겨집니다.
    - Accuracy (답변 품질): 면접 중 각 턴마다 LLM이 평가한 질문 대비 답변 퀄리티 점수(0~100)를 10점 만점으로 환산한 전체 평균.
    - Density Score (내용 밀도): 면접 종료 후 스크립트 전체를 보고 "쓸데없이 말만 길지 않고 핵심이 있는가"를 10점 만점으로 평가. 이 점수는 Answer Length(물리적 길이) 점수와 50:50 비중으로 합산되어 최종 답변 길이 점수가 됩니다.
    - Filler Words (추임새): 면접 종료 후 스크립트에서 "어...", "그니까..." 등 불필요한 추임새 남용 여부를 10점 만점으로 평가.
    """
    
    # 목소리 크기 (RMS)
    # 공식: 10 - int(abs(평균볼륨 - VOICE_VOLUME_MEAN) / VOICE_VOLUME_TOLERANCE)
    VOICE_VOLUME_MEAN = 0.1
    VOICE_VOLUME_TOLERANCE = 0.05  # 오차가 이 범위를 초과할 때마다(계단식) 1점 감점
    
    # 페널티: 목소리의 분산(들쭉날쭉함)이 THRESHOLD를 넘으면 PENALTY만큼 추가 감점
    VOLUME_VARIANCE_THRESHOLD = 0.05
    VOLUME_VARIANCE_PENALTY = 1
    
    # 페널티: 목소리가 지나치게 작은 구간의 비율이 THRESHOLD(예: 30%)를 넘으면 PENALTY 감점
    LOW_VOLUME_RATIO_THRESHOLD = 0.3
    LOW_VOLUME_RATIO_PENALTY = 1
    
    # 발화 속도 (초당 글자 수 CPS)
    # 공식: 10 - int(abs(평균속도 - VOICE_SPEED_MEAN) / VOICE_SPEED_TOLERANCE)
    VOICE_SPEED_MEAN = 5.0
    VOICE_SPEED_TOLERANCE = 1.0    # 허용 오차를 벗어날 때마다 1점 감점
    
    # 페널티: 한 턴당 의미 있는 퍼즈(침묵) 횟수가 허용치(ALLOWANCE)를 넘은 횟수만큼 추가 감점
    PAUSE_ALLOWANCE = 1.0
    
    # 반응 속도 (초)
    # 공식: 턴별로 10 - int(abs(실제응답시간 - 상황별 MEAN) / 1.5) 계산 후 전체 평균
    RESPONSE_TIME_INTRO_MEAN = 1.5      # 자기소개 시 기대 응답시간
    RESPONSE_TIME_FOLLOWUP_MEAN = 3.5   # 꼬리질문 시 기대 응답시간 (생각할 시간이 더 필요함)
    RESPONSE_TIME_TOLERANCE = 1.5       # 이 범위를 벗어날 때마다 1점 감점
    
    # 이상적인 답변 길이 (초)
    # 길이 미달 시: int((평균답변시간 / ANSWER_LENGTH_MIN) * 10) (예: 최소치의 절반만 답하면 5점)
    # 길이 초과 시: 10 - int((평균답변시간 - ANSWER_LENGTH_MAX) / 10) (예: 최대치에서 10초 초과할 때마다 1점 감점)
    ANSWER_LENGTH_MIN = 40
    ANSWER_LENGTH_MAX = 80

class VisionScoringConfig:
    """
    웹캠 기반 비언어적 평가 항목(gaze / gesture / posture / expression) 채점 상수.

    [역할 분담]
    - Vision Worker(aggregator.py) : 프레임 -> 원시 피쳐 1차 가공 (임계값은 그쪽 상수)
    - 이 클래스                    : 원시 피쳐 -> 0~10점 감점 (실험 튜닝은 여기서만)

    음성 항목과 동일하게 '턴별로 점수를 매기고 전체 평균' 방식을 사용한다.
    """
    # 이 비율 미만으로 얼굴이 검출된 턴은 신뢰 불가로 채점에서 제외
    MIN_FACE_RATIO = 0.5
    MIN_POSE_RATIO = 0.5
    UNMEASURABLE_SCORE = 5

    # ---- 1. 시선 (gaze) ----
    GAZE_RATIO_FULL = 0.80        # 정면 응시 비율이 이 이상이면 만점
    GAZE_RATIO_ZERO = 0.30        # 이 이하이면 0점 (사이는 선형 보간)
    GAZE_JITTER_TOLERANCE = 5.0   # 두부 각도 표준편차(deg) 허용치
    GAZE_JITTER_STEP = 2.5      # 초과분이 이만큼 늘 때마다 1점 감점

    # ---- 2. 자세 (posture) ----
    POSTURE_TILT_TOLERANCE = 8.0    # 어깨 기울기 평균(deg)
    POSTURE_TILT_STEP = 3.0
    POSTURE_SWAY_TOLERANCE = 0.030   # 어깨너비 정규화 상체 흔들림 표준편차
    POSTURE_SWAY_STEP = 0.020
    POSTURE_DRIFT_ENABLED = False
    POSTURE_DRIFT_TOLERANCE = 0.25
    POSTURE_DRIFT_STEP = 0.10

    # ---- 3. 손짓 = 손 사용 빈도(gesture) ----
    GESTURE_USAGE_HARD_ZERO = 0.02
    GESTURE_ZERO_SCORE = 2   
    GESTURE_USAGE_MIN = 0.15      # 이 미만 = 손을 거의 안 씀(경직)
    GESTURE_USAGE_MAX = 0.75      # 이 초과 = 과도한 손짓
    GESTURE_UNDER_STEP = 0.05     # 하한 미달분이 이만큼 늘 때마다 1점 감점
    GESTURE_OVER_STEP = 0.15      # 상한 초과분이 이만큼 늘 때마다 1점 감점
    GESTURE_EXTENT_MIN = 0.15     # 손이 보이지만 한 자리에 굳어 있으면 감점
    GESTURE_EXTENT_PENALTY = 1
    GESTURE_FACE_TOUCH_ALLOWANCE = 1   # 턴당 얼굴 만지기 허용 횟수

    # ---- 4. 표정 (expression) ----
    EXPRESSION_RIGID_THRESHOLD = 0.012  # 표정 변화량이 이 이하이면 '굳은 표정'
    EXPRESSION_RIGID_PENALTY = 2
    EXPRESSION_FROWN_TOLERANCE = 0.15   # 찌푸림 프레임 비율 허용치
    EXPRESSION_FROWN_STEP = 0.15
    EXPRESSION_BLINK_MIN = 8.0          # 분당 눈깜빡임 정상 하한 (응시 경직)
    EXPRESSION_BLINK_MAX = 32.0         # 상한 (긴장)
    EXPRESSION_BLINK_STEP = 10.0
    EXPRESSION_SMILE_BONUS_RATIO = 0.15 # 이 이상 미소 유지 시 +1
    EXPRESSION_BLINK_ENABLED = False