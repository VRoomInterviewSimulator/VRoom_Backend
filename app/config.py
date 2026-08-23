"""환경 변수(.env) 로딩과 채점/개입 상수.

이 파일은 '값'만 담는다. 로직은 두지 않는다.
  - Settings          : .env 로 주입되는 런타임 설정 (환경마다 달라지는 값)
  - ScoringConfig     : 음성 6항목 채점 상수
  - VisionScoringConfig : 웹캠 4항목 채점 상수
  - BargeInConfig     : 개입(끼어들기) 판정 상수

실험 튜닝은 전부 여기서만 한다. 다른 파일에 숫자를 흩뿌리지 않는다.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


# ===========================================================================
# 1. 런타임 설정 (.env)
# ===========================================================================
class Settings(BaseSettings):
    """`.env` 파일에서 읽어오는 환경별 설정. 코드 수정 없이 바꿀 수 있는 값만 둔다."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── LLM 제공자 ───────────────────────────────────────────────────
    llm_provider: str = "openai"                        # 사용할 LLM 제공자 ("openai" | "groq")
    openai_api_key: str = ""                            # OpenAI API 키
    openai_model: str = "gpt-4o-mini"                   # OpenAI 모델명
    groq_api_key: str = ""                              # Groq API 키 (llm_provider=groq 일 때)
    groq_model: str = "llama-3.3-70b-versatile"         # Groq 모델명

    # ── TTS 워커(Node B) 주소 ────────────────────────────────────────
    tts_worker_url: str = "http://host.docker.internal:8001/process"   # HTTP 합성 엔드포인트(레거시)
    tts_ws_url: str = "ws://host.docker.internal:8001/ws/tts"         # 스트리밍 합성 웹소켓(실사용)

    # ── 서버 ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"                               # 바인딩 주소
    port: int = 8080                                    # 바인딩 포트
    proxy_audio_to_stt: bool = False                    # True면 /process HTTP 응답으로 음성을 되돌려준다(호환 모드)
    skip_tts: bool = False                              # True면 TTS 합성을 생략한다(Node B 없이 테스트)

    # ── 첫 질문 정책 / 프리웜 ────────────────────────────────────────
    template_first_question: bool = True                # True면 첫 질문을 LLM 없이 템플릿으로 만든다(지연 0)
    warmup_llm_on_prepare: bool = True                  # prepare 시 OpenAI 커넥션을 미리 연다(1토큰 요청)

    # ── 개입(끼어들기) 실험 스위치 ───────────────────────────────────
    bargein_force_negative: bool = False                # True면 G5(부정 페르소나) 게이트를 우회한다. 검증 전용, 실험 시 반드시 False
    session_log_dir: str = "logs"                       # 세션 로그(JSON) 출력 디렉터리

    def resolve_llm(self) -> tuple[str | None, str, str]:
        """현재 llm_provider 에 맞는 (base_url, api_key, model) 조합을 돌려준다."""
        if self.llm_provider == "groq":
            return ("https://api.groq.com/openai/v1", self.groq_api_key, self.groq_model)
        return (None, self.openai_api_key, self.openai_model)   # openai 는 기본 base_url 사용


settings = Settings()   # 전역 단일 인스턴스. 다른 모듈은 `from .config import settings` 로 쓴다.


# ===========================================================================
# 2. 음성 채점 상수
# ===========================================================================
class ScoringConfig:
    """음성 기반 평가 항목(0~10점)의 채점 상수.

    [채점 원리]
    전체 평균이 아니라 '매 턴마다 개별 점수를 내고 그 평균'을 최종 점수로 쓴다.
    턴별 기본 감점 공식:  10 - int(abs(해당 턴 값 - MEAN) / TOLERANCE)

    [Unity 쪽 설정과의 관계]
    백엔드는 Unity 가 1차 가공해 보낸 피쳐로 감점만 한다. 원천 기준값 2개는
    Unity 인스펙터(VoiceActivityDetector)에서 조절해야 한다.
      - meaningfulPauseThreshold : 몇 초 이상 침묵해야 '의미 있는 퍼즈' 1회인가
      - lowVolumeRatioThreshold  : 평균 볼륨의 몇 % 이하를 '작은 목소리'로 볼 것인가

    [여기서 다루지 않는 항목 = LLM 정성 평가]
      - accuracy      : 턴별 답변 품질 점수(0~100)의 평균을 10점으로 환산
      - density_score : 내용 밀도. answerLength 와 50:50 으로 합산
      - filler_score  : 추임새 남용 여부
    """

    # ── 1. 목소리 크기 (RMS) ────────────────────────────────────────
    VOICE_VOLUME_MEAN = 0.1                 # 이상적인 평균 볼륨
    VOICE_VOLUME_TOLERANCE = 0.05           # 이 폭을 벗어날 때마다 계단식으로 1점 감점
    VOLUME_VARIANCE_THRESHOLD = 0.05        # 볼륨 분산이 이 이상이면 '들쭉날쭉'으로 판정
    VOLUME_VARIANCE_PENALTY = 1             # 위 조건 충족 시 추가 감점
    LOW_VOLUME_RATIO_THRESHOLD = 0.3        # 작은 목소리 구간 비율이 이 이상이면 감점
    LOW_VOLUME_RATIO_PENALTY = 1            # 위 조건 충족 시 추가 감점

    # ── 2. 발화 속도 (초당 글자 수, CPS) ────────────────────────────
    VOICE_SPEED_MEAN = 5.0                  # 이상적인 초당 글자 수
    VOICE_SPEED_TOLERANCE = 1.0             # 이 폭을 벗어날 때마다 1점 감점
    PAUSE_ALLOWANCE = 1.0                   # 턴당 허용 퍼즈 횟수. 초과분만큼 추가 감점

    # ── 3. 반응 속도 (초) ───────────────────────────────────────────
    RESPONSE_TIME_INTRO_MEAN = 1.5          # 자기소개 단계의 기대 응답시간
    RESPONSE_TIME_FOLLOWUP_MEAN = 3.5       # 꼬리질문 단계의 기대 응답시간(생각할 시간이 더 필요)
    RESPONSE_TIME_TOLERANCE = 1.5           # 이 폭을 벗어날 때마다 1점 감점

    # ── 4. 답변 길이 (초) ───────────────────────────────────────────
    ANSWER_LENGTH_MIN = 40                  # 이 미만이면 (길이/MIN)*10 으로 비례 감점
    ANSWER_LENGTH_MAX = 80                  # 이 초과분 10초마다 1점 감점


# ===========================================================================
# 3. 웹캠(시각) 채점 상수
# ===========================================================================
class VisionScoringConfig:
    """웹캠 기반 평가 항목(gaze/gesture/posture/expression)의 채점 상수.

    [역할 분담]
      - vision_process/aggregator.py : 프레임 -> 원시 피쳐 1차 가공 (임계값은 그쪽 상수)
      - 이 클래스                     : 원시 피쳐 -> 0~10점 감점

    음성과 동일하게 '턴별 점수를 매기고 전체 평균' 방식을 쓴다.
    """

    # ── 신뢰도 게이트 ───────────────────────────────────────────────
    MIN_FACE_RATIO = 0.5                    # 얼굴 검출 비율이 이 미만인 턴은 채점에서 제외
    MIN_POSE_RATIO = 0.5                    # 자세 검출 비율이 이 미만이면 손짓/자세를 중립 처리
    UNMEASURABLE_SCORE = 5                  # 측정 불가 시 부여할 중립 점수(감점하지 않음)

    # ── 1. 시선 (gaze) ──────────────────────────────────────────────
    GAZE_RATIO_FULL = 0.80                  # 정면 응시 비율이 이 이상이면 만점
    GAZE_RATIO_ZERO = 0.30                  # 이 이하이면 0점 (사이는 선형 보간)
    GAZE_JITTER_TOLERANCE = 5.0             # 두부 각도 표준편차(deg) 허용치
    GAZE_JITTER_STEP = 2.5                  # 초과분이 이만큼 늘 때마다 1점 감점

    # ── 2. 자세 (posture) ───────────────────────────────────────────
    POSTURE_TILT_TOLERANCE = 8.0            # 어깨 기울기 평균(deg) 허용치
    POSTURE_TILT_STEP = 3.0                 # 초과분이 이만큼 늘 때마다 1점 감점
    POSTURE_SWAY_TOLERANCE = 0.030          # 상체 흔들림 표준편차(어깨너비 정규화) 허용치
    POSTURE_SWAY_STEP = 0.020               # 초과분이 이만큼 늘 때마다 1점 감점
    POSTURE_DRIFT_ENABLED = False           # torsoDrift 감점 사용 여부. 턴 길이 편향이 있어 기본 비활성
    POSTURE_DRIFT_TOLERANCE = 0.25          # (활성 시) 기준 위치 이탈 허용치
    POSTURE_DRIFT_STEP = 0.10               # (활성 시) 초과분 단위 감점 폭

    # ── 3. 손짓 (gesture) = 손 사용 빈도 ────────────────────────────
    GESTURE_USAGE_HARD_ZERO = 0.02          # 이 이하 = 손을 아예 안 씀
    GESTURE_ZERO_SCORE = 2                  # 위 경우 부여할 점수
    GESTURE_USAGE_MIN = 0.15                # 이 미만 = 손을 거의 안 씀(경직)
    GESTURE_USAGE_MAX = 0.75                # 이 초과 = 과도한 손짓
    GESTURE_UNDER_STEP = 0.05               # 하한 미달분이 이만큼 늘 때마다 1점 감점
    GESTURE_OVER_STEP = 0.15                # 상한 초과분이 이만큼 늘 때마다 1점 감점
    GESTURE_EXTENT_MIN = 0.15               # 손이 보이지만 이동 범위가 이 미만이면 '굳음'으로 감점
    GESTURE_EXTENT_PENALTY = 1              # 위 조건 충족 시 감점 폭
    GESTURE_FACE_TOUCH_ALLOWANCE = 1        # 턴당 얼굴 만지기 허용 횟수. 초과분만큼 감점

    # ── 4. 표정 (expression) ────────────────────────────────────────
    EXPRESSION_RIGID_THRESHOLD = 0.012      # 표정 변화량이 이 이하이면 '굳은 표정'
    EXPRESSION_RIGID_PENALTY = 2            # 위 조건 충족 시 감점 폭
    EXPRESSION_FROWN_TOLERANCE = 0.15       # 찌푸림 프레임 비율 허용치
    EXPRESSION_FROWN_STEP = 0.15            # 초과분이 이만큼 늘 때마다 1점 감점
    EXPRESSION_SMILE_BONUS_RATIO = 0.15     # 미소 비율이 이 이상이면 +1점
    EXPRESSION_BLINK_ENABLED = False        # 눈깜빡임 감점 사용 여부. 10fps 에서 신뢰 불가라 기본 비활성
    EXPRESSION_BLINK_MIN = 8.0              # (활성 시) 분당 깜빡임 정상 하한 - 미만이면 응시 경직
    EXPRESSION_BLINK_MAX = 32.0             # (활성 시) 정상 상한 - 초과면 긴장
    EXPRESSION_BLINK_STEP = 10.0            # (활성 시) 초과분 단위 감점 폭


# ===========================================================================
# 4. 개입(끼어들기) 상수
# ===========================================================================
class BargeInConfig:
    """개입 판정 게이트(G1~G7)와 후속 처리에 쓰는 상수.

    게이트 순서는 bargein.py 참조. 비용이 싼 것부터 검사한다.
    """

    # ── G2: 개입 대상 단계 ──────────────────────────────────────────
    # 주의: 설계서의 'PERSONALITY' 는 실제 코드에서 'BEHAVIORAL' 이다.
    #   SELF_INTRO 제외 - 페르소나 구성에 답변 전체가 필요하고 '이탈' 기준이 없다.
    #   CLOSING    제외 - 자유 발언이라 주제 이탈 개념이 성립하지 않는다.
    TARGET_STAGES = {"TECH_Q1", "FOLLOWUP_1", "FOLLOWUP_2", "BEHAVIORAL"}

    # ── G4: 세션당 개입 횟수 상한 ───────────────────────────────────
    # 자기소개는 채점 대상이 아니라 TECH_Q1 답변 중 페르소나가 항상 NEUTRAL 이므로 실질 3회.
    MAX_PER_SESSION = 3

    # ── G6: 판정 유예 (실측 후 조정) ────────────────────────────────
    GRACE_SEC = 8.0                         # 발화 시작 후 이 시간 전에는 개입을 판정하지 않는다
    MIN_PARTIAL_CHARS = 25                  # 누적 부분 전사가 이 글자 수 미만이면 판정하지 않는다

    # ── 개입 후 채점 가중치 (Type A 전용) ───────────────────────────
    W_TRUNCATED = 0.5                       # 잘린 답변 점수의 가중치
    W_REANSWER = 0.5                        # 재답변 점수의 가중치

    # ── Type B 대기 상한 ────────────────────────────────────────────
    # STT 가 빈 전사(stt_skip)를 반환하면 텍스트가 영영 오지 않으므로 워치독이 필요하다.
    FINAL_WAIT_TIMEOUT = 7.0                # 잘린 전사 / 발화1 완료를 기다리는 최대 시간(초)

    # ── 개입 시 발신할 비언어 ID ────────────────────────────────────
    # ExpressionID 5(FIRM_STOP)는 블렌드셰이프 프리셋으로 구현되어 있다.
    # GestureID 8(PALM_STOP)은 애니메이션 리소스 미확보로 예약 상태이므로,
    # 실제로는 기존 부정 제스처(ARMS_CROSSED=3)를 재사용한다.
    EXPRESSION_FIRM_STOP = 5                # 개입 시 표정 ID
    GESTURE_BARGEIN = 3                     # 개입 시 제스처 ID