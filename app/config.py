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

    # provider에 맞는 (base_url, api_key, model) 묶음을 돌려준다.
    def resolve_llm(self) -> tuple[str | None, str, str]:
        if self.llm_provider == "groq":
            return ("https://api.groq.com/openai/v1", self.groq_api_key, self.groq_model)
        return (None, self.openai_api_key, self.openai_model)  # openai 기본 base_url


settings = Settings()
