# VRoom 백엔드 서버.

이 서버는 음성 처리 서버(STT/TTS)와 Unity 사이에서 **면접의 두뇌 + 통신 허브** 역할을 합니다.
사용자 답변을 LLM으로 **채점**하고, 점수에 따라 면접관 **페르소나(긍정/중립/부정)를 실시간 가변**시키며,
다음 질문 대사 + 비언어 메타데이터(`Expression_ID`, `Gesture_ID`)를 만들어 Unity로 보내고,
그 대사를 TTS로 합성해 **음성**까지 흐르게 합니다.

> 이 문서는 음성 처리, LLM 팀원이 레포지토리를 받아 바로 통합 실행할 수 있도록 작성되었습니다.
> 

---

## 1. 전체 구조 (최종 · 모두 WebSocket 연동)

면접의 두뇌는 **이 백엔드 하나**입니다. TTS 워커는 받은 텍스트를 그대로 읽는 **TTS 전용**으로 동작합니다.

```
            ┌──────────────────────── Unity (VR 클라이언트) ────────────────────────┐
            │  마이크 음성                                  행동패킷(JSON)+자막+음성  │
            ▼ (ws /ws/interview)                              ▲ (ws /ws/control)     │
      ┌───────────┐   ws {"text": 사용자답변}        ┌────────────────┐              │
      │ STT(Node A)│ ───────────────────────────────▶│  VRoom 백엔드   │──────────────┘
      │ Whisper+VAD│ ◀─── 음성청크 + {"type":"end"} ──│ (채점/페르소나/ │
      └───────────┘                                   │  질문/메모리)   │
            ▲ 음성을 Unity로 패스스루                  └───────┬────────┘
            └──────────────────────────────────────           │ ws {"text": 면접관대사}
                                                               ▼ (ws /ws/tts)
                                                        ┌────────────────┐
                                                        │  TTS(Node B)    │
                                                        │ Fish Speech 1.5 │ (TTS_ONLY=true)
                                                        └────────────────┘
```

데이터 흐름(한 턴):

1. Unity 마이크 → STT(Node A)가 전사
2. STT가 전사 텍스트를 백엔드 `/ws/tts`로 전송
3. 백엔드: 채점 → 페르소나 결정 → 다음 질문 대사 + 제스처/표정 ID 생성
4. 백엔드 → Unity `/ws/control`: 행동패킷(JSON) + 자막
5. 백엔드 → TTS `/ws/tts`: 면접관 대사 합성 → 음성을 STT로 릴레이 → STT가 Unity로 패스스루
6. 면접관 음성 재생 + 자막/표정/제스처 동기화

면접 시나리오: 자기소개 → 기술질문 → 꼬리질문1 → 꼬리질문2 → 인성 → 마무리 → 종합 피드백

> **첫 질문(자기소개 요청)** 은 사용자 발화 없이 Unity 접속(`init`) 직후 백엔드가 자동 생성하며,
이 경로는 TTS의 **HTTP `/process`** 를 사용합니다(아래 4.4 참고). 그래서 TTS는 `/process`와 `/ws/tts`**양쪽 모두** `TTS_ONLY` 분기가 있어야 합니다.
> 

---

## 2. 사용된 기술

- **FastAPI (async)** — WebSocket 2개(`/ws/control`, `/ws/tts`) + HTTP(`/process`, `/health`)
- **OpenAI / Groq SDK** — LLM 채점 + 대사 생성. `LLM_PROVIDER`로 전환(최종 OpenAI, 테스트 Groq)
- **WebSocket** — STT↔백엔드↔TTS 전 구간. 음성 패킷 경계 유지(자막 기능에 필요)
- **Pydantic** — 행동 패킷 / 피드백 스키마 검증

---

## 3. 스크립트 설명 (`app/`)

- **main.py** — FastAPI 앱. WebSocket `/ws/control`(Unity), `/ws/tts`(STT 입구), HTTP `/process`, `/health`
- **config.py** — `.env` 로딩(LLM 키, TTS 주소 2종, 동작 플래그)
- **domain.py** — 면접 단계 · 페르소나 · `Expression_ID`/`Gesture_ID` 코드표 · 스키마
- **llm.py** — LLM 구조화 출력(대사 + 0~100 채점 + 제스처/표정) 및 피드백 생성
- **session.py** — 면접 상태머신 + 점수→페르소나 가변 + 대화 메모리 + 피드백 집계
- **tts_client.py** — 대사를 구(phrase) 단위로 분할(`split_phrases`) + HTTP TTS 호출(`synthesize_stream`)

---

## 4. 통신 명세

### 4.1 엔드포인트

| 경로 | 종류 | 용도 |
| --- | --- | --- |
| `/ws/control` | WebSocket | Unity ↔ 백엔드 (행동패킷 + 자막 + 음성) |
| `/ws/tts` | WebSocket | STT 워커 → 백엔드 (사용자 답변 입구). TTS와 동일 인터페이스 |
| `/process` | HTTP POST | (옵션) STT가 HTTP로 텍스트를 줄 때 |
| `/health` | HTTP GET | 상태 확인 |

### 4.2 Unity ↔ 백엔드 (`/ws/control`)

Unity → 백엔드:

- `{"type":"init","session_id":"default","company":"네이버","job_title":"백엔드 개발자","resume":"..."}`
- `{"type":"request_feedback","session_id":"default"}`

백엔드 → Unity:

- 행동패킷(JSON, 자막 포함):
    
    ```json
    {"type":"interviewer_turn","session_id":"default","stage":"FOLLOWUP_1", "persona":"NEGATIVE","dialogue":"면접관 대사(자막)","expression_id":2,"gesture_id":3, "score":42,"is_final":false}
    ```
    
- `{"type":"thinking",...}` — LLM 연산 중 '검토 중' 더미 모션
- 바이너리 프레임 = 면접관 음성(첫 질문 등 백엔드가 직접 보내는 경우)
- `{"type":"audio_end"}` — 음성 종료
- `{"type":"feedback_report",...}` — 면접 종료 후 종합 피드백

### 4.3 STT → 백엔드 (`/ws/tts`)

STT 워커는 TTS에 보내던 것과 **똑같은 형식**으로 백엔드에 보냅니다(인터페이스 호환).

- STT → 백엔드: `{"text":"사용자 답변 전사 결과"}`
- 백엔드 → STT: 음성 청크(바이너리) … 그리고 마지막에 `{"type":"end"}`

백엔드는 동시에 행동패킷+자막을 `/ws/control`로 Unity에 push 합니다.

### 4.4 첫 질문 경로 주의

첫 질문은 `/ws/control`의 `init` 처리 중 `speak()` → `tts_client.synthesize_stream()` → TTS **HTTP `/process`** 로 합성됩니다.
따라서 TTS 워커는 `/process`와 `/ws/tts` **양쪽 모두** TTS 전용 분기가 있어야 첫 질문도 면접관 대사로 나옵니다(5절).

### 4.5 Expression_ID / Gesture_ID 코드표 (상세 구현 시 수정 필요)

| ID | Expression_ID | Gesture_ID |
| --- | --- | --- |
| 0 | 무표정 | Idle |
| 1 | 온화한 미소(긍정) | 깊게 끄덕임 |
| 2 | 미간 찌푸림(부정) | 고개 갸우뚱 |
| 3 | 경청/관심 | 팔짱(강한 부정) |
| 4 | 생각 중 | 펜 만지작 |
|  |  | 5=시작 안내, 6=이력서 검토, 7=경청 끄덕임 |

페르소나 규칙: 점수 70↑ 긍정 / 40~69 중립 / 40 미만 부정. 저점 2회 연속 시 강한 부정으로 고착.

---

## 5. 음성 처리 서버(Node A/B) 연동

면접의 두뇌는 백엔드 하나로 모읍니다. 핵심은 두 가지입니다.

### 5.1 STT 워커(Node A) — 전송 대상을 백엔드로

`stt-worker-docker/.env`:

```
# STT가 전사 결과를 보내는 곳 = 백엔드(8080). (TTS 8001 아님)
TTS_WORKER_WS_URL=ws://host.docker.internal:8080/ws/tts
```

STT 워커 코드는 그대로 두면 됩니다. `{"text": final_text}`를 보내고 음성 청크 + `{"type":"end"}`를 받는
기존 로직이 백엔드와 그대로 호환됩니다

### 5.2 TTS 워커(Node B) — TTS 전용

자체 LLM을 끄고 **받은 텍스트를 그대로 합성**해야 합니다. `/ws/tts`와 `/process` **양쪽 모두** 분기 필요:

```python
# 상단: load_dotenv() 다음에 읽을 것 (순서 중요)
load_dotenv()
TTS_ONLY = os.getenv("TTS_ONLY", "false").lower() == "true"

# /ws/tts 안
gen = tts_only_generator(text) if TTS_ONLY else response_generator(text)

# /process 안 (첫 질문 경로 — 빠뜨리기 쉬움!)
@app.post("/process")
async def process_text_to_audio(request: TTSRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is empty")
    tts_only = os.getenv("TTS_ONLY", "false").lower() == "true"
    gen = tts_only_generator(request.text) if tts_only else response_generator(request.text)
    return StreamingResponse(gen, media_type="application/octet-stream")
```

`tts-worker-docker/.env`:

```
TTS_ONLY=true
```

> **검증 포인트:** TTS 콘솔의 `[TTS] Synthesizing:` 뒤에 *면접관 대사*가 찍히면 정상,
"저는 OpenAI가 개발한…" 같은 ChatGPT 자기소개가 찍히면 `TTS_ONLY`가 false(또는 `/process`에 분기 누락)입니다.
> 

---

## 6. LLM 서버 연동

현재 LLM 호출은 `app/llm.py` 한 곳에 캡슐화돼 있습니다. 교체할 함수는 둘:

- `generate_turn(...)` — 단계·페르소나·대화기록 → **대사 + 0~100 점수 + 제스처/표정 ID**(JSON)
- `generate_feedback(...)` — 전체 기록 → 강점/개선점/총평

이 두 함수의 **입출력 계약(`LLMTurn` 스키마, 피드백 dict)만 유지**하면 내부를 자유 교체 가능합니다.
시나리오 진행·페르소나 임계값은 `session.py`가 통제하므로, LLM은 "현재 턴 생성"에만 집중하면 됩니다.

---

## 7. 실행 방법

### 7.1 준비물

1. Python 3.10+
2. LLM API 키 (OpenAI `sk-...` 또는 Groq `gsk_...`)
3. 음성까지 테스트하려면 STT/TTS 워커(GPU + 모델 파일) 실행

### 7.2 `.env` (백엔드 — 호스트에서 실행하므로 `localhost`)

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
# OPENAI_API_KEY=sk-...   # 최종 통합 시
# OPENAI_MODEL=gpt-4o-mini

# 백엔드 → 진짜 TTS(8001). 백엔드가 호스트면 localhost, 도커 안이면 host.docker.internal
TTS_WORKER_URL=http://localhost:8001/process
TTS_WS_URL=ws://localhost:8001/ws/tts

HOST=0.0.0.0
PORT=8080
PROXY_AUDIO_TO_STT=false
SKIP_TTS=false
```

### 7.3 설치 & 실행

```bash
python -m venv .venv
Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

확인: `http://127.0.0.1:8080/health` → `{"status":"ok","provider":"groq",...}`

### 7.4 전체 실행 순서 (의존성 순)

1. **TTS 워커(Node B)** 기동 → `:8001` (`TTS_ONLY=true`)
2. **백엔드** 기동 → `:8080`
3. **STT 워커(Node A)** 기동 → `:8000` (`TTS_WORKER_WS_URL`이 8080을 가리킴)
4. **Unity Play**

### 7.5 단독 테스트 (음성 서버 없이)

`.env`에 `SKIP_TTS=true`로 두면 TTS 없이 채점·페르소나·질문 생성 로직만 검증할 수 있습니다.
Unity의 `InterviewDebugInput`(타이핑 패널)로 답변을 넣으면 됩니다.

---

## 8. 파일 구조

```
VRoom_Backend/
├── app/
│   ├── main.py        # FastAPI 앱 (/ws/control, /ws/tts, /process, /health)
│   ├── config.py      # .env 설정 (LLM 키, TTS 주소 2종, 플래그)
│   ├── domain.py      # 단계/페르소나/ID 코드표 + 스키마
│   ├── llm.py         # LLM 턴/피드백 생성 (어요솝드 통합 지점)
│   ├── session.py     # 면접 상태머신 + 페르소나 가변 + 메모리
│   └── tts_client.py  # 대사 구 분할 + TTS HTTP 호출
├── requirements.txt
└── .env
```
