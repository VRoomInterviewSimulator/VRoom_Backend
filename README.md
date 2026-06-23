# VRoom 백엔드 서버 (AI 면접관 두뇌 · 4.2.4 백엔드 및 통신)

졸업과제 **VRoom**(VR 모의면접 시뮬레이터)의 백엔드 서버 레포지토리입니다.

이 서버는 음성 처리 서버(STT/TTS)와 Unity 사이에서 **면접의 두뇌 + 통신 허브** 역할을 합니다.
사용자 답변을 LLM(OpenAI)으로 **채점**하고, 점수에 따라 면접관 **페르소나(긍정/중립/부정)를 실시간 가변**시키며,
다음 질문 대사 + 비언어 메타데이터(`Expression_ID`, `Gesture_ID`)를 만들어 Unity로 보내고,
그 대사를 TTS로 합성해 **음성**까지 흐르게 합니다.

> 이 문서 하나로 **프론트/백엔드 담당(나)** 과 **음성 처리 담당(팀원)** 이 각자 다른 PC·다른 네트워크에서
> 프로그램을 띄워 **전체 면접 루프를 테스트**할 수 있도록 모든 절차를 담았습니다.

---

## 0. 역할 분담과 분산 구조 한눈에 보기

| 담당 | PC | 실행 프로그램 | 포트 |
|:--|:--|:--|:--|
| **프론트 담당** | PC-A (Tailscale 설치된 데스크탑) | Unity(프론트) + FastAPI 백엔드 | 백엔드 8080 |
| **음성 처리 담당** | PC-B | STT 워커(Node A) + TTS 워커(Node B), Docker | STT 8000, TTS 8001 |
| **LLM 담당** | — | **별도 서버 없음.** 백엔드가 OpenAI API를 직접 호출 | — |

> LLM 담당이 만든 `example.py`/`test_api.py`는 로직 레퍼런스로, 해당 로직을 참고하여 현재 레포지토리의 `app/llm.py`에 구현 완료했습니다.
> LLM 담당 팀원은 앞으로 `app/llm.py`에서 자신의 코드를 구현해주시면 됩니다.
> 따라서 **네트워크로 연결할 대상은 PC-A ↔ PC-B 둘뿐**입니다.

```
   PC-A (나)                                  PC-B (팀원)
┌──────────────────────────┐            ┌────────────────────────┐
│  Unity (프론트)           │  ws 8000   │  STT 워커(Node A)       │
│   └ 마이크 ───────────────┼───────────▶│   Whisper + Silero VAD  │
│                           │◀───음성────┤                        │
│  FastAPI 백엔드 :8080     │            │        ▲ ws {"text"}    │
│   ├ /ws/control (Unity)   │  ws 8080   │        │ 음성+{"end"}   │
│   ├ /ws/tts (STT 입구) ◀──┼────────────┘        ▼               │
│   ├ 채점·페르소나·질문    │  ws 8001   │  TTS 워커(Node B)       │
│   └ 면접관 대사 ──────────┼───────────▶│   Fish Speech 1.5       │
│         │ OpenAI 직접 호출│            │   (TTS_ONLY=true)      │
└─────────┼────────────────┘            └────────────────────────┘
          ▼ 인터넷
     OpenAI API (클라우드)

두 PC는 Tailscale 가상 네트워크로 100.x.x.x IP를 통해 연결된다.
```

데이터 흐름(한 턴):
1. Unity 마이크 → STT(PC-B)가 전사
2. STT가 전사 텍스트를 백엔드(PC-A) `/ws/tts`로 전송
3. 백엔드: (자기소개면 정보 추출) → 채점 → 페르소나 결정 → 다음 질문 + 제스처/표정 ID 생성
4. 백엔드 → Unity `/ws/control`: 행동패킷(JSON) + 자막
5. 백엔드 → TTS(PC-B) `/ws/tts`: 면접관 대사 합성 → 음성을 STT로 릴레이 → STT가 Unity로 패스스루
6. 면접관 음성 재생 + 자막/표정/제스처 동기화

면접 시나리오(6단계): 자기소개 → 기술질문 → 꼬리질문1 → 꼬리질문2 → 인성 → 마무리 → 종합 피드백

---

## 1. 사전 준비 (최초 1회)

### 1.1 공통 — Tailscale로 두 PC 연결 (다른 네트워크여도 OK)
서로 다른 네트워크라 포트포워딩 대신 **Tailscale**(무료 VPN)로 두 PC를 같은 가상 LAN에 묶습니다.
공유기 설정·공인 IP 불필요.

**프론트 담당(PC-A, 이미 Tailscale 설치됨):**
1. 데스크탑에서 Tailscale 로그인 상태 확인:
   ```powershell
   tailscale ip -4
   ```
   `100.x.x.x` 가 나오면 연결됨. 이 값이 **백엔드 IP**(예: `100.10.10.10`). 메모.
2. 이 데스크탑만 음성 처리 담당 팀원에게 공유: [login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines)
   → 데스크탑 행 `...` → **Share** → 생성된 링크를 팀원에게 전달.

**음성 처리 담당(PC-B):**
1. Tailscale 설치 → 로그인 → `tailscale ip -4`로 본인 IP 확인(예: `100.20.20.20`). 프론트 담당 팀원에게 전달.
2. 본인 PC를 나에게 Share(관리 콘솔 → 본인 기기 → Share → 링크 전달).
3. 내가 보낸 Share 링크 수락.
> Share는 **양방향** 모두 필요(프론트 담당→음성 처리 담당, 음성 처리 담당→프론트 담당).

연결 확인: 한쪽에서 `ping 상대_100.x.x.x` 또는 `tailscale ping 상대IP` 응답이 오면 성공.

> 이후 문서에서 `BACKEND_IP`=PC-A의 100.x.x.x, `VOICE_IP`=PC-B의 100.x.x.x 로 표기.

### 1.2 프론트 담당(PC-A) — 백엔드 환경
1. Python 3.10+ 설치
2. 레포 클론 후 가상환경:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate   | macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. OpenAI API 키 준비(`sk-...`). 결제/크레딧 잔액 확인(gpt-4o-mini는 저렴하나 면접당 7~8회 호출).

### 1.3 음성 처리 담당(PC-B) — 음성 서버 환경
1. NVIDIA GPU + Docker Desktop(WSL2) + VS Code(Dev Containers)
2. STT/TTS 워커 폴더 + 모델 파일 배치(구글드라이브)

---

## 2. 설정 파일 (IP를 서로의 Tailscale 주소로)

### 2.1 프론트 담당(PC-A) — 백엔드 `.env`
```ini
# ----- LLM: OpenAI -----
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-여기에_본인_키
OPENAI_MODEL=gpt-4o-mini

# ----- TTS 워커(Node B) = 음성 처리 담당 PC-B -----
TTS_WORKER_URL=http://VOICE_IP:8001/process
TTS_WS_URL=ws://VOICE_IP:8001/ws/tts

# ----- 서버 -----
HOST=0.0.0.0
PORT=8080
PROXY_AUDIO_TO_STT=false
SKIP_TTS=false
```
> `VOICE_IP`를 음성 처리 담당 팀원의 실제 Tailscale IP(예: `100.20.20.20`)로 교체.

### 2.2 프론트 담당(PC-A) — Unity Inspector
- `BackendControlClient.backendUrl` = `ws://127.0.0.1:8080/ws/control`  (백엔드는 같은 PC)
- `STTManager` 의 STT URL = `ws://VOICE_IP:8000/ws/interview`  (STT는 음성 처리 담당 팀원 PC)
- `InterviewDebugInput.backendHttp` = `http://127.0.0.1:8080/process`

### 2.3 음성 처리 담당 팀원(PC-B) — STT 워커 `.env` (`stt-worker-docker/.env`)
```ini
TTS_WORKER_WS_URL=ws://BACKEND_IP:8080/ws/tts
```
> `BACKEND_IP`를 프론트 담당의 Tailscale IP(예: `100.10.10.10`)로 교체.

### 2.4 음성 처리 담당 팀원(PC-B) — TTS 워커 `.env` (`tts-worker-docker/.env`)
```ini
TTS_ONLY=true
```
TTS는 받은 텍스트를 그대로 합성(TTS 전용). `/ws/tts`와 `/process` **양쪽 모두** 아래 분기 필요:
```python
load_dotenv()
TTS_ONLY = os.getenv("TTS_ONLY", "false").lower() == "true"   # load_dotenv 다음에 읽기!

# /ws/tts 안
gen = tts_only_generator(text) if TTS_ONLY else response_generator(text)

# /process 안 (첫 질문 경로 — 빠뜨리기 쉬움)
@app.post("/process")
async def process_text_to_audio(request: TTSRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is empty")
    tts_only = os.getenv("TTS_ONLY", "false").lower() == "true"
    gen = tts_only_generator(request.text) if tts_only else response_generator(request.text)
    return StreamingResponse(gen, media_type="application/octet-stream")
```

---

## 3. 방화벽 & 도커 포트 노출

### 3.1 프론트 담당(PC-A) — 방화벽 8080 허용
관리자 PowerShell(우클릭 → 관리자 권한으로 실행):
```powershell
New-NetFirewallRule -DisplayName "VRoom Backend 8080" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
```

### 3.2 음성 처리 담당(PC-B) — 방화벽 8000·8001 허용
관리자 PowerShell:
```powershell
New-NetFirewallRule -DisplayName "VRoom STT 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "VRoom TTS 8001" -Direction Inbound -LocalPort 8001 -Protocol TCP -Action Allow
```

### 3.3 음성 처리 담당(PC-B) — 도커 포트를 호스트로 노출 (분산에서 가장 흔한 함정)
STT/TTS는 컨테이너 안에서 도므로, 컨테이너 포트가 **호스트 PC로 바인딩**돼야 Tailscale 너머에서 접근됨.
- `.devcontainer/devcontainer.json` 에 `"appPort": [8000]` / `[8001]`  (단순 `forwardPorts`로는 외부 접근 안 될 수 있음)
- 또는 `docker run -p 8000:8000 -p 8001:8001 ...`
- 워커는 `uvicorn.run(app, host="0.0.0.0", ...)` 로 기동(이미 그러함)

---

## 4. 실행 순서 (의존성 순서대로)

> 두 PC 모두 Tailscale이 켜져 있어야 함.

**음성 처리 담당(PC-B) 먼저:**
1. TTS 워커 컨테이너 기동 → 콘솔에 `Fish Speech 1.5 models loaded and ready.` + `Uvicorn running on http://0.0.0.0:8001`
2. STT 워커 컨테이너 기동 → `Uvicorn running on http://0.0.0.0:8000`

**프론트 담당(PC-A):**
3. 백엔드 기동:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
   ```
   → `http://127.0.0.1:8080/health` 가 `{"status":"ok","provider":"openai",...}`
4. Unity 실행 후 **Play**

---

## 5. 단계별 연결 검증 (막히면 여기서 원인 파악)

### 5.1 Tailscale 도달 확인
- 프론트 담당 → 음성 처리 담당: 브라우저로 `http://VOICE_IP:8001/` 접근 시 응답(404여도 연결은 된 것)
- 음성 처리 담당 → 프론트 담당: 브라우저로 `http://BACKEND_IP:8080/health` 가 `{"status":"ok"}`
- 안 되면: Tailscale Share 양방향 확인 → 방화벽 → 도커 포트 노출 순으로 점검

### 5.2 면접 루프 확인 (백엔드 콘솔 기준)
Unity Play 후:
- `WebSocket /ws/control [accepted]` — Unity 연결됨
- `[/ws/tts] STT 워커 연결됨` — 팀원 STT가 백엔드에 붙음 ✅
- 첫 질문(자기소개 요청) 음성이 Unity에서 재생 + Unity 콘솔 자막
- 마이크로 답하면:
  - STT 콘솔: `Success: (전사된 답변)`
  - 백엔드 콘솔: `[STT→백엔드 수신] ...` → `[백엔드→TTS 대사] ...`
  - TTS 콘솔: `[TTS] Synthesizing: (면접관 대사)`
  - Unity: 다음 질문 음성 + 자막/표정/제스처

### 5.3 가변 페르소나 확인 (연구 핵심)
답변을 여러 번 넣으며 백엔드 콘솔의 `persona`/`score` 관찰:
- 좋은 답변 → score 높음 → `persona=POSITIVE`
- 부실한 답변 2~3회 → score 낮음 → `persona=NEGATIVE`(미간 찌푸림·팔짱) 고착

---

## 6. 통신 명세 (요약)

| 경로 | 종류 | 용도 |
|:--|:--|:--|
| `/ws/control` | WebSocket | Unity ↔ 백엔드 (행동패킷 + 자막 + 음성) |
| `/ws/tts` | WebSocket | STT 워커 → 백엔드 (사용자 답변 입구). TTS와 동일 인터페이스 |
| `/process` | HTTP POST | (옵션) STT가 HTTP로 텍스트를 줄 때 |
| `/health` | HTTP GET | 상태 확인 |

행동패킷(백엔드 → Unity, 자막 포함):
```json
{"type":"interviewer_turn","session_id":"default","stage":"FOLLOWUP_1",
 "persona":"NEGATIVE","dialogue":"면접관 대사(자막)","expression_id":2,"gesture_id":3,
 "score":42,"is_final":false}
```

Expression_ID / Gesture_ID 코드표(아직 미구현):
| ID | Expression_ID | Gesture_ID |
|:-:|:--|:--|
| 0 | 무표정 | Idle |
| 1 | 온화한 미소(긍정) | 깊게 끄덕임 |
| 2 | 미간 찌푸림(부정) | 고개 갸우뚱 |
| 3 | 경청/관심 | 팔짱(강한 부정) |
| 4 | 생각 중 | 펜 만지작 |
|   |  | 5=시작 안내, 6=이력서 검토, 7=경청 끄덕임 |

페르소나: 점수 70↑ 긍정 / 40~69 중립 / 40 미만 부정. 저점 2회 연속 시 강한 부정 고착.

---

## 7. 명세 정렬 — 가변 페르소나 동작 원리

- **자기소개 기반 동적 페르소나(시나리오 1단계)**: 자기소개 답변에서 Function Calling으로
  `company_name/job_role/experience_level/skills/strengths` 추출(`llm.extract_info`) → 이후 모든 질문에 주입.
- **정확도 연동 가변(요구사항 핵심)**: FOLLOWUP_1=분기점1, FOLLOWUP_2=STAR 재채점·변별 입증.
  점수→페르소나 결정은 코드(session.py)가 통제, LLM은 대사/점수/제스처만 생성.
- **질문 규칙**: 직전 답변 인용 + STAR 누락 콕 집기 + 경력 수준별 난이도 조정(`_system_prompt`).

---

## 8. LLM 연동 — `app/llm.py`

| 함수 | 입력 | 출력 |
|:--|:--|:--|
| `extract_info(self_intro, ...)` | 자기소개 텍스트 | `ExtractedInfo` (example.py의 extract_interview_info와 동일 스키마) |
| `generate_turn(stage, persona, info, resume, history, user_answer)` | 단계·페르소나·추출정보·기록 | `LLMTurn`(대사+0~100점수+제스처/표정) |
| `generate_feedback(...)` | 전체 기록 | dict(강점/개선점/총평) |

이 입출력 계약만 유지하면 내부 구현 자유 교체 가능.

---

## 9. 트러블슈팅

| 증상 | 원인 / 해결 |
|:--|:--|
| 음성 처리 담당 → 프론트 담당 `/health` 접속 안 됨 | Tailscale Share 양방향 확인 → 방화벽(8080) → 둘 다 Tailscale ON |
| 프론트 담당 → 음성 처리 담당 8001/8000 접속 안 됨 | 팀원 방화벽 + **도커 포트 호스트 노출(appPort)** 확인 |
| `[/ws/tts] STT 워커 연결됨` 안 뜸 | STT `.env` `TTS_WORKER_WS_URL`이 `ws://BACKEND_IP:8080/ws/tts` 인지 |
| 면접관이 ChatGPT처럼 말함 | TTS `TTS_ONLY=true` + `/process`·`/ws/tts` 양쪽 분기 + `load_dotenv` 다음에 읽기 |
| 첫 질문만 ChatGPT | TTS의 `/process` 분기 누락(첫 질문은 HTTP 경로) |
| Unity 자막과 음성 다름 | STT가 백엔드를 안 거침 → `TTS_WORKER_WS_URL` 점검 |
| `[/ws/tts] TTS 릴레이 실패` | 백엔드 `TTS_WS_URL`이 `ws://VOICE_IP:8001/ws/tts` 인지 |
| 페르소나가 회사/직무 모름 | 자기소개 추출 실패 → init의 company/job_title 폴백 확인 |
| `no active session` | Unity가 먼저 `/ws/control`로 `init` 전송해야 함 |
| 첫 질문 음성 치지직 | Unity `Speaker.cs`에서 32-bit float PCM 바이트 경계 보존(잔여 바이트 버퍼링) |
| 5070 Ti 등 `no kernel image` | 워커 컨테이너 PyTorch가 sm_120 미지원 → CUDA 12.8+/cu128 빌드 |
| OpenAI 401/429 | 키 오류/크레딧 소진 → `OPENAI_API_KEY`·잔액 확인 |

---

## 10. 파일 구조
```
VRoom_Backend/
├── app/
│   ├── main.py        # FastAPI 앱 (/ws/control, /ws/tts, /process, /health)
│   ├── config.py      # .env 설정 (OpenAI/Groq, TTS 주소 2종, 플래그)
│   ├── domain.py      # 단계/페르소나/ID 코드표 + 스키마 + ExtractedInfo
│   ├── llm.py         # extract_info + generate_turn + generate_feedback
│   ├── session.py     # 상태머신 + 자기소개 정보추출 + 페르소나 가변 + 메모리
│   └── tts_client.py  # 대사 구 분할 + TTS HTTP 호출
├── requirements.txt
├── .env.example
└── README.md
```

---

## 11. 보안 체크 (제출/공유 전)
- `.gitignore`에 `.env` 포함 — OpenAI 키가 깃허브에 올라가지 않게.
- 노출된 적 있는 키는 폐기 후 재발급.
- Tailscale Share는 데모 후 해제(관리 콘솔에서 Unshare) 가능.
