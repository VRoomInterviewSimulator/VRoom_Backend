"""
면접 세션 상태머신. 사용자별로 하나씩 만들어 들고 있으며,
- 현재 단계(Stage) 진행
- 자기소개 답변에서 동적 페르소나 정보 추출(extract_info) 후 보관
- 점수 -> 페르소나(긍정/중립/부정) 가변 전환 + 연속 저점 시 압박 고착
- 대화 기록(메모리) 누적
- 멀티모달 피쳐(발화시간/침묵 등) 집계
- 종료 시 피드백 산출
을 담당한다.

명세 반영:
  - 시나리오: 자기소개 답변이 들어오면 그 텍스트에서 회사/직무/경력/기술/강점을
    추출(llm.extract_info)해 동적 페르소나를 활성화하고, 이후 모든 질문 생성에 주입한다.
  - 페르소나 변별 핵심 분기: 꼬리질문1, 꼬리질문2.

메모리는 면접이 6턴 내외로 짧으므로 전체 기록을 그대로 보관한다.
(세션이 길어지면 여기서 요약 압축 = LangChain Summary Memory 역할을 넣으면 된다.)
"""
from __future__ import annotations

from . import llm
from .config import settings     
from .domain import (
    AnswerRequest,
    BehaviorPacket,
    ExpressionID,
    ExtractedInfo,
    FeedbackReport,
    GestureID,
    LLMTurn,
    Persona,
    Stage,
    STAGE_ORDER,
    StageScore,
    persona_from_score,
    persona_value_from_score,
    InterviewScore
)


class InterviewSession:
    def __init__(self, session_id: str, company: str = "", job_title: str = "", resume: str = ""):
        self.session_id = session_id
        self.company = company
        self.job_title = job_title
        self.resume = resume

        # 동적 페르소나 정보: Unity init 값으로 우선 채우고, 자기소개 답변에서 추출해 갱신.
        self.info = ExtractedInfo(company_name=company, job_role=job_title)
        self._info_extracted = False  # 자기소개 1회만 추출

        self.stage: Stage = Stage.INIT
        self.persona: Persona = Persona.NEUTRAL
        self.consecutive_low = 0  # 연속 저점 카운트 (압박 고착용)

        self.turns: list[dict] = []          # {"role","stage","text"}
        self.stage_scores: list[tuple[str, int]] = []
        self.speaking_times: list[float] = []
        self.total_pauses = 0

    # ----- 메모리 -----
    def _history_text(self) -> str:
        lines = []
        for t in self.turns:
            who = "면접관" if t["role"] == "interviewer" else "지원자"
            lines.append(f"{who}({t['stage']}): {t['text']}")
        return "\n".join(lines)

    def _record(self, role: str, text: str):
        self.turns.append({"role": role, "stage": self.stage.value, "text": text})

    def _advance_stage(self):
        """다음 단계로 한 칸 이동."""
        if self.stage == Stage.INIT:
            self.stage = STAGE_ORDER[0]
            return
        idx = STAGE_ORDER.index(self.stage)
        self.stage = STAGE_ORDER[min(idx + 1, len(STAGE_ORDER) - 1)]


    # ----- 핵심: 면접관의 다음 발화 생성 -----
    FIRST_QUESTION_TEMPLATE = (
        "안녕하세요. {company} {job} 직무 면접에 참여해 주셔서 감사합니다. "
        "오늘 면접을 진행하게 된 면접관입니다. "
        "먼저 {company} {job} 직무에 지원하신 동기와, "
        "본인이 이 직무에 적합하다고 생각하는 이유를 포함해서 자기소개를 부탁드립니다."
    )

    def template_first_question(self) -> BehaviorPacket:
        if self.stage == Stage.INIT:
            self._advance_stage()

        company = (self.company or "").strip().splitlines()[0] if self.company else ""
        job = (self.job_title or "").strip().splitlines()[0] if self.job_title else ""
        dialogue = self.FIRST_QUESTION_TEMPLATE.format(
            company=company or "저희 회사",
            job=job or "지원하신",
        )

        self._record("interviewer", dialogue)
        return BehaviorPacket(
            type="interviewer_turn",
            session_id=self.session_id,
            stage=self.stage.value,
            persona=self.persona.value,
            persona_value=0.0,
            dialogue=dialogue,
            expression_id=ExpressionID.WARM_SMILE.value,
            gesture_id=GestureID.WELCOME.value,
            score=-1,
            is_final=False,
        )


    async def first_question(self) -> BehaviorPacket:
        """면접 시작: 면접관이 먼저 자기소개를 요청하는 첫 발화."""
        if settings.template_first_question:
            return self.template_first_question()

        self._advance_stage()
        turn = await llm.generate_turn(
            stage=self.stage, persona=self.persona,
            info=self.info, resume=self.resume,
            history="", user_answer="",
        )
        self._record("interviewer", turn.dialogue)
        return self._to_packet(turn, is_final=False)

    async def on_user_answer(self, text: str, features: dict) -> BehaviorPacket:
        """사용자 답변(STT 결과)을 받아 채점하고 다음 단계 발화를 생성."""
        self._record("user", text)
        self._collect_features(features)

        # [1단계] 자기소개 답변이면 동적 페르소나 정보 추출 (1회).
        #  - 현재 단계가 SELF_INTRO == 방금 받은 답변이 자기소개라는 뜻.
        if self.stage == Stage.SELF_INTRO and not self._info_extracted:
            extracted = await llm.extract_info(
                text, fallback_company=self.company, fallback_job=self.job_title
            )
            # 추출 결과를 보관(빈 값은 기존 값 유지)
            self.info = extracted
            if not self.info.company_name:
                self.info.company_name = self.company
            if not self.info.job_role:
                self.info.job_role = self.job_title
            self._info_extracted = True

        was_closing = self.stage == Stage.CLOSING
        self._advance_stage()
        if was_closing:
            # 마무리 답변까지 끝남 -> DONE. 짧은 종료 멘트만.
            self.stage = Stage.DONE
            closing = BehaviorPacket(
                session_id=self.session_id, stage=Stage.DONE.value,
                persona=self.persona.value,
                dialogue="면접에 응해 주셔서 감사합니다. 잠시 후 결과를 안내해 드리겠습니다.",
                expression_id=ExpressionID.WARM_SMILE.value,
                gesture_id=GestureID.DEEP_NOD.value, score=-1, is_final=True,
            )
            self._record("interviewer", closing.dialogue)
            return closing

        # 직전 답변을 채점 + 다음 질문 생성 (동적 페르소나 정보 주입)
        turn = await llm.generate_turn(
            stage=self.stage, persona=self.persona,
            info=self.info, resume=self.resume,
            history=self._history_text(), user_answer=text,
        )

        # 점수 -> 페르소나 가변 전환 (코드가 최종 결정)
        if turn.score >= 0:
            prev_stage_name = self.turns[-2]["stage"] if len(self.turns) >= 2 else self.stage.value
            self.stage_scores.append((prev_stage_name, turn.score))
            self.consecutive_low = self.consecutive_low + 1 if turn.score < 40 else 0
            self.persona = persona_from_score(turn.score, self.consecutive_low)
            turn = llm._clamp_to_set(turn, self.persona)
            # 압박 고착: 연속 2회 이상 저점이면 강한 부정 제스처 강제
            if self.persona == Persona.NEGATIVE and self.consecutive_low >= 2:
                turn.expression_id = ExpressionID.SLIGHT_FROWN.value
                turn.gesture_id = GestureID.ARMS_CROSSED.value

        self._record("interviewer", turn.dialogue)
        return self._to_packet(turn, is_final=False)

    def _to_packet(self, turn: LLMTurn, is_final: bool) -> BehaviorPacket:
        return BehaviorPacket(
            session_id=self.session_id,
            stage=self.stage.value,
            persona=self.persona.value,
            persona_value=persona_value_from_score(turn.score, self.consecutive_low),
            dialogue=turn.dialogue,
            expression_id=turn.expression_id,
            gesture_id=turn.gesture_id,
            score=turn.score,
            is_final=is_final,
        )

    # ----- 멀티모달 피쳐 집계 -----
    def _collect_features(self, features: dict):
        if not features:
            return
        st = features.get("speakingTime")
        if isinstance(st, (int, float)):
            self.speaking_times.append(float(st))
        pc = features.get("pauseCount")
        if isinstance(pc, (int, float)):
            self.total_pauses += int(pc)

    # ----- 종료 피드백 -----
    async def build_feedback(self) -> FeedbackReport:
        avg_speak = (sum(self.speaking_times) / len(self.speaking_times)) if self.speaking_times else 0.0
        data = await llm.generate_feedback(
            company=self.info.company_name or self.company,
            job_title=self.info.job_role or self.job_title,
            transcript=self._history_text(), stage_scores=self.stage_scores,
            avg_speaking_time=avg_speak, total_pauses=self.total_pauses,
        )

        scored = [v for _, v in self.stage_scores if v >= 0]
        accuracy10 = round((sum(scored) / len(scored)) / 10) if scored else 0
        scores = InterviewScore(accuracy=accuracy10)

        overall = sum(scores.model_dump().values())

        return FeedbackReport(
            session_id=self.session_id,
            scores=scores,   
            overall_score=overall, 
            stage_scores=[StageScore(stage=s, score=v) for s, v in self.stage_scores],
            strengths=data.get("strengths", ""),
            improvements=data.get("improvements", ""),
            summary=data.get("summary", ""),
            avg_speaking_time=round(avg_speak, 1),
            total_pauses=self.total_pauses,
        )