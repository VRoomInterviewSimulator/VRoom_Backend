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
        
        self.cps_list: list[float] = []
        self.meaningful_pauses: list[int] = []
        self.volume_variances: list[float] = []
        self.low_volume_ratios: list[float] = []
        self.response_times: list[float] = []
        self.average_volumes: list[float] = []
        self.turn_stages: list[str] = []
        self.vision_turns: list[dict] = []

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
        "안녕하세요. {company} {job} 직무 면접을 시작하겠습니다. "
        "먼저 지원 동기를 포함해 자기소개를 부탁드립니다."
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
        self.turn_stages.append(self.stage.value)
        self._collect_features(features, text)

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
    def _collect_features(self, features: dict, text: str = ""):
        if not features:
            return
        st = features.get("speakingTime")
        if isinstance(st, (int, float)) and st > 0:
            self.speaking_times.append(float(st))
            cps = len(text) / float(st)
            self.cps_list.append(cps)
            
        self.meaningful_pauses.append(int(features.get("meaningfulPauseCount", features.get("pauseCount", 0))))
        self.volume_variances.append(float(features.get("volumeVariance", 0.0)))
        self.low_volume_ratios.append(float(features.get("lowVolumeRatio", 0.0)))
        self.response_times.append(float(features.get("responseTime", 0.0)))
        self.average_volumes.append(float(features.get("averageVolume", 0.1)))

    # ----- 종료 피드백 -----
    async def build_feedback(self) -> FeedbackReport:
        from .config import ScoringConfig
        
        turn_count = len(self.speaking_times)
        avg_speak = (sum(self.speaking_times) / turn_count) if turn_count > 0 else 0.0
        total_pauses = sum(self.meaningful_pauses)
        
        data = await llm.generate_feedback(
            company=self.info.company_name or self.company,
            job_title=self.info.job_role or self.job_title,
            transcript=self._history_text(), stage_scores=self.stage_scores,
            avg_speaking_time=avg_speak, total_pauses=total_pauses,
        )

        scored = [v for _, v in self.stage_scores if v >= 0]
        accuracy10 = round((sum(scored) / len(scored)) / 10) if scored else 0
        
        total_vol_score = 0
        total_speed_score = 0
        total_length_score = 0
        total_rt_score = 0

        # 각 턴별로 개별 점수를 매기고 합산
        for i in range(turn_count):
            # 1. Voice Volume (턴별)
            vol = self.average_volumes[i] if i < len(self.average_volumes) else ScoringConfig.VOICE_VOLUME_MEAN
            var = self.volume_variances[i] if i < len(self.volume_variances) else 0.0
            low = self.low_volume_ratios[i] if i < len(self.low_volume_ratios) else 0.0
            
            s_vol = 10 - int(abs(vol - ScoringConfig.VOICE_VOLUME_MEAN) / ScoringConfig.VOICE_VOLUME_TOLERANCE)
            if var > ScoringConfig.VOLUME_VARIANCE_THRESHOLD: s_vol -= ScoringConfig.VOLUME_VARIANCE_PENALTY
            if low > ScoringConfig.LOW_VOLUME_RATIO_THRESHOLD: s_vol -= ScoringConfig.LOW_VOLUME_RATIO_PENALTY
            total_vol_score += max(0, min(10, s_vol))
            
            # 2. Voice Speed (턴별)
            cps = self.cps_list[i] if i < len(self.cps_list) else ScoringConfig.VOICE_SPEED_MEAN
            pauses = self.meaningful_pauses[i] if i < len(self.meaningful_pauses) else 0
            
            s_speed = 10 - int(abs(cps - ScoringConfig.VOICE_SPEED_MEAN) / ScoringConfig.VOICE_SPEED_TOLERANCE)
            s_speed -= int(max(0, pauses - ScoringConfig.PAUSE_ALLOWANCE))
            total_speed_score += max(0, min(10, s_speed))
            
            # 3. Answer Length (턴별 물리적 시간 점수)
            st = self.speaking_times[i]
            if st < ScoringConfig.ANSWER_LENGTH_MIN:
                s_len = int((st / ScoringConfig.ANSWER_LENGTH_MIN) * 10)
            elif st > ScoringConfig.ANSWER_LENGTH_MAX:
                s_len = 10 - int((st - ScoringConfig.ANSWER_LENGTH_MAX) / 10)
            else:
                s_len = 10
            total_length_score += max(0, min(10, s_len))
            
            # 4. Response Time (턴별)
            rt = self.response_times[i] if i < len(self.response_times) else 0.0
            stage = self.turn_stages[i] if i < len(self.turn_stages) else "UNKNOWN"
            target_rt = ScoringConfig.RESPONSE_TIME_INTRO_MEAN if stage == "SELF_INTRO" else ScoringConfig.RESPONSE_TIME_FOLLOWUP_MEAN
            
            s_rt = 10 - int(abs(rt - target_rt) / ScoringConfig.RESPONSE_TIME_TOLERANCE)
            total_rt_score += max(0, min(10, s_rt))

        # 합산된 점수의 평균 산출 (턴이 없으면 기본 만점)
        vol_score = round(total_vol_score / turn_count) if turn_count > 0 else 10
        speed_score = round(total_speed_score / turn_count) if turn_count > 0 else 10
        rt_score = round(total_rt_score / turn_count) if turn_count > 0 else 10
        
        # Answer Length 최종 산출: (턴별 물리적 시간 점수 평균 + LLM이 평가한 내용 밀도 점수) / 2
        time_score_avg = round(total_length_score / turn_count) if turn_count > 0 else 10
        density_score = data.get("density_score", 5)
        length_score = (time_score_avg + density_score) // 2
        length_score = max(0, min(10, length_score))
        
        # 5. Filler Words (LLM 평가)
        filler_score = data.get("filler_score", 5)
        
        vision_ok, gaze_s, gesture_s, posture_s, expr_s = self._score_vision()

        scores = InterviewScore(
            accuracy=accuracy10,
            voiceVolume=vol_score,
            voiceSpeed=speed_score,
            answerLength=length_score,
            fillerWords=filler_score,
            responseTime=rt_score,
            gaze=gaze_s,
            gesture=gesture_s,
            posture=posture_s,
            expression=expr_s,
        )

        if vision_ok:
            overall = sum(scores.model_dump().values())      # 10항목 / 100점
        else:
            # 웹캠 미사용 세션: 음성 6항목(60점 만점)을 100점으로 환산
            voice_sum = (accuracy10 + vol_score + speed_score
                         + length_score + filler_score + rt_score)
            overall = round(voice_sum / 60 * 100)

        return FeedbackReport(
            session_id=self.session_id,
            scores=scores,   
            overall_score=overall, 
            stage_scores=[StageScore(stage=s, score=v) for s, v in self.stage_scores],
            strengths=data.get("strengths", ""),
            improvements=data.get("improvements", ""),
            summary=data.get("summary", ""),
            avg_speaking_time=round(avg_speak, 1),
            total_pauses=total_pauses,
        )

    # ----- 시각(웹캠) 피쳐 -----
    def collect_vision_features(self, features: dict):
        if features:
            self.vision_turns.append(features)

    def _score_vision(self) -> tuple[bool, int, int, int, int]:
        """(사용가능여부, gaze, gesture, posture, expression)"""
        import math
        from .config import VisionScoringConfig as V

        turns = [t for t in self.vision_turns
                 if t.get("faceDetectedRatio", 0.0) >= V.MIN_FACE_RATIO]
        if not turns:
            print("[채점] 유효한 시각 피쳐 없음 -> 시각 4항목 제외")
            return (False, 0, 0, 0, 0)

        g = ge = p = e = 0
        for t in turns:
            # --- gaze ---
            r = t.get("gazeOnTargetRatio", 0.0)
            if r >= V.GAZE_RATIO_FULL:
                s = 10
            elif r <= V.GAZE_RATIO_ZERO:
                s = 0
            else:
                s = round(10 * (r - V.GAZE_RATIO_ZERO)
                          / (V.GAZE_RATIO_FULL - V.GAZE_RATIO_ZERO))
            jitter = math.hypot(t.get("headYawStd", 0.0), t.get("headPitchStd", 0.0))
            s -= int(max(0.0, jitter - V.GAZE_JITTER_TOLERANCE) / V.GAZE_JITTER_STEP)
            g += max(0, min(10, s))

            # --- posture ---
            s = 10
            s -= int(max(0.0, t.get("shoulderTiltMean", 0.0)
                         - V.POSTURE_TILT_TOLERANCE) / V.POSTURE_TILT_STEP)
            s -= int(max(0.0, t.get("bodySwayStd", 0.0)
                         - V.POSTURE_SWAY_TOLERANCE) / V.POSTURE_SWAY_STEP)
            if t.get("calibrated"):
                s -= int(max(0.0, t.get("torsoDriftMean", 0.0)
                             - V.POSTURE_DRIFT_TOLERANCE) / V.POSTURE_DRIFT_STEP)
            p += max(0, min(10, s))

            # --- gesture ---
            u = t.get("handUsageRatio", 0.0)
            if u <= V.GESTURE_USAGE_HARD_ZERO:
                s = V.GESTURE_ZERO_SCORE
            elif u < V.GESTURE_USAGE_MIN:
                s = 10 - int((V.GESTURE_USAGE_MIN - u) / V.GESTURE_UNDER_STEP)
            elif u > V.GESTURE_USAGE_MAX:
                s = 10 - int((u - V.GESTURE_USAGE_MAX) / V.GESTURE_OVER_STEP)
            else:
                s = 10
            # 손이 보이지만 한 자리에 굳어 있으면 감점
            if u >= V.GESTURE_USAGE_MIN and \
                    t.get("handExtent", 0.0) < V.GESTURE_EXTENT_MIN:
                s -= V.GESTURE_EXTENT_PENALTY
            s -= max(0, t.get("faceTouchCount", 0) - V.GESTURE_FACE_TOUCH_ALLOWANCE)
            ge += max(0, min(10, s))

            # --- expression ---
            s = 10
            if t.get("expressionVariance", 0.0) < V.EXPRESSION_RIGID_THRESHOLD:
                s -= V.EXPRESSION_RIGID_PENALTY
            s -= int(max(0.0, t.get("frownRatio", 0.0)
                         - V.EXPRESSION_FROWN_TOLERANCE) / V.EXPRESSION_FROWN_STEP)
            if V.EXPRESSION_BLINK_ENABLED:
                bpm = t.get("blinkPerMinute", 15.0)
                if bpm > V.EXPRESSION_BLINK_MAX:
                    s -= int((bpm - V.EXPRESSION_BLINK_MAX) / V.EXPRESSION_BLINK_STEP)
                elif bpm < V.EXPRESSION_BLINK_MIN:
                    s -= 1
            if t.get("smileRatio", 0.0) >= V.EXPRESSION_SMILE_BONUS_RATIO:
                s += 1
            e += max(0, min(10, s))

        n = len(turns)
        return (True, round(g / n), round(ge / n), round(p / n), round(e / n))