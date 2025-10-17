import time
import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QVBoxLayout
from core.evaluators.pose_angles import compute_joint_angles  # 각도 계산 함수 (이미 네 코드에 있음)

import numpy as np
from core.evaluators.pose_angles import update_meta_with_angles

from core.page_base import PageBase
from core.hailo_cam_adapter import HailoCamAdapter

from ui.overlay_painter import  PoseAnglePanel, VideoCanvas, ExerciseCard, ScoreAdvicePanel, ActionButtons
from ui.score_painter import ScoreOverlay

from core.evaluators import get_evaluator_by_label, EvalResult, ExerciseEvaluator

_LABEL_KO = {
    None: "휴식중",
    "idle": "휴식중",
    "squat": "스쿼트",
    "leg_raise": "레그 레이즈",
    "pushup": "푸시업",
    "shoulder_press": "숄더 프레스",
    "Side_lateral_raise": "사이드 레터럴 레이즈",
    "Dumbbell_Row": "덤벨 로우",
    "burpee": "버피",
}

class ExercisePage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("ExercisePage")

        self.cam = None
        self.state = "UP"
        self.reps = 0

        self._score_sum = 0.0
        self._score_n = 0
        self._session_started_ts = None
        self._last_label = None

        self._no_person_since: float | None = None
        self.NO_PERSON_TIMEOUT_SEC = 10.0
        self._entered_at: float = 0.0
        self.NO_PERSON_GRACE_SEC = 1.5

        self._active = False

        self.canvas = VideoCanvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        self.canvas.set_fit_mode("cover")

        self.card = ExerciseCard("휴식중")
        self.panel = ScoreAdvicePanel()
        self.panel.set_avg(0)
        self.panel.set_advice("올바른 자세로 준비하세요.")
        self.actions = ActionButtons()
        self.actions.endClicked.connect(self._end_clicked)
        self.actions.infoClicked.connect(self._info_clicked)
        self.pose_panel = PoseAnglePanel()

        self.score_overlay = ScoreOverlay(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.canvas, 1)

        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.PAGE_FPS_MS = 33

        self._title_hold = {"label": None, "cnt": 0}
        self._evaluator: ExerciseEvaluator | None = None
        self._last_eval_label: str | None = None

    def _draw_skeleton(self, frame_bgr, people, conf_thr=0.65, show_idx=False):
        EDGES = [
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        if not people:
            return

        H, W = frame_bgr.shape[:2]
        max_len2 = (max(W, H) * 0.6) ** 2
        LINE_COLOR = (144, 238, 144)

        for p in people:
            pts = p.get("kpt", [])
            vis = [False] * len(pts)
            for j, pt in enumerate(pts):
                if len(pt) < 3:
                    continue
                if float(pt[2]) >= conf_thr:
                    vis[j] = True

            for a, b in EDGES:
                if a < len(pts) and b < len(pts) and vis[a] and vis[b]:
                    x1_, y1_ = int(pts[a][0]), int(pts[a][1])
                    x2_, y2_ = int(pts[b][0]), int(pts[b][1])
                    dx, dy = x1_ - x2_, y1_ - y2_
                    if (dx * dx + dy * dy) <= max_len2:
                        cv2.line(frame_bgr, (x1_, y1_), (x2_, y2_), LINE_COLOR, 2)

    def _mount_overlays(self):
        self.canvas.clear_overlays()
        self.canvas.add_overlay(self.card, anchor="top-left")
        self.canvas.add_overlay(self.panel, anchor="top-right")
        self.canvas.add_overlay(self.actions, anchor="bottom-right")
        self.canvas.add_overlay(self.pose_panel, anchor="bottom-left")   # ← 추가

        self.card.show(); self.panel.show(); self.actions.show()
        self._sync_panel_sizes()

    def _sync_panel_sizes(self):
        W, H = self.width(), self.height()
        target_w = int(max(320, min(W * 0.26, 460)))
        target_h = int(target_w * 0.90)
        self.card.setFixedSize(target_w, target_h)
        self.panel.setFixedSize(target_w, target_h)

        # 좌하단 각도 패널은 조금 더 작게
        pa_w = int(max(260, min(W * 0.22, 380)))
        pa_h = int(pa_w * 0.88)
        self.pose_panel.setFixedSize(pa_w, pa_h)


    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    def _build_summary(self):
        per_list = []  # TODO: 세션 중 실제 기록을 원하면 여기에 누적해서 넣기
        w_sum = sum(float(x.get("avg", 0.0)) * int(x.get("reps", 0)) for x in per_list)
        reps_sum = sum(int(x.get("reps", 0)) for x in per_list) or 1
        avg_total = w_sum / reps_sum
        ended_at = time.time()
        started_at = self._session_started_ts or ended_at
        duration_sec = int(max(0, ended_at - started_at))
        return {"duration_sec": duration_sec, "avg_score": round(avg_total, 1), "exercises": per_list}

    def _end_clicked(self):
        self._active = False
        if self.timer.isActive():
            self.timer.stop()
        try:
            self.ctx.cam.stop()
        except Exception:
            pass
        summary = self._build_summary()
        try:
            if hasattr(self.ctx, "save_workout_session"):
                self.ctx.save_workout_session(summary)
        except Exception:
            pass
        if hasattr(self.ctx, "goto_summary"):
            self.ctx.goto_summary(summary)
        self.canvas.clear_overlays()

    def _info_clicked(self):
        try:
            if hasattr(self.ctx, "goto_profile"):
                self.ctx.goto_profile()
        except Exception:
            pass

    def on_enter(self, ctx):
        self.ctx = ctx
        self._session_started_ts = time.time()
        self._score_sum = 0.0
        self._score_n = 0
        self._reset_state()

        self._evaluator = None
        self._last_eval_label = None

        self._no_person_since = None
        self._entered_at = time.time()

        title_text = getattr(self.ctx, "current_exercise", None) or "휴식중"
        self.card.set_title(title_text)

        self._mount_overlays()
        self.pose_panel.set_angles({})

        try:
            self.ctx.face.stop_stream()
        except Exception:
            pass

        if not hasattr(self.ctx, "cam") or self.ctx.cam is None:
            self.ctx.cam = HailoCamAdapter()

        self.ctx.cam.start()

        self._active = True
        if self.timer.isActive():
            self.timer.stop()
        self.timer.start(self.PAGE_FPS_MS)

    def on_leave(self, ctx):
        self._active = False
        if self.timer.isActive():
            self.timer.stop()
        try:
            ctx.cam.stop()
        except Exception:
            pass
        self.canvas.clear_overlays()
        # (선택) 깔끔 정리
        self._evaluator = None
        self._last_eval_label = None

    def _tick(self):
        if not self._active or not self.timer.isActive():
            return

        meta = self.ctx.cam.meta() or {}
        now = time.time()
        in_grace = (now - self._entered_at) < self.NO_PERSON_GRACE_SEC

        # --- no-person 타임아웃 ---
        m_ok = bool(meta.get("ok", False))
        if in_grace:
            self._no_person_since = None
        else:
            if not m_ok:
                if self._no_person_since is None:
                    self._no_person_since = now
                elif (now - self._no_person_since) >= self.NO_PERSON_TIMEOUT_SEC:
                    self._active = False
                    try:
                        if self.timer.isActive():
                            self.timer.stop()
                    except Exception:
                        pass
                    try:
                        self.ctx.cam.stop()
                    except Exception:
                        pass
                    self._no_person_since = None
                    self._goto("guide")
                    return
            else:
                self._no_person_since = None

        # --- 프레임 + 스켈레톤 ---
        if not self._active:
            return

        frame = self.ctx.cam.frame()

        if frame is not None:
            try:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                people = self.ctx.cam.people()
                self._draw_skeleton(bgr, people, conf_thr=0.65)

                try:
                    if people:
                        kpt = people[0].get("kpt", [])
                        if kpt and len(kpt) >= 17:
                            kxy = np.array([[pt[0], pt[1]] for pt in kpt], dtype=np.float32)
                            kcf = np.array([(pt[2] if len(pt) > 2 else 1.0) for pt in kpt], dtype=np.float32)

                            # meta에 각도 자동 추가 (Elbow, Shoulder, Knee, Hip 등)

                            angles = update_meta_with_angles(
                                meta,
                                kxy,
                                kcf,
                                conf_thr=0.5,
                                ema=0.2,
                                prev=getattr(self, "_angles_prev", None),
                            )
                            #print(angles)
                            self._angles_prev = angles
                            meta["_kpt"] = kpt   # ← 좌표 직접 쓰는 evaluator(버피)가 사용


                        
                except Exception as _e:
                    # 각도 계산 오류 발생해도 멈추지 않게
                    print(f"[angle_calc error] {_e}")
                    pass

                frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                if self._active:
                    self.canvas.set_frame(qimg)
            except cv2.error:
                return

        # --- TCN 라벨 → 한글 제목 (2프레임 홀드) ---
        raw_label = meta.get("label", None)
        title_kor = _LABEL_KO.get(raw_label, (raw_label if raw_label else "휴식중"))

        hold = self._title_hold
        if hold["label"] != title_kor:
            hold["label"] = title_kor
            hold["cnt"] = 1
        else:
            hold["cnt"] += 1

        if hold["cnt"] >= 2 and title_kor != self._last_label:
            self.card.set_title(title_kor)
            self._last_label = title_kor

        # === evaluator 연결 (라벨 기반) ===
        label = raw_label if raw_label else "idle"

        # 라벨 변경 시 evaluator 교체 + reset
        if self._last_eval_label != label:
            self._last_eval_label = label
            self._evaluator = get_evaluator_by_label(label) if label not in (None, "idle") else None
            print(type(self._evaluator))
            if self._evaluator:
                self._evaluator.reset()

        # 휴식/미인식 상태면 기본 코칭만 표시
        if label in (None, "idle") or not self._evaluator:
            self.panel.set_advice("올바른 자세로 준비하세요.")
            return

        # evaluator로 한 프레임 평가
        try:
            res: EvalResult = self._evaluator.update(meta)
        except Exception as e:
                # 현장 디버깅용: 문제 생겨도 UI 멈추지 않게
                print(f"[Evaluator Error] {e}")
                return
        # 결과 없으면 끝
        if not res:
            return

        # 코칭
        if res.advice:
            self.panel.set_advice(res.advice)

        # rep 증가
        if res.rep_inc:
            self.reps += res.rep_inc
            if hasattr(self.card, "set_count"):
                self.card.set_count(self.reps)
            elif hasattr(self.card, "set_reps"):
                self.card.set_reps(self.reps)

        # 점수 표시 및 평균 갱신
        if res.score is not None:
            if res.color:
                self.score_overlay.show_score(str(int(res.score)), 100, text_qcolor=res.color)
            else:
                self.score_overlay.show_score(str(int(res.score)), 100)

            self._score_sum += float(res.score)
            self._score_n += 1
            avg = round(self._score_sum / max(1, self._score_n), 1)
            self.panel.set_avg(avg)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_panel_sizes()
        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

    def _reset_state(self):
        self.state = "UP"
        self.reps = 0
        self.card.set_count(0)
        self.panel.set_avg(0)
        self.panel.set_advice("올바른 자세로 준비하세요.")
        if self._evaluator:
            self._evaluator.reset()
        self.pose_panel.set_angles({})


    def on_pose_frame(self, kxy, kcf):
        """
        kxy: (17,2) numpy array (x,y)
        kcf: (17,) numpy array (confidence)
        네가 이미 사용중인 compute_joint_angles()로 각도 dict 생성.
        """
        try:
            angles = compute_joint_angles(kxy, kcf)  # dict 형태 반환 가정
            # 예시 keys:
            # 'Knee(L)','Knee(R)','Hip(L)','Hip(R)','Shoulder(L)','Shoulder(R)',
            # 'Elbow(L)','Elbow(R)','HipLine(L)','HipLine(R)'

            # 3) 패널 업데이트
            self.pose_panel.set_angles(angles)
        except Exception as e:
            # 안전하게 실패 시 화면만 유지
            # print(f"[pose] angle update error: {e}")
            pass
