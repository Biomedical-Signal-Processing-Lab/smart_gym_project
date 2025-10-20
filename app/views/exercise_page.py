import time
import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QColor
from PySide6.QtWidgets import QVBoxLayout
from core.evaluators.pose_angles import compute_joint_angles  

import numpy as np
from core.evaluators.pose_angles import update_meta_with_angles

from core.page_base import PageBase
from core.hailo_cam_adapter import HailoCamAdapter

from ui.overlay_painter import PoseAnglePanel, VideoCanvas, ExerciseCard, ScoreAdvicePanel, ActionButtons
from ui.score_painter import ScoreOverlay

from core.evaluators import get_evaluator_by_label, EvalResult, ExerciseEvaluator

_LABEL_KO = {
    None: "휴식중",
    "idle": "휴식중",
    "squat": "스쿼트",
    "leg_raise": "레그 레이즈",
    "pushup": "푸시업",
    "shoulder_press": "숄더 프레스",
    "side_lateral_raise": "사레레",
    "Bentover_Dumbbell": "덤벨 로우",
    "bentover_dumbbell": "덤벨 로우",
    "burpee": "버피",
    "jumping_jack": "팔벌려 뛰기",
}

EXERCISE_ORDER = [
    "squat",
    "pushup",
    "shoulder_press",
    "side_lateral_raise",
    "bentover_dumbbell",
    "leg_raise",
    "burpee",
    "jumping_jack",
]

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
        self._exercise_order: list[str] = list(EXERCISE_ORDER)
        self._per_stats: dict[str, dict] = {}  

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

        self._evaluator: ExerciseEvaluator | None = None
        self._last_eval_label: str | None = None

        self.sll_cnt = 0
        self.db_cnt = 0

        self.title_kor = "휴식중"

        # ===== 라벨 동기화(상단 디바운스) 상태 =====
        self.LABEL_HOLD_FRAMES = 30
        self._label_candidate: str | None = None
        self._label_cnt: int = 0
        self._stable_label: str = "idle"

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
        self.canvas.add_overlay(self.pose_panel, anchor="bottom-left")

        self.card.show(); self.panel.show(); self.actions.show()
        self._sync_panel_sizes()

    def _sync_panel_sizes(self):
        W, H = self.width(), self.height()
        if W <= 0 or H <= 0:
            return

        def clamp(v, lo, hi): return max(lo, min(hi, v))
        card_w = 600
        card_h = int(H * 0.5)
        self.card.setFixedSize(card_w, card_h)

        panel_w = 600
        panel_h = int(H * 0.5)
        self.panel.setFixedSize(panel_w, panel_h)

        pa_w = int(clamp(W * 0.22, 260, 380))
        pa_h = int(pa_w * 0.88)
        self.pose_panel.setFixedSize(pa_w, pa_h)

    def _init_per_stats(self):
        self._exercise_order = list(EXERCISE_ORDER)
        self._per_stats = {}
        for lb in self._exercise_order:
            name_ko = _LABEL_KO.get(lb, lb)
            self._per_stats[lb] = {
                "name": name_ko,
                "reps": 0,
                "score_sum": 0.0,
                "score_n": 0,
            }

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    def _build_summary(self):
        per_list = []
        for lb in self._exercise_order:
            ps = self._per_stats.get(lb) or {}
            reps = int(ps.get("reps", 0))
            ssum = float(ps.get("score_sum", 0.0))
            sn   = int(ps.get("score_n", 0))
            avg  = (ssum / sn) if sn > 0 else 0.0
            per_list.append({
                "name": ps.get("name", _LABEL_KO.get(lb, lb)),
                "reps": reps,
                "avg": round(avg, 1),
            })

        w_sum    = sum(float(it["avg"]) * int(it["reps"]) for it in per_list)
        reps_sum = sum(int(it["reps"]) for it in per_list)
        avg_total = (w_sum / max(reps_sum, 1)) if reps_sum > 0 else 0.0

        ended_at = time.time()
        started_at = self._session_started_ts or ended_at
        duration_sec = int(max(0, ended_at - started_at))

        return {
            "duration_sec": duration_sec,
            "avg_score": round(avg_total, 1),
            "exercises": per_list,  
        }

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
        self._init_per_stats()

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
                            self._angles_prev = angles
                            meta["_kpt"] = kpt   # ← 좌표 직접 쓰는 evaluator(버피)가 사용
                            self.pose_panel.set_angles(angles)

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

        # 라벨 정규화 + 상단 디바운스 (동기화 원천)
        raw_label = meta.get("label", None)
        curr = (str(raw_label).strip().lower().replace("-", "_") if raw_label else "idle")

        if self._label_candidate != curr:
            self._label_candidate = curr
            self._label_cnt = 1
        else:
            self._label_cnt += 1

        if not getattr(self, "_stable_label", None):
            self._stable_label = curr

        if self._label_cnt >= self.LABEL_HOLD_FRAMES and curr != self._stable_label:
            self._stable_label = curr

        stable = self._stable_label  

        # --- 한글 타이틀 갱신(안정 라벨 기준) ---
        title_kor = _LABEL_KO.get(stable, (stable if stable else "휴식중"))
        if title_kor != self._last_label:
            self.card.set_title(title_kor)
            self._last_label = title_kor

        label = stable if stable else "idle"

        # 라벨 변경 시 evaluator 교체 + reset
        if self._last_eval_label != label:
            self._last_eval_label = label
            self._evaluator = get_evaluator_by_label(label) if label not in (None, "idle") else None
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
            print(f"[Evaluator Error] {e}")
            return

        if not res:
            return

        # 코칭
        if res.advice:
            self.panel.set_advice(res.advice)

        # --- 누적: reps ---
        if res.rep_inc:
            self.reps += res.rep_inc
            if hasattr(self.card, "set_count"):
                self.card.set_count(self.reps)
            elif hasattr(self.card, "set_reps"):
                self.card.set_reps(self.reps)

            ps = self._per_stats.get(label)
            if ps is not None:
                ps["reps"] = int(ps.get("reps", 0)) + int(res.rep_inc)

        if res.score is not None:
            s = int(res.score)
            if s >= 80:
                color = QColor(0, 128, 255)     # 파란색
            elif s >= 60:
                color = QColor(0, 200, 0)       # 초록색
            elif s >= 40:
                color = QColor(255, 140, 0)     # 주황색
            else:
                color = QColor(255, 0, 0)       # 빨강  
            self.score_overlay.show_score(str(s), 250, text_qcolor=color)

            self._score_sum += float(res.score)
            self._score_n += 1
            avg = round(self._score_sum / max(1, self._score_n), 1)
            self.panel.set_avg(avg)

            # 운동별 평균용 누적
            ps = self._per_stats.get(label)
            if ps is not None:
                ps["score_sum"] = float(ps.get("score_sum", 0.0)) + float(res.score)
                ps["score_n"]   = int(ps.get("score_n", 0)) + 1

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
        try:
            angles = compute_joint_angles(kxy, kcf)  # dict 형태 반환 가정
            self.pose_panel.set_angles(angles)
        except Exception:
            pass
