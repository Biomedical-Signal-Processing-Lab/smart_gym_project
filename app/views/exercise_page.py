# app/ui/views/exercise_page.py
import time
import os, shlex, subprocess, signal
from pathlib import Path
import cv2
import numpy as np

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout

from core.evaluators.pose_angles import compute_joint_angles, update_meta_with_angles
from core.page_base import PageBase
from core.hailo_cam_adapter import HailoCamAdapter

from ui.overlay_painter import PoseAnglePanel, VideoCanvas, ExerciseCard, ScoreAdvicePanel, ActionButtons
from ui.score_painter import ScoreOverlay
from core.evaluators import get_evaluator_by_label, EvalResult, ExerciseEvaluator

## 1) 서비스(CWD 보장 + 상대경로 사용)
#export SGP_SERVICE_CMD='/bin/bash -lc "cd ~/workspace/python/smart_gym_project && python3 sensor/실전_tempo_balance_fatigue/squat_service_dual.py --user-seq --imu-master L --pair-lag-ms 980"'

# 2) UI에서 읽을 로그 위치 (두 파일이 있는 폴더)
#export SGP_LOG_DIR="~/workspace/python/smart_gym_project/sensor/data/logs"

# ==========================
# 로그 폴더 자동 탐색
# ==========================
def _resolve_log_dir() -> Path:
    # 1) 환경변수
    p = os.getenv("SGP_LOG_DIR")
    if p:
        pp = Path(p).expanduser().resolve()
        if (pp / "reps_pred_dual.tsv").exists():
            return pp

    here = Path(__file__).resolve()
    app_dir = here.parents[2]
    proj_root = app_dir.parent

    # 2) 앱 내부 기본
    c1 = app_dir / "data" / "logs"
    # 3-1) 형제 sensor 프로젝트(기존 경로)
    c2a = proj_root / "sensor" / "실전_tempo_balance_fatigue" / "data" / "logs"
    # 3-2) 형제 sensor 프로젝트(새 구조: sensor/data/logs)
    c2b = proj_root / "sensor" / "data" / "logs"
    # 4) 현재 작업 폴더
    c3 = Path.cwd() / "data" / "logs"

    for c in (c1, c2a, c2b, c3):
        if (c / "reps_pred_dual.tsv").exists():
            return c.resolve()

    return c1.resolve()


LOG_DIR  = _resolve_log_dir()
PRED_TSV = LOG_DIR / "reps_pred_dual.tsv"
IMU_TSV  = LOG_DIR / "imu_tempo.tsv"


# ==========================
# 좌하단: 실시간 AI/IMU 패널 (한글 라벨)
# ==========================
class AIMetricsPanel(QWidget):
    """
    imu_tempo.tsv:  user_id, tempo_score, tempo_level, imu_state
    reps_pred_dual.tsv: FI_L, FI_R, stage_L, stage_R, BI, BI_stage, BI_text
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 160);
                border-radius: 16px;
            }
            QLabel#title {
                color: #FFFFFF; font-size: 18px; font-weight: 800; letter-spacing: 0.5px;
                background: transparent;
            }
            QLabel#dim {
                color: #BFC6CF; font-size: 14px; font-weight: 600; background: transparent;
            }
            QLabel#val {
                color: #FFFFFF; font-size: 16px; font-weight: 600; background: transparent;
            }
        """)
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(14, 12, 14, 12)
        wrap.setSpacing(8)

        title = QLabel("실시간 지표", self)
        title.setObjectName("title")
        wrap.addWidget(title)

        grid_host = QWidget(self)
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        rows = [
            # IMU (imu_tempo.tsv)
            ("이름",        "user_id"),
            ("템포점수",     "tempo_score"),
            ("템포판정",     "tempo_level"),
            ("상태",        "imu_state"),
            # AI (reps_pred_dual.tsv)
            ("L_피로도",      "fi_l"),
            ("R_피로도",      "fi_r"),
            ("L_피로도판정",  "stage_l"),
            ("R_피로도판정",  "stage_r"),
            ("불균형",        "bi"),
            ("불균형판정",    "bi_stage"),
            ("불균형설명",    "bi_text"),
        ]

        self._cells = {}
        for r, (lab, key) in enumerate(rows):
            k = QLabel(lab, self); k.setObjectName("dim")
            v = QLabel("-",  self); v.setObjectName("val")
            grid.addWidget(k, r, 0)
            grid.addWidget(v, r, 1)
            self._cells[key] = v

        wrap.addWidget(grid_host)

    @staticmethod
    def _fmt_num(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "-"

    def set_imu(self, user_id=None, tempo_score=None, tempo_level=None, imu_state=None):
        if user_id is not None:     self._cells["user_id"].setText(str(user_id))
        if tempo_score is not None: self._cells["tempo_score"].setText(str(int(tempo_score)))
        if tempo_level is not None: self._cells["tempo_level"].setText(str(tempo_level))
        if imu_state is not None:   self._cells["imu_state"].setText(str(imu_state))

    def set_ai(self, fi_l=None, fi_r=None, stage_l=None, stage_r=None,
               bi=None, bi_stage=None, bi_text=None):
        if fi_l is not None:     self._cells["fi_l"].setText(self._fmt_num(fi_l))
        if fi_r is not None:     self._cells["fi_r"].setText(self._fmt_num(fi_r))
        if stage_l is not None:  self._cells["stage_l"].setText(str(stage_l))
        if stage_r is not None:  self._cells["stage_r"].setText(str(stage_r))
        if bi is not None:       self._cells["bi"].setText(self._fmt_num(bi))
        if bi_stage is not None: self._cells["bi_stage"].setText(str(bi_stage))
        if bi_text is not None:  self._cells["bi_text"].setText(str(bi_text))


# ==========================
# 페이지 본체
# ==========================
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

        self._no_person_since = None
        self.NO_PERSON_TIMEOUT_SEC = 10.0
        self._entered_at: float = 0.0
        self.NO_PERSON_GRACE_SEC = 1.5

        self._active = False

        # 외부 서비스 프로세스 핸들 (squat_service_dual.py)
        self._svc_proc = None

        # 비디오/오버레이 UI
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

        # 좌하단: 사용자 요구 포맷의 실시간 패널
        self.ai_panel = AIMetricsPanel()

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

        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self._poll_tsv)
        self.AI_POLL_MS = 250

        self._title_hold = {"label": None, "cnt": 0}
        self._evaluator: ExerciseEvaluator | None = None
        self._last_eval_label: str | None = None

        self._last_pred_size = 0
        self._last_imu_size = 0

        self.pose_panel = PoseAnglePanel()
        self._angles_prev = None

    # ---------- 서비스 실행/종료 ----------
    def _start_service_if_needed(self):
        """
        환경변수 SGP_SERVICE_CMD 설정 시 서비스 자동 실행.
        예) SGP_SERVICE_CMD='/bin/bash -lc "cd ~/workspace/python/smart_gym_project && python3 sensor/실전_tempo_balance_fatigue/squat_service_dual.py --user-seq --imu-master L --pair-lag-ms 980"'
        """
        if self._svc_proc is not None and self._svc_proc.poll() is None:
            return
        cmd = os.getenv("SGP_SERVICE_CMD")
        if not cmd:
            return
        try:
            args = shlex.split(cmd)
            # ★ 핵심: 새 세션으로 실행하여 프로세스그룹 리더가 되게 함
            self._svc_proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
                start_new_session=True,  # ← 이 옵션이 있어야 os.killpg로 Ctrl+C 전파 가능
            )
            print("[DEBUG] SGP_SERVICE_CMD =", cmd)
            print("[DEBUG] parsed args =", args)

            print(f"[ExercisePage] started service pid={self._svc_proc.pid}")
        except Exception as e:
            print(f"[ExercisePage] start service failed: {e}")
            self._svc_proc = None

    def _stop_service(self):
        """버튼으로 서비스 정상 종료: (프로세스그룹 전체) SIGINT → TERM → KILL"""
        p = self._svc_proc
        self._svc_proc = None
        if not p:
            return
        try:
            if p.poll() is None:
                # ★ 그룹 전체에 Ctrl+C 신호
                os.killpg(p.pid, signal.SIGINT)
                p.wait(timeout=5)
        except Exception:
            pass
        try:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=2)
        except Exception:
            pass
        try:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        print("[ExercisePage] service stopped")

    # ---------- skeleton ----------
    def _draw_skeleton(self, frame_bgr, people, conf_thr=0.65):
        EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]
        if not people: return
        H, W = frame_bgr.shape[:2]
        max_len2 = (max(W, H)*0.6) ** 2
        LINE_COLOR = (144, 238, 144)
        for p in people:
            pts = p.get("kpt", [])
            vis = [len(pt)>=3 and float(pt[2])>=conf_thr for pt in pts]
            for a,b in EDGES:
                if a<len(pts) and b<len(pts) and vis[a] and vis[b]:
                    x1_,y1_ = int(pts[a][0]), int(pts[a][1])
                    x2_,y2_ = int(pts[b][0]), int(pts[b][1])
                    dx,dy = x1_-x2_, y1_-y2_
                    if (dx*dx + dy*dy) <= max_len2:
                        cv2.line(frame_bgr, (x1_,y1_), (x2_,y2_), LINE_COLOR, 2)

    # ---------- overlays ----------
    def _mount_overlays(self):
        self.canvas.clear_overlays()
        self.canvas.add_overlay(self.card, anchor="top-left")
        self.canvas.add_overlay(self.panel, anchor="top-right")
        self.canvas.add_overlay(self.actions, anchor="bottom-right")
        self.canvas.add_overlay(self.ai_panel, anchor="bottom-left")
        self.card.show(); self.panel.show(); self.actions.show(); self.ai_panel.show()
        self._sync_panel_sizes()

    def _sync_panel_sizes(self):
        W, H = self.width(), self.height()
        target_w = int(max(320, min(W * 0.26, 460)))
        target_h = int(target_w * 0.90)
        self.card.setFixedSize(target_w, target_h)
        self.panel.setFixedSize(target_w, target_h)
        pa_w = int(max(260, min(W * 0.22, 380)))
        pa_h = int(pa_w * 1.10)
        self.ai_panel.setFixedSize(pa_w, pa_h)

    # ---------- nav ----------
    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    # ---------- session ----------
    def _build_summary(self):
        per_list = []
        w_sum = sum(float(x.get("avg", 0.0)) * int(x.get("reps", 0)) for x in per_list)
        reps_sum = sum(int(x.get("reps", 0)) for x in per_list) or 1
        avg_total = w_sum / reps_sum
        ended_at = time.time()
        started_at = self._session_started_ts or ended_at
        duration_sec = int(max(0, ended_at - started_at))
        return {"duration_sec": duration_sec, "avg_score": round(avg_total, 1), "exercises": per_list}

    def _end_clicked(self):
        self._active = False
        if self.timer.isActive(): self.timer.stop()
        if self.ai_timer.isActive(): self.ai_timer.stop()
        try: self.ctx.cam.stop()
        except Exception: pass

        # 서비스도 종료
        self._stop_service()

        summary = self._build_summary()
        try:
            if hasattr(self.ctx, "save_workout_session"): self.ctx.save_workout_session(summary)
        except Exception: pass
        if hasattr(self.ctx, "goto_summary"): self.ctx.goto_summary(summary)
        self.canvas.clear_overlays()

    def _info_clicked(self):
        try:
            if hasattr(self.ctx, "goto_profile"): self.ctx.goto_profile()
        except Exception:
            pass

    # ---------- lifecycle ----------
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
        self.ai_panel.set_imu(user_id="-", tempo_score=None, tempo_level=None, imu_state=None)
        self.ai_panel.set_ai(fi_l=None, fi_r=None, stage_l=None, stage_r=None, bi=None, bi_stage=None, bi_text=None)

        try:
            self.ctx.face.stop_stream()
        except Exception:
            pass

        if not hasattr(self.ctx, "cam") or self.ctx.cam is None:
            self.ctx.cam = HailoCamAdapter()
        self.ctx.cam.start()

        # 필요 시 서비스 자동 실행
        self._start_service_if_needed()

        self._active = True
        if self.timer.isActive(): self.timer.stop()
        self.timer.start(self.PAGE_FPS_MS)

        self._last_pred_size = 0
        self._last_imu_size = 0
        if self.ai_timer.isActive(): self.ai_timer.stop()
        self.ai_timer.start(self.AI_POLL_MS)

    def on_leave(self, ctx):
        self._active = False
        if self.timer.isActive(): self.timer.stop()
        if self.ai_timer.isActive(): self.ai_timer.stop()
        try: ctx.cam.stop()
        except Exception: pass

        # 페이지 이탈 시에도 안전 종료
        self._stop_service()

        self.canvas.clear_overlays()
        self._evaluator = None
        self._last_eval_label = None

    # ---------- TSV polling ----------
    def _poll_tsv(self):
        try:
            # reps_pred_dual.tsv  (ts,user_id,rep_id,FI_L,FI_R,AIF,AI_RMS,AI_iEMG,BI,stage_L,stage_R,BI_stage,BI_text)
            if PRED_TSV.exists():
                sz = PRED_TSV.stat().st_size
                if sz != self._last_pred_size and sz > 0:
                    self._last_pred_size = sz
                    with PRED_TSV.open("r", encoding="utf-8") as f:
                        lines = f.readlines()
                    if len(lines) >= 2:
                        last = lines[-1].strip().split("\t")
                        fi_l     = float(last[3]) if len(last) > 3 else None
                        fi_r     = float(last[4]) if len(last) > 4 else None
                        bi       = float(last[8]) if len(last) > 8 else None
                        stage_l  = last[9]  if len(last) > 9  else None
                        stage_r  = last[10] if len(last) > 10 else None
                        bi_stage = last[11] if len(last) > 11 else None
                        bi_text  = last[12] if len(last) > 12 else None
                        self.ai_panel.set_ai(fi_l=fi_l, fi_r=fi_r, stage_l=stage_l,
                                             stage_r=stage_r, bi=bi, bi_stage=bi_stage, bi_text=bi_text)

            # imu_tempo.tsv (ts_unix,user_id,side,ts_ms,imu_state_num,imu_state,rep_id,desc_ms,rise_ms,tempo_cv,tempo_score,tempo_level,pitch_deg,pitch_vel)
            if IMU_TSV.exists():
                sz2 = IMU_TSV.stat().st_size
                if sz2 != self._last_imu_size and sz2 > 0:
                    self._last_imu_size = sz2
                    with IMU_TSV.open("r", encoding="utf-8") as f:
                        lines2 = f.readlines()
                    if len(lines2) >= 2:
                        last2 = lines2[-1].strip().split("\t")
                        user_id     = last2[1] if len(last2) > 1 else "-"
                        imu_state   = last2[5] if len(last2) > 5 else None
                        tempo_score = int(float(last2[10])) if len(last2) > 10 else None
                        tempo_level = last2[11] if len(last2) > 11 else None
                        self.ai_panel.set_imu(user_id=user_id, tempo_score=tempo_score,
                                              tempo_level=tempo_level, imu_state=imu_state)
        except Exception:
            # 파일 append 중 읽기 충돌 등은 조용히 무시
            pass

    # ---------- video tick ----------
    def _tick(self):
        if not self._active or not self.timer.isActive():
            return

        meta = self.ctx.cam.meta() or {}
        now = time.time()
        in_grace = (now - self._entered_at) < self.NO_PERSON_GRACE_SEC

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
                        if self.timer.isActive(): self.timer.stop()
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

        frame = self.ctx.cam.frame()
        if frame is not None:
            try:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                people = self.ctx.cam.people()
                self._draw_skeleton(bgr, people, conf_thr=0.65)

                # 포즈 각도(옵션)
                try:
                    if people:
                        kpt = people[0].get("kpt", [])
                        if kpt and len(kpt) >= 17:
                            kxy = np.array([[pt[0], pt[1]] for pt in kpt], dtype=np.float32)
                            kcf = np.array([(pt[2] if len(pt) > 2 else 1.0) for pt in kpt], dtype=np.float32)
                            angles = update_meta_with_angles(
                                meta, kxy, kcf, conf_thr=0.5, ema=0.2,
                                prev=getattr(self, "_angles_prev", None),
                            )
                            self._angles_prev = angles
                            meta["_kpt"] = kpt
                except Exception:
                    pass

                frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                if self._active:
                    self.canvas.set_frame(qimg)
            except cv2.error:
                return

        # 제목(한국어)
        raw_label = meta.get("label", None)
        title_kor = _LABEL_KO.get(raw_label, (raw_label if raw_label else "휴식중"))
        hold = self._title_hold
        if hold["label"] != title_kor:
            hold["label"] = title_kor; hold["cnt"] = 1
        else:
            hold["cnt"] += 1
        if hold["cnt"] >= 2 and title_kor != self._last_label:
            self.card.set_title(title_kor)
            self._last_label = title_kor

        # evaluator
        label = raw_label if raw_label else "idle"
        if self._last_eval_label != label:
            self._last_eval_label = label
            self._evaluator = get_evaluator_by_label(label) if label not in (None, "idle") else None
            if self._evaluator: self._evaluator.reset()

        if label in (None, "idle") or not self._evaluator:
            self.panel.set_advice("올바른 자세로 준비하세요.")
            return

        try:
            res: EvalResult = self._evaluator.update(meta)
        except Exception:
            return
        if not res: return

        if res.advice: self.panel.set_advice(res.advice)
        if res.rep_inc:
            self.reps += res.rep_inc
            if hasattr(self.card, "set_count"): self.card.set_count(self.reps)
            elif hasattr(self.card, "set_reps"): self.card.set_reps(self.reps)

        if res.score is not None:
            if res.color:
                self.score_overlay.show_score(str(int(res.score)), 100, text_qcolor=res.color)
            else:
                self.score_overlay.show_score(str(int(res.score)), 100)
            self._score_sum += float(res.score)
            self._score_n += 1
            avg = round(self._score_sum / max(1, self._score_n), 1)
            self.panel.set_avg(avg)

    # ---------- misc ----------
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
        if self._evaluator: self._evaluator.reset()
        self.ai_panel.set_imu(user_id="-", tempo_score=None, tempo_level=None, imu_state=None)
        self.ai_panel.set_ai(fi_l=None, fi_r=None, stage_l=None, stage_r=None, bi=None, bi_stage=None, bi_text=None)
