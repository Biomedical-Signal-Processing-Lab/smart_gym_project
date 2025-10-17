# SQUAT/app/ui/effect_overlay.py
from __future__ import annotations
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget

# 최소 틀: 점수 오버레이처럼 화면 위에 반투명 레이어로 이펙트만 그리는 위젯
class EffectOverlay(QWidget):
    """
    화면 합성 전용 오버레이. paintEvent 안에서만 그림.
    - spawn_burst(cx, cy): 링/파티클 터짐
    - spawn_confetti(cx, cy): 색종이 파편
    - clear(): 모든 이펙트 제거
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        # 애니메이션 타이머 (고정 프레임 업데이트)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # ~60 FPS

        # 파티클/링 컨테이너
        self._bursts: list[dict] = []
        self._confetti: list[dict] = []

        # 전역 스위치
        self._enabled = True

    # --- 레이아웃/표시 ---
    def resizeEvent(self, e):
        # 부모 크기에 맞추어 꽉 채우기
        if self.parent():
            self.setGeometry(self.parent().rect())
        return super().resizeEvent(e)

    def set_enabled(self, on: bool):
        self._enabled = bool(on)
        if on:
            self.show()
        else:
            self.hide()

    def clear(self):
        self._bursts.clear()
        self._confetti.clear()
        self.update()

    # --- 트리거 API ---
    def spawn_burst(self, cx: int, cy: int, power: float = 1.0, color: QColor | None = None):
        """중앙에서 바깥으로 퍼지는 링 + 몇 개의 파편"""
        if not self._enabled:
            return
        self._bursts.append({
            "x": float(cx), "y": float(cy),
            "t": 0.0,             # 0.0~1.0 정규화 시간
            "power": float(power),
            "color": color or QColor(255, 255, 255),
            # TODO: 필요 시 추가 파라미터(두께, 지속시간 등)
        })
        # 가벼운 파편도 함께 생성
        self._spawn_confetti_cluster(cx, cy, n=12, spread=1.0 * power)

    def spawn_confetti(self, cx: int, cy: int, amount: int = 24, spread: float = 1.0):
        """색종이 파편만 흩뿌리기"""
        if not self._enabled:
            return
        self._spawn_confetti_cluster(cx, cy, n=amount, spread=spread)

    # --- 내부 생성 도우미 ---
    def _spawn_confetti_cluster(self, cx: int, cy: int, n: int, spread: float):
        import random
        for _ in range(max(1, n)):
            ang = random.uniform(0, 6.28318)  # 0~2π
            spd = random.uniform(80, 220) * spread
            vx, vy = spd * float(__import__("math").cos(ang)), spd * float(__import__("math").sin(ang))
            self._confetti.append({
                "pos": QPointF(cx, cy),
                "vel": QPointF(vx, vy),
                "t": 0.0,
                "life": random.uniform(0.6, 1.2),   # 수명(초) 간단 모델
                "size": random.uniform(3.0, 6.0),
                "color": QColor(random.randint(180,255), random.randint(120,255), random.randint(120,255)),
                # TODO: 회전/중력/드래그 파라미터 필요하면 추가
            })

    # --- 업데이트 루프 ---
    def _tick(self):
        dt = 0.016  # 고정 timestep
        changed = False

        # 링(버스트) 업데이트
        if self._bursts:
            next_bursts = []
            for b in self._bursts:
                b["t"] += dt / 0.6  # 0.6s 정도로 사라지게
                if b["t"] < 1.0:
                    next_bursts.append(b)
            if len(next_bursts) != len(self._bursts):
                changed = True
            self._bursts = next_bursts

        # 콘페티 업데이트 (중력/감쇠)
        if self._confetti:
            gravity = 650.0
            drag = 0.98
            next_conf = []
            for c in self._confetti:
                c["t"] += dt
                if c["t"] < c["life"]:
                    v = c["vel"]
                    # 중력 적용
                    v.setY(v.y() + gravity * dt)
                    # 간단 드래그
                    v.setX(v.x() * drag); v.setY(v.y() * drag)
                    c["vel"] = v
                    # 위치 적분
                    p = c["pos"]
                    p.setX(p.x() + v.x() * dt)
                    p.setY(p.y() + v.y() * dt)
                    c["pos"] = p
                    next_conf.append(c)
            if len(next_conf) != len(self._confetti):
                changed = True
            self._confetti = next_conf

        if changed or self._bursts or self._confetti:
            self.update()

    # --- 렌더링 ---
    def paintEvent(self, e):
        if not (self._bursts or self._confetti):
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # 1) 버스트 링
        for b in self._bursts:
            t = max(0.0, min(1.0, b["t"]))
            # 반경/알파 간단 이징
            radius = 30 + (180 * t * b["power"])
            alpha = int(255 * (1.0 - t))
            col = QColor(b["color"])
            col.setAlpha(max(0, alpha))
            pen = QPen(col, max(1.0, 6.0 * (1.0 - t)))
            p.setPen(pen); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(b["x"], b["y"]), radius, radius)

        # 2) 콘페티 파편
        for c in self._confetti:
            life_t = c["t"] / max(0.001, c["life"])
            alpha = int(255 * (1.0 - life_t))
            col = QColor(c["color"]); col.setAlpha(max(0, alpha))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            s = c["size"]
            # 원형 점(필요 시 사각형/회전 도형으로 확장)
            p.drawEllipse(c["pos"], s, s)

        p.end()

# ---- 사용 예시 (트리거는 페이지 로직에서) -------------------------------------
# self.effect_overlay = EffectOverlay(self)
# self.effect_overlay.setGeometry(self.rect())
# self.effect_overlay.raise_()
#
# # rep 완료 직후:
# cx, cy = self.width() // 2, self.height() // 2
# self.effect_overlay.spawn_burst(cx, cy, power=1.0)
# # 또는
# self.effect_overlay.spawn_confetti(cx, cy, amount=30, spread=1.2)
