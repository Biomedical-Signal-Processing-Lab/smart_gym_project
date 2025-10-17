# app/core/evaluators/__init__.py
from .base import EvalResult, ExerciseEvaluator
from .lower_body import LowerBodyEvaluator
from .upper_body import UpperBodyEvaluator
from .core_full import CoreFullEvaluator

# 라벨 정규화 (대소문자/언더스코어 차이를 흡수)
def _normalize_label(label: str) -> str:
    if not label:
        return "idle"
    l = label.strip()
    # 외부에서 오는 혼합 케이스 정리
    if l in ("Side_lateral_raise", "side_lateral_raise"):
        return "side_lateral_raise"
    if l in ("Dumbbell_Row", "dumbbell_row"):
        return "dumbbell_row"
    return l

def get_evaluator_by_label(label: str) -> ExerciseEvaluator:
    """exercise_page의 meta['label']와 1:1 매핑"""
    lab = _normalize_label(label)
    # 🦵 Lower body
    if label in {"squat", "leg_raise"}:
        return LowerBodyEvaluator(label=lab)
    # 💪 Upper body
    if label in {"pushup", "shoulder_press", "Side_lateral_raise", "Dumbbell_Row"}:
        return UpperBodyEvaluator(label=lab)
    # 🧘 Core / Full body
    if label in {"burpee"}:
        return CoreFullEvaluator(label=lab)
    # 기본값(idle 등)
    return None  # idle 등은 None 반환 → ExercisePage에서 안내문 표시
