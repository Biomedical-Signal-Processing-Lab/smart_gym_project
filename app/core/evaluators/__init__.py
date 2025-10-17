# app/core/evaluators/__init__.py
from .lower_body import LowerBodyEvaluator
from .upper_body import UpperBodyEvaluator   # ✅ [ADD]
from .base import EvalResult, ExerciseEvaluator

__all__ = [
    "get_evaluator_by_label",
    "EvalResult",
    "ExerciseEvaluator",
    "LowerBodyEvaluator",
    "UpperBodyEvaluator",   # ✅ [ADD]
]

# ===== 싱글톤 보관 (Evaluator 인스턴스 1회 생성) =====
_EVAL_SINGLETONS = {
    # 하체
    "squat": LowerBodyEvaluator("squat"),
    "leg_raise": LowerBodyEvaluator("leg_raise"),

    # 상체
    "pushup": UpperBodyEvaluator("pushup"),
    "shoulder_press": UpperBodyEvaluator("shoulder_press"),
    "side_lateral_raise": UpperBodyEvaluator("side_lateral_raise"),
    "dumbbell_row": UpperBodyEvaluator("dumbbell_row"),
}

# ===== 라벨 별칭(한글/대소문자/공백 호환) =====
_ALIAS = {
    # 하체
    "squat": "squat",
    "스쿼트": "squat",
    "legraise": "leg_raise",
    "leg_raise": "leg_raise",
    "레그레이즈": "leg_raise",
    "레그 레이즈": "leg_raise",

    # 상체
    "pushup": "pushup",
    "푸쉬업": "pushup",
    "shoulderpress": "shoulder_press",
    "shoulder_press": "shoulder_press",
    "숄더프레스": "shoulder_press",
    "side_lateral_raise": "side_lateral_raise",
    "side_lateral": "side_lateral_raise",
    "side_lateral-raise": "side_lateral_raise",
    "사이드레터럴레이즈": "side_lateral_raise",
    "dumbbellrow": "dumbbell_row",
    "dumbbell_row": "dumbbell_row",
    "덤벨로우": "dumbbell_row",
}

# ===== 공용 팩토리 함수 =====
def get_evaluator_by_label(label: str) -> ExerciseEvaluator | None:
    """운동 이름(라벨)에 맞는 평가자 반환"""
    if not label:
        return None

    key = label.strip().lower().replace("-", "_").replace(" ", "")
    key = _ALIAS.get(key, key)
    return _EVAL_SINGLETONS.get(key)
