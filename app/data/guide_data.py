from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class Exercise:
    key: str
    title: str
    category: str
    sets_reps: str
    description: str
    goal_muscles: str
    recommend: str
    steps: List[str]
    tips: List[str]

def _to_exercise_list(raw_list: List[Dict[str, Any]]) -> List[Exercise]:
    return [Exercise(**d) for d in raw_list]

def list_all() -> List[Exercise]:
    return _to_exercise_list(EXERCISES_DATA)

EXERCISES_DATA: List[Dict[str, Any]] = [
    {
        "key": "squat",
        "title": "스쿼트",
        "category": "하체",
        "sets_reps": "3세트 x 15회",
        "description": "하체 근력 강화를 위한 기본 운동으로, 다리와 엉덩이 근육을 집중적으로 발달시킵니다.",
        "goal_muscles": "대퇴사두근, 둔근",
        "recommend": "3세트 x 15회",
        "steps": [
            "발을 어깨 너비로 벌리고 서세요",
            "무릎이 발끝을 넘지 않도록 주의하며 앉으세요",
            "허벅지가 바닥과 평행이 될 때까지 내려가세요",
            "발뒤꿈치로 밀어내며 일어서세요",
        ],
        "tips": [
            "등을 항상 곧게 유지하세요",
            "무릎은 발끝과 같은 방향으로 향하게 하세요",
            "호흡을 자연스럽게 유지",
        ],
    },
    {
        "key": "shoulder_press",
        "title": "숄더프레스",
        "category": "어깨",
        "sets_reps": "3세트 x 12회",
        "description": "어깨 전면과 측면 근육을 강화하는 대표적인 프리웨이트 밀기 동작입니다.",
        "goal_muscles": "전면삼각근, 측면삼각근, 상완삼두근",
        "recommend": "3세트 x 12회",
        "steps": [
            "덤벨 또는 바벨을 어깨 높이에서 잡습니다",
            "팔꿈치를 살짝 앞으로 향하게 유지합니다",
            "호흡을 내쉬며 팔을 위로 밀어 올립니다",
            "팔꿈치를 완전히 잠그기 전까지만 올리고 천천히 내립니다",
        ],
        "tips": [
            "허리가 과도하게 꺾이지 않도록 복부에 힘을 줍니다",
            "손목이 꺾이지 않게 수직으로 유지합니다",
            "덤벨을 내릴 때 어깨 근육의 긴장을 유지하세요",
        ],
    },
    {
        "key": "push_up",
        "title": "푸쉬업",
        "category": "가슴",
        "sets_reps": "3세트 x 15회",
        "description": "상체 전반 근력을 강화하는 대표적인 맨몸 밀기 운동입니다.",
        "goal_muscles": "대흉근, 삼두근, 전면삼각근",
        "recommend": "3세트 x 15회",
        "steps": [
            "어깨 너비보다 약간 넓게 손을 짚고 플랭크 자세를 잡습니다",
            "팔꿈치를 굽혀 몸을 천천히 바닥으로 낮춥니다",
            "가슴이 거의 바닥에 닿기 직전까지 내려갑니다",
            "손바닥으로 밀며 다시 시작 자세로 올라옵니다",
        ],
        "tips": [
            "허리가 꺾이거나 엉덩이가 들리지 않게 몸통을 일직선으로 유지합니다",
            "팔꿈치는 몸통에서 약 45도 각도를 유지합니다",
            "근육의 수축을 느끼며 반동 없이 천천히 수행합니다",
        ],
    },
    {
        "key": "leg_raise",
        "title": "레그레이즈",
        "category": "복부",
        "sets_reps": "3세트 x 12회",
        "description": "하복부를 집중적으로 자극하는 맨몸 코어 운동입니다.",
        "goal_muscles": "하복부, 고관절 굴곡근",
        "recommend": "3세트 x 12회",
        "steps": [
            "바닥에 누운 상태에서 다리를 곧게 펴고 손은 옆에 둡니다",
            "호흡을 내쉬며 다리를 45도 이상 천천히 들어 올립니다",
            "허리가 뜨지 않도록 복부에 힘을 유지합니다",
            "다리를 바닥에 완전히 닿지 않도록 10cm 위에서 멈춥니다",
        ],
        "tips": [
            "허리가 뜬다면 손을 엉덩이 아래에 받쳐도 좋습니다",
            "반동 없이 천천히 올리고 천천히 내립니다",
            "복부 긴장을 유지하며 호흡을 멈추지 않습니다",
        ],
    },
    {
        "key": "bent_over_dumbbell_row",
        "title": "벤트오버 덤벨로우",
        "category": "등",
        "sets_reps": "3세트 x 12회",
        "description": "허리 각도를 고정한 상태에서 등 근육을 당기는 프리웨이트 운동입니다.",
        "goal_muscles": "광배근, 능형근, 척추기립근",
        "recommend": "3세트 x 12회",
        "steps": [
            "덤벨을 들고 무릎을 살짝 굽힌 자세에서 상체를 45도 숙입니다",
            "허리를 곧게 펴고 시선은 아래를 향합니다",
            "숨을 들이마시고 덤벨을 복부 쪽으로 끌어당깁니다",
            "광배근의 수축을 느끼며 천천히 덤벨을 내립니다",
        ],
        "tips": [
            "허리가 말리지 않도록 척추 중립을 유지합니다",
            "덤벨은 팔이 아니라 등으로 당긴다는 느낌으로 수행합니다",
            "최대 수축 지점에서 1초 정도 정지하면 자극이 증가합니다",
        ],
    },
    {
    "key": "rest",
    "title": "휴식",
    "category": "휴식",
    "sets_reps": "-",
    "description": "운동 사이의 회복 시간입니다. 호흡을 정리하고 다음 운동을 준비하세요.",
    "goal_muscles": "-",
    "recommend": "60초 휴식",
    "steps": [
        "호흡을 천천히 정리합니다",
        "필요하다면 가볍게 스트레칭합니다",
        "심박수가 내려가는 것을 느끼며 몸의 긴장을 풀어줍니다",
    ],
    "tips": [
        "휴식은 다음 운동의 퍼포먼스를 위해 매우 중요합니다",
        "너무 길게 쉬면 몸의 긴장이 풀리므로 1분 내외로 제한합니다",
    ],
},
]