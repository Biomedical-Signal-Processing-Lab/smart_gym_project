import os
import csv
import time
import numpy as np
from core.evaluators.pose_angles import compute_joint_angles

def test_pose_angles(save_dir: str = "app/core/evaluators"):
    """각도 계산 테스트 + CSV 저장"""

    # 저장 디렉토리 설정
    os.makedirs(save_dir, exist_ok=True)
    filename = f"test_pose_angles_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    file_path = os.path.join(save_dir, filename)

    # 임의의 좌표 및 신뢰도 생성
    kxy = np.random.rand(17, 2) * 1000  # (x, y)
    kcf = np.ones(17)                   # 신뢰도 = 1.0

    # 각도 계산
    angles = compute_joint_angles(kxy, kcf)

    # 결과 콘솔 출력
    print("==== 관절 각도 계산 결과 ====")
    for k, v in angles.items():
        print(f"{k:15s}: {v:.2f}")

    # CSV 파일 저장
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["관절명", "각도(deg)"])
        for k, v in angles.items():
            writer.writerow([k, round(v, 2)])

    print(f"\n✅ CSV 파일 생성 완료: {file_path}")

if __name__ == "__main__":
    test_pose_angles()
