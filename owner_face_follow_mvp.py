"""CyberPet software MVP: owner recognition, tracking, and simulated actions."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).parent
OWNER_DIR = PROJECT_DIR / "owner_images"
VIDEO_DIR = PROJECT_DIR / "videos"
RESULT_DIR = PROJECT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "models"
YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
COSINE_THRESHOLD = 0.363


def require_models() -> None:
    missing = [str(path) for path in (YUNET_MODEL, SFACE_MODEL) if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing face models:\n"
            + "\n".join(missing)
            + "\nDownload them according to the Task 5 README instructions."
        )


def create_models() -> Tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    require_models()
    detector = cv2.FaceDetectorYN_create(
        str(YUNET_MODEL), "", (320, 320), 0.85, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "")
    return detector, recognizer


def detect_faces(detector, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    return faces if faces is not None else np.empty((0, 15), dtype=np.float32)


def extract_feature(detector, recognizer, image: np.ndarray) -> Optional[np.ndarray]:
    faces = detect_faces(detector, image)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda item: float(item[14]))
    aligned = recognizer.alignCrop(image, face)
    return recognizer.feature(aligned)


def register_owner(detector, recognizer) -> np.ndarray:
    owner_images = sorted(
        path
        for path in OWNER_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not owner_images:
        raise SystemExit(f"Put an owner photo in: {OWNER_DIR}")

    features = []
    for image_path in owner_images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: cannot read owner image: {image_path.name}")
            continue
        feature = extract_feature(detector, recognizer, image)
        if feature is None:
            print(f"Warning: no face found in owner image: {image_path.name}")
            continue
        features.append(feature)

    if not features:
        raise SystemExit("No usable face found in owner_images/")
    return np.mean(np.stack(features), axis=0).astype(np.float32)


def choose_action(face, frame_width: int, face_height: int) -> str:
    """Convert visual position into a simulated robot command."""
    x, _, width, _ = [float(value) for value in face[:4]]
    center_x = (x + width / 2) / frame_width
    relative_size = face_height / max(frame_width, 1)
    if relative_size > 0.65:
        return "stop_and_keep_distance"
    if center_x < 0.35:
        return "turn_left_and_follow"
    if center_x > 0.65:
        return "turn_right_and_follow"
    return "move_forward_and_follow"


def simulated_robot_driver(action: str) -> str:
    """Placeholder for a real base, speaker, or actuator driver."""
    messages = {
        "stop_and_keep_distance": "主人距离过近，模拟停止并保持安全距离",
        "turn_left_and_follow": "主人在左侧，模拟向左调整并跟随",
        "turn_right_and_follow": "主人在右侧，模拟向右调整并跟随",
        "move_forward_and_follow": "主人在前方，模拟向前跟随",
        "owner_lost": "未识别到主人，模拟停止并等待重新识别",
    }
    return messages.get(action, "模拟保持观察")


def process_video(video_path: Path, owner_feature: np.ndarray) -> Dict:
    detector, recognizer = create_models()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    success, first_frame = capture.read()
    if not success or first_frame is None:
        capture.release()
        raise SystemExit(f"Cannot read video: {video_path}")
    height, width = first_frame.shape[:2]
    RESULT_DIR.mkdir(exist_ok=True)
    output_video = RESULT_DIR / f"{video_path.stem}_owner_follow_mvp.mp4"
    output_json = RESULT_DIR / f"{video_path.stem}_owner_follow_mvp.json"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = 0
    recognized_frames = 0
    action_counts: Dict[str, int] = {}
    timeline: List[Dict] = []
    try:
        frame = first_frame
        while frame is not None:
            faces = detect_faces(detector, frame)
            best_match = None
            best_score = -1.0
            for face in faces:
                aligned = recognizer.alignCrop(frame, face)
                feature = recognizer.feature(aligned)
                score = float(
                    recognizer.match(
                        owner_feature, feature, cv2.FaceRecognizerSF_FR_COSINE
                    )
                )
                if score > best_score:
                    best_score = score
                    best_match = face

            if best_match is not None and best_score >= COSINE_THRESHOLD:
                action = choose_action(best_match, width, height)
                recognized_frames += 1
                face_box = [round(float(value), 2) for value in best_match[:4]]
            else:
                action = "owner_lost"
                face_box = None

            action_counts[action] = action_counts.get(action, 0) + 1
            message = simulated_robot_driver(action)
            timeline.append(
                {
                    "frame": frame_count,
                    "owner_score": round(max(best_score, 0.0), 4),
                    "face_box": face_box,
                    "action": action,
                    "driver_mode": "simulation",
                }
            )

            annotated = frame.copy()
            if face_box is not None:
                x, y, box_width, box_height = [int(value) for value in face_box]
                cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"owner score: {max(best_score, 0.0):.2f} | {action}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if action != "owner_lost" else (0, 0, 255),
                2,
            )
            writer.write(annotated)
            frame_count += 1
            success, frame = capture.read()
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
    finally:
        capture.release()
        writer.release()

    result = {
        "source_video": str(video_path),
        "frame_count": frame_count,
        "recognized_owner_frames": recognized_frames,
        "recognition_rate": round(recognized_frames / max(frame_count, 1), 4),
        "cosine_threshold": COSINE_THRESHOLD,
        "action_counts": action_counts,
        "timeline": timeline,
        "driver_mode": "simulation",
        "limitations": [
            "Owner photos must be used with consent; this prototype does not publish face data.",
            "The driver only simulates movement and does not control a real robot.",
            "Recognition may fail under occlusion, poor lighting, or an unseen face angle.",
        ],
    }
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MVP video saved to: {output_video}")
    print(f"MVP data saved to: {output_json}")
    print(f"Frames: {frame_count}; owner recognized: {recognized_frames}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CyberPet owner recognition and follow MVP")
    parser.add_argument("--video", type=Path, help="optional video path; defaults to videos/ first video")
    args = parser.parse_args()
    detector, recognizer = create_models()
    owner_feature = register_owner(detector, recognizer)
    video_path = args.video
    if video_path is None:
        candidates = sorted(
            path
            for path in VIDEO_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not candidates:
            raise SystemExit(f"Put a video in: {VIDEO_DIR}")
        video_path = candidates[0]
    process_video(video_path, owner_feature)


if __name__ == "__main__":
    main()
