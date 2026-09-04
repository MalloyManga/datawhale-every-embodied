"""Track people and pets in videos for the CyberPet MVP."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).parent
VIDEO_DIR = PROJECT_DIR / "videos"
RESULT_DIR = PROJECT_DIR / "results"
MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TARGET_CLASSES = {"person", "cat", "dog"}


def find_input_videos() -> List[Path]:
    """Return all supported videos in the fixed input directory."""
    return sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def track_video(video_path: Path) -> Dict:
    """Track people and pets and save an annotated video and trajectory data."""
    RESULT_DIR.mkdir(exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = RESULT_DIR / f"{video_path.stem}_tracking.mp4"
    output_json = RESULT_DIR / f"{video_path.stem}_tracking.json"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    trajectories = defaultdict(list)
    frame_count = 0
    tracked_frames = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.track(
                frame,
                persist=True,
                classes=[0, 15, 16],
                conf=0.35,
                verbose=False,
            )[0]
            annotated_frame = result.plot()
            frame_detections = []
            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                tracked_frames += 1
                for track_id, class_id, box, confidence in zip(
                    track_ids, class_ids, boxes, confidences
                ):
                    class_name = result.names[class_id]
                    if class_name not in TARGET_CLASSES:
                        continue
                    x1, y1, x2, y2 = box
                    center = {
                        "x": round((x1 + x2) / 2, 2),
                        "y": round((y1 + y2) / 2, 2),
                    }
                    detection = {
                        "track_id": track_id,
                        "class": class_name,
                        "confidence": round(float(confidence), 4),
                        "center": center,
                    }
                    frame_detections.append(detection)
                    trajectories[str(track_id)].append(
                        {"frame": frame_count, "class": class_name, "center": center}
                    )

            cv2.putText(
                annotated_frame,
                f"CyberPet tracking | frame: {frame_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            writer.write(annotated_frame)
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    tracking_data = {
        "source_video": str(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "tracked_frames": tracked_frames,
        "target_classes": sorted(TARGET_CLASSES),
        "trajectories": dict(trajectories),
        "limitations": [
            "Tracking is based on a pretrained model and has not been tested on a robot camera.",
            "An identity can be lost during occlusion or when the person leaves the frame.",
            "The output is image-plane motion, not a metric 3D trajectory.",
        ],
    }
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(tracking_data, file, indent=2, ensure_ascii=False)

    print(f"Tracked video saved to: {output_video}")
    print(f"Tracking data saved to: {output_json}")
    print(f"Frames: {frame_count}; tracked frames: {tracked_frames}")
    return tracking_data


def main() -> None:
    videos = find_input_videos()
    if not videos:
        raise SystemExit(f"No video found in: {VIDEO_DIR}")
    for video_path in videos:
        print(f"\nProcessing: {video_path.name}")
        track_video(video_path)


if __name__ == "__main__":
    main()
