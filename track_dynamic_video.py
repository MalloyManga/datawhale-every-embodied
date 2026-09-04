"""Track people and pets in videos for the CyberPet MVP."""

import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO, settings


PROJECT_DIR = Path(__file__).parent
VIDEO_DIR = PROJECT_DIR / "videos"
RESULT_DIR = PROJECT_DIR / "results"
MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TARGET_CLASSES = {"person", "cat", "dog"}

# Disable background telemetry so offline video processing is deterministic.
settings.update({"sync": False})


def find_input_videos() -> List[Path]:
    """Return all supported videos in the fixed input directory."""
    return sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def track_video(video_path: Path, max_frames: int = None) -> Dict:
    """Detect people and pets, then match centers across frames."""
    RESULT_DIR.mkdir(exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    success, first_frame = capture.read()
    if not success or first_frame is None:
        capture.release()
        raise RuntimeError(f"Cannot read the first frame: {video_path}")
    height, width = first_frame.shape[:2]
    output_video = RESULT_DIR / f"{video_path.stem}_tracking.mp4"
    output_json = RESULT_DIR / f"{video_path.stem}_tracking.json"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    trajectories = defaultdict(list)
    frame_count = 0
    tracked_frames = 0
    errors = []
    next_track_id = 0
    active_tracks = {}
    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            frame = first_frame if frame_count == 0 else capture.read()[1]
            if frame is None:
                break

            try:
                result = model.predict(
                    frame,
                    classes=[0, 15, 16],
                    conf=0.35,
                    verbose=False,
                )[0]
                annotated_frame = result.plot()
                if result.boxes is not None:
                    class_ids = result.boxes.cls.int().cpu().tolist()
                    boxes = result.boxes.xyxy.cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()
                    current_tracks = {}
                    for class_id, box, confidence in zip(class_ids, boxes, confidences):
                        class_name = result.names[class_id]
                        if class_name not in TARGET_CLASSES:
                            continue
                        x1, y1, x2, y2 = box
                        center = {
                            "x": round((x1 + x2) / 2, 2),
                            "y": round((y1 + y2) / 2, 2),
                        }
                        matching_id = _match_track(
                            active_tracks, class_name, center, max_distance=120
                        )
                        if matching_id is None:
                            matching_id = next_track_id
                            next_track_id += 1
                        current_tracks[matching_id] = {
                            "class": class_name,
                            "center": center,
                        }
                        trajectories[str(matching_id)].append(
                            {
                                "frame": frame_count,
                                "class": class_name,
                                "center": center,
                            }
                        )
                    active_tracks = current_tracks
                    tracked_frames += 1
            except Exception as error:
                errors.append({"frame": frame_count, "error": str(error)})
                annotated_frame = frame

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
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
    finally:
        capture.release()
        writer.release()

    tracking_data = {
        "source_video": str(video_path),
        "frame_count": frame_count,
        "max_frames": max_frames,
        "fps": fps,
        "tracked_frames": tracked_frames,
        "target_classes": sorted(TARGET_CLASSES),
        "trajectories": dict(trajectories),
        "errors": errors,
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


def _match_track(active_tracks, class_name: str, center: Dict, max_distance: float):
    """Match a detection to the nearest active track of the same class."""
    candidates = []
    for track_id, track in active_tracks.items():
        if track["class"] != class_name:
            continue
        previous = track["center"]
        distance = ((center["x"] - previous["x"]) ** 2 + (center["y"] - previous["y"]) ** 2) ** 0.5
        if distance <= max_distance:
            candidates.append((distance, track_id))
    return min(candidates)[1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Track dynamic targets in videos")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    videos = find_input_videos()
    if not videos:
        raise SystemExit(f"No video found in: {VIDEO_DIR}")
    for video_path in videos:
        print(f"\nProcessing: {video_path.name}")
        track_video(video_path, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
