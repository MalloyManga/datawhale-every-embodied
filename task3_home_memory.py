"""Build a simple multi-room memory from object-detection JSON files."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


MAP_DIR = Path(__file__).parent / "maps"


def load_environment_maps(map_dir: Path) -> List[Dict]:
    """Load all saved room observations from a directory."""
    observations = []
    for map_path in sorted(map_dir.glob("*.json")):
        if map_path.name == "home_environment_memory.json":
            continue
        try:
            with map_path.open("r", encoding="utf-8") as file:
                observation = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: skipped invalid environment map: {map_path.name}")
            continue
        observation["source_file"] = map_path.name
        observations.append(observation)
    return observations


def build_home_memory(observations: List[Dict]) -> Dict:
    """Aggregate room observations into a compact semantic memory."""
    rooms = {}
    object_counts = Counter()

    for observation in observations:
        image_path = Path(observation.get("image_path", "unknown_room"))
        room_name = image_path.stem
        detected_objects = observation.get("detected_objects", {})
        object_counts.update(detected_objects)
        rooms[room_name] = {
            "source_image": observation.get("image_path"),
            "total_objects": observation.get("total_objects", 0),
            "detected_objects": detected_objects,
            "object_positions": observation.get("object_positions", {}),
            "confidence_scores": observation.get("confidence_scores", {}),
        }

    return {
        "description": "CyberPet multi-room environment memory",
        "observation_count": len(observations),
        "rooms": rooms,
        "known_objects": dict(object_counts),
        "limitations": [
            "Object positions are normalized 2D image coordinates.",
            "This memory does not yet contain metric depth or a common SLAM coordinate frame.",
            "Repeated observations of the same room currently replace the previous room entry.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CyberPet home memory from detection maps")
    parser.add_argument("--input", type=Path, default=MAP_DIR, help="directory containing environment maps")
    parser.add_argument("--output", type=Path, default=MAP_DIR / "home_environment_memory.json")
    args = parser.parse_args()

    observations = load_environment_maps(args.input)
    if not observations:
        raise SystemExit(f"No environment maps found in: {args.input}")

    memory = build_home_memory(observations)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(memory, file, indent=2, ensure_ascii=False)

    print(f"Loaded {len(observations)} room observations")
    print(f"Known objects: {memory['known_objects']}")
    print(f"Home memory saved to: {args.output}")


if __name__ == "__main__":
    main()
