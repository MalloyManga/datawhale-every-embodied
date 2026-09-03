"""Build a simple multi-room memory from object-detection JSON files."""

import argparse
import json
from collections import Counter
from collections import deque
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


def add_route_memory(memory: Dict, route: List[str]) -> None:
    """Store a user-confirmed sequence of connected rooms."""
    if len(route) < 2:
        raise ValueError("A route must contain at least two room names")

    known_rooms = set(memory["rooms"])
    unknown_rooms = [room for room in route if room not in known_rooms]
    if unknown_rooms:
        raise ValueError(f"Unknown rooms in route: {', '.join(unknown_rooms)}")

    graph = memory.setdefault("route_memory", {})
    for current_room, next_room in zip(route, route[1:]):
        graph.setdefault(current_room, [])
        graph.setdefault(next_room, [])
        if next_room not in graph[current_room]:
            graph[current_room].append(next_room)
        if current_room not in graph[next_room]:
            graph[next_room].append(current_room)


def find_route(memory: Dict, start: str, goal: str) -> List[str]:
    """Find a shortest remembered route between two rooms."""
    graph = memory.get("route_memory", {})
    if start == goal:
        return [start]
    if start not in graph or goal not in graph:
        return []

    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        current_room = path[-1]
        for next_room in graph.get(current_room, []):
            if next_room in visited:
                continue
            next_path = path + [next_room]
            if next_room == goal:
                return next_path
            visited.add(next_room)
            queue.append(next_path)
    return []


def simulate_robot_action(action: str, room_name: str, object_name: str) -> Dict:
    """Simulate a robot-driver call and return the planned action.

    In a real robot this is where commands would be sent to the base, speaker,
    or actuator driver. This project only records and prints the command.
    """
    action_result = {
        "room": room_name,
        "object": object_name,
        "action": action,
        "mode": "simulation",
        "message": "",
    }
    if action == "approach_and_greet":
        action_result["message"] = f"发现{object_name}，前往附近并进行问候"
    elif action == "step_back_and_warn":
        action_result["message"] = f"发现{object_name}，后退一步并提醒注意"
    else:
        action_result["message"] = f"发现{object_name}，保持观察"

    print(f"Simulated action: {action_result['message']}")
    return action_result


def build_behavior_feedback(memory: Dict) -> List[Dict]:
    """Generate behavior feedback from remembered room objects."""
    feedback = []
    for room_name, room in memory["rooms"].items():
        for object_name in room["detected_objects"]:
            if object_name == "bed":
                action = "approach_and_greet"
            elif object_name in {"toilet", "sink"}:
                action = "step_back_and_warn"
            else:
                action = "observe"
            feedback.append(simulate_robot_action(action, room_name, object_name))
    return feedback


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CyberPet home memory from detection maps")
    parser.add_argument("--input", type=Path, default=MAP_DIR, help="directory containing environment maps")
    parser.add_argument("--output", type=Path, default=MAP_DIR / "home_environment_memory.json")
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="remember a connected room sequence, e.g. sample_room,sample_bathroom",
    )
    parser.add_argument("--from-room", help="start room for a remembered route query")
    parser.add_argument("--to-room", help="destination room for a remembered route query")
    parser.add_argument(
        "--simulate-behavior",
        action="store_true",
        help="simulate behavior feedback from remembered objects",
    )
    args = parser.parse_args()

    observations = load_environment_maps(args.input)
    if not observations:
        raise SystemExit(f"No environment maps found in: {args.input}")

    memory = build_home_memory(observations)
    for route_text in args.route:
        route = [room.strip() for room in route_text.split(",") if room.strip()]
        add_route_memory(memory, route)

    if args.from_room and args.to_room:
        route = find_route(memory, args.from_room, args.to_room)
        if route:
            print(f"Remembered route: {' -> '.join(route)}")
        else:
            print(f"No remembered route: {args.from_room} -> {args.to_room}")

    if args.simulate_behavior:
        memory["behavior_feedback"] = build_behavior_feedback(memory)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(memory, file, indent=2, ensure_ascii=False)

    print(f"Loaded {len(observations)} room observations")
    print(f"Known objects: {memory['known_objects']}")
    print(f"Home memory saved to: {args.output}")


if __name__ == "__main__":
    main()
