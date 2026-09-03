"""
赛博宠物 - 家庭环境物体检测系统
CyberPet Environment Object Detection System

功能：检测和识别家庭环境中的物体，构建环境地图
- 使用YOLOv8进行快速物体检测
- 识别家具和家庭物体（床、椅子、桌子、台灯、窗户等）
- 输出物体位置和类别信息
- 为赛博宠物的环境认知提供基础

Author: CyberPet Team
Date: 2026-09-02
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor


class CyberPetEnvironmentDetector:
    """
    赛博宠物环境检测器

    功能：
    1. 加载YOLOv8模型
    2. 处理图像和视频
    3. 识别家庭物体
    4. 建立环境地图和物体数据库
    """

    def __init__(
        self,
        model_size: str = "n",
        use_depth: bool = True,
        depth_model_name: str = "Intel/dpt-hybrid-midas",
    ):
        """
        初始化检测器

        Args:
            model_size: YOLOv8模型大小 ('n'=nano, 's'=small, 'm'=medium, 'l'=large)
                       - 'n': 最快，精度较低，适合实时推理
                       - 'm': 平衡速度和精度
                       - 'l': 精度高，速度慢
        """
        print(f"🤖 正在加载YOLOv8-{model_size}模型...")
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ 模型加载完成 (运行设备: {self.device})")

        self.depth_enabled = False
        self.depth_processor = None
        self.depth_model = None
        if use_depth:
            try:
                print(f"📏 正在加载单目深度模型: {depth_model_name}")
                self.depth_processor = DPTImageProcessor.from_pretrained(depth_model_name)
                self.depth_model = DPTForDepthEstimation.from_pretrained(
                    depth_model_name
                ).to(self.device)
                self.depth_model.eval()
                self.depth_model_name = depth_model_name
                self.depth_enabled = True
                print("✅ 单目深度模型加载完成")
            except Exception as error:
                print(f"⚠️ 单目深度模型加载失败，将只运行二维检测: {error}")

        # 家庭物体的关键词映射 - 赛博宠物需要识别的物体
        self.home_objects_mapping = {
            # 床和家具
            "bed": ["bed", "床"],
            "sofa": ["sofa", "couch", "沙发"],
            "chair": ["chair", "椅子"],
            "table": ["table", "dining table", "桌子", "餐桌"],
            "desk": ["desk", "书桌"],
            # 照明
            "lamp": ["lamp", "台灯", "灯"],
            "light": ["light", "ceiling light", "灯"],
            # 窗户和门
            "window": ["window", "窗户"],
            "door": ["door", "门"],
            # 其他家电
            "tv": ["tv", "monitor", "电视", "显示器"],
            "refrigerator": ["refrigerator", "fridge", "冰箱"],
            "microwave": ["microwave", "微波炉"],
            "sink": ["sink", "水槽"],
            # 人和宠物（赛博宠物关心的）
            "person": ["person", "people", "人", "人类"],
            "cat": ["cat", "猫"],
            "dog": ["dog", "狗"],
        }

        self.environment_map = {
            "timestamp": None,
            "total_objects": 0,
            "detected_objects": {},
            "object_positions": {},
            "confidence_scores": {},
        }

    def detect_image(
        self, image_path: str, conf_threshold: float = 0.5, output_name: str = None
    ) -> Dict:
        """
        检测单张图像中的物体

        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值（0-1），只保留置信度高于此值的检测

        Returns:
            包含检测结果的字典
        """
        print(f"\n📸 正在处理图像: {image_path}")

        # 读取图像
        if not Path(image_path).exists():
            print(f"❌ 错误：图像文件不存在 - {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 错误：无法读取图像")
            return None

        height, width = image.shape[:2]
        print(f"   图像大小: {width}x{height}")

        # 运行检测
        results = self.model.predict(image, conf=conf_threshold, verbose=False)

        depth_map = self._estimate_depth(image) if self.depth_enabled else None

        # 处理检测结果
        detections = self._process_detections(results[0], image.shape, depth_map)

        # 保存结果
        self.environment_map = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "image_size": (width, height),
            "total_objects": len(detections["objects"]),
            "objects": detections["objects"],
            "detected_objects": detections["summary"],
            "object_positions": detections["positions"],
            "confidence_scores": detections["confidences"],
            "depth": {
                "enabled": self.depth_enabled,
                "model": getattr(self, "depth_model_name", None),
                "unit": "relative_depth",
            },
        }

        # 打印结果摘要
        self._print_detection_summary(detections)

        # 保存可视化结果
        output_path = self._save_visualization(image, results[0], output_name)
        print(f"✅ 检测完成，结果已保存到: {output_path}")

        return self.environment_map

    def detect_images(self, image_paths: List[str], conf_threshold: float = 0.5) -> List[Dict]:
        """批量检测多张图片，并为每张图片保存独立结果。"""
        results = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for image_path in image_paths:
            path = Path(image_path)
            if path.suffix.lower() not in valid_extensions:
                print(f"⚠️ 跳过不支持的图片格式: {image_path}")
                continue

            result = self.detect_image(
                str(path), conf_threshold=conf_threshold, output_name=path.stem
            )
            if result is not None:
                map_path = Path(__file__).parent / "maps" / f"{path.stem}_environment_map.json"
                self.save_environment_map(map_path)
                results.append(result)

        print(f"\n✅ 批量检测完成，共处理 {len(results)} 张图片")
        return results

    def detect_video(self, video_source, max_frames: int = 100):
        """
        检测视频中的物体（可以是视频文件或摄像头）

        Args:
            video_source: 视频文件路径或摄像头索引 (0表示默认摄像头)
            max_frames: 最多处理的帧数

        Returns:
            检测结果列表
        """
        print(f"\n🎥 正在处理视频源: {video_source}")

        cap = cv2.VideoCapture(
            video_source if isinstance(video_source, int) else str(video_source)
        )

        if not cap.isOpened():
            print("❌ 错误：无法打开视频源")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"   FPS: {fps}")

        all_detections = []
        frame_count = 0

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 每5帧检测一次（加快速度）
            if frame_count % 5 == 0:
                results = self.model.predict(frame, conf=0.5, verbose=False)
                detections = self._process_detections(results[0], frame.shape)
                all_detections.append({"frame": frame_count, "detections": detections})

                # 实时显示
                annotated_frame = results[0].plot()
                cv2.imshow("CyberPet Detection", annotated_frame)

                if frame_count % 50 == 0:
                    print(f"   已处理 {frame_count} 帧...")

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"✅ 视频处理完成，共检测 {len(all_detections)} 个关键帧")
        return all_detections

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate relative depth from a single RGB image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        inputs = self.depth_processor(images=image_pil, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        depth = prediction.squeeze().cpu().numpy()
        depth_min, depth_max = float(depth.min()), float(depth.max())
        if depth_max - depth_min < 1e-8:
            return np.zeros_like(depth, dtype=np.float32)
        return ((depth - depth_min) / (depth_max - depth_min)).astype(np.float32)

    def _process_detections(
        self, results, image_shape, depth_map: np.ndarray = None
    ) -> Dict:
        """
        处理YOLO检测结果，提取有用的信息

        Returns:
            处理后的检测信息
        """
        detections = {"objects": [], "summary": {}, "positions": {}, "confidences": {}}

        height, width = image_shape[:2]

        for box in results.boxes:
            class_id = int(box.cls)
            class_name = results.names[class_id]
            confidence = float(box.conf)

            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detection_info = {
                "class": class_name,
                "confidence": confidence,
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "center": (int(center_x), int(center_y)),
                },
                "position": self._normalize_position(center_x, center_y, width, height),
            }
            if depth_map is not None:
                center_depth = depth_map[
                    min(int(center_y), height - 1), min(int(center_x), width - 1)
                ]
                detection_info["relative_depth"] = float(round(float(center_depth), 4))

            detections["objects"].append(detection_info)

            # 统计物体
            if class_name not in detections["summary"]:
                detections["summary"][class_name] = 0
                detections["positions"][class_name] = []
                detections["confidences"][class_name] = []

            detections["summary"][class_name] += 1
            detections["positions"][class_name].append(detection_info["position"])
            detections["confidences"][class_name].append(confidence)

        return detections

    def _normalize_position(self, x: float, y: float, width: int, height: int) -> Dict:
        """
        将像素坐标转换为归一化位置信息
        用于赛博宠物理解物体在环境中的相对位置

        Returns:
            {'quadrant': '象限', 'normalized_x': 0-1, 'normalized_y': 0-1}
        """
        norm_x = x / width
        norm_y = y / height

        # 确定象限
        if norm_x < 0.33:
            h_pos = "left"
        elif norm_x < 0.67:
            h_pos = "center"
        else:
            h_pos = "right"

        if norm_y < 0.33:
            v_pos = "top"
        elif norm_y < 0.67:
            v_pos = "middle"
        else:
            v_pos = "bottom"

        return {
            "quadrant": f"{h_pos}-{v_pos}",
            "normalized_x": float(round(norm_x, 2)),
            "normalized_y": float(round(norm_y, 2)),
        }

    def _print_detection_summary(self, detections: Dict):
        """打印检测摘要"""
        print(f"\n📊 检测结果摘要:")
        print(f"   总物体数: {len(detections['objects'])}")

        if detections["summary"]:
            print(f"   物体分类统计:")
            for obj_class, count in detections["summary"].items():
                avg_conf = np.mean(detections["confidences"][obj_class])
                print(f"      - {obj_class}: {count}个 (平均置信度: {avg_conf:.2%})")

                # 显示物体位置
                positions = detections["positions"][obj_class]
                if positions:
                    quadrant = (
                        positions[0]["quadrant"]
                        if isinstance(positions[0], dict)
                        else "unknown"
                    )
                    print(f"        位置: {quadrant}")
        else:
            print("   ⚠️  未检测到任何物体")

    def _save_visualization(self, image, results, output_name: str = None):
        """
        保存可视化结果
        """
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        # 绘制检测结果
        annotated_image = results.plot()

        # 保存图像
        if output_name:
            output_path = output_dir / f"{output_name}_detection.jpg"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"detection_{timestamp}.jpg"

        cv2.imwrite(str(output_path), annotated_image)

        return output_path

    def save_environment_map(self, output_path: str = None):
        """
        保存环境地图为JSON文件
        赛博宠物可以读取这个文件来了解家庭环境
        """
        if output_path is None:
            output_dir = Path(__file__).parent / "maps"
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"environment_map_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.environment_map, f, indent=2, ensure_ascii=False)

        print(f"💾 环境地图已保存: {output_path}")
        return output_path

    def get_home_layout(self) -> Dict:
        """
        获取家庭布局信息
        返回房间中各个区域和物体的分布

        赛博宠物用这个信息导航和规划行为
        """
        layout = {
            "zones": {"left": [], "center": [], "right": []},
            "heights": {"top": [], "middle": [], "bottom": []},
        }

        for obj_class, positions in self.environment_map["object_positions"].items():
            for pos in positions:
                if isinstance(pos, dict):
                    zone = pos["quadrant"].split("-")[0]  # left, center, right
                    height = pos["quadrant"].split("-")[1]  # top, middle, bottom

                    layout["zones"][zone].append(obj_class)
                    layout["heights"][height].append(obj_class)

        return layout


def main():
    """
    主函数 - 演示如何使用赛博宠物环境检测系统
    """
    print("=" * 60)
    print("🤖 赛博宠物 - 环境物体检测系统")
    print("CyberPet Environment Object Detection System")
    print("=" * 60)

    # 初始化检测器
    detector = CyberPetEnvironmentDetector(model_size="n")  # 使用nano模型，最快

    # ========== 使用示例 ==========

    # 示例1：检测本地图像
    print("\n" + "=" * 60)
    print("示例1: 检测本地图像")
    print("=" * 60)

    image_dir = Path(__file__).parent / "images"
    image_paths = sorted(
        str(path)
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ) if image_dir.exists() else []

    if not image_paths:
        print(f"ℹ️  提示: images/ 目录中没有找到图片: {image_dir}")
        print("   请将家庭环境图片放入 images/ 后重新运行")
    else:
        detector.detect_images(image_paths, conf_threshold=0.5)

    # 示例2：检测摄像头视频（实时检测）
    print("\n" + "=" * 60)
    print("示例2: 实时摄像头检测")
    print("=" * 60)
    print("ℹ️  如果想使用摄像头实时检测，可以这样调用：")
    print("   detector.detect_video(0, max_frames=100)")
    print("   按 'q' 键退出")
    print("   (当前演示模式下未启用，取消注释下面的代码来启用)")

    # 取消下面的注释来启用实时摄像头检测
    # detector.detect_video(0, max_frames=100)

    # 示例3：检测视频文件
    print("\n" + "=" * 60)
    print("示例3: 检测视频文件")
    print("=" * 60)
    video_path = "home_video.mp4"
    if Path(video_path).exists():
        detector.detect_video(video_path, max_frames=100)
    else:
        print(f"ℹ️  提示: 没有找到 {video_path}")
        print("   你可以提供一个家庭环境的视频来进行检测")

    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
