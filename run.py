#!/usr/bin/env python
"""
赛博宠物快速启动脚本
CyberPet Quick Start Script

功能：一键运行赛博宠物物体检测系统
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from cyberpet_object_detection import CyberPetEnvironmentDetector


IMAGE_DIR = Path(__file__).parent / "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "=" * 70)
    print(" " * 15 + "🤖 赛博宠物 - 环境检测系统")
    print(" " * 10 + "CyberPet Environment Detection System - Task 2")
    print("=" * 70 + "\n")


def print_menu():
    """打印菜单"""
    print("请选择检测方式:")
    print("1. 检测本地图像 (image)")
    print("2. 使用摄像头实时检测 (camera)")
    print("3. 检测视频文件 (video)")
    print("4. 查看帮助 (help)")
    print("5. 退出 (exit)")
    print()


def detect_image_mode(detector):
    """检测 images/ 目录中的全部图片。"""
    print("\n--- 图像检测模式 ---")
    image_paths = sorted(
        str(path) for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if IMAGE_DIR.exists() else []

    if not image_paths:
        print(f"❌ {IMAGE_DIR} 中没有找到支持的图片")
        print("   请将图片放入 images/ 目录后重试")
        return

    conf = input("输入置信度阈值 (默认: 0.5, 范围: 0-1): ").strip()
    try:
        conf_threshold = float(conf) if conf else 0.5
    except ValueError:
        conf_threshold = 0.5

    print(f"\n正在检测 images/ 中的 {len(image_paths)} 张图片 (置信度: {conf_threshold})...")
    results = detector.detect_images(image_paths, conf_threshold=conf_threshold)

    if results:
        print("\n✅ 检测完成！")
        print("   每张图片的环境地图已自动保存到 maps/ 目录")


def detect_camera_mode(detector):
    """摄像头检测模式"""
    print("\n--- 摄像头实时检测模式 ---")
    print("按 'q' 键退出")

    max_frames = input("输入最多处理的帧数 (默认: 100): ").strip()
    try:
        max_frames = int(max_frames) if max_frames else 100
    except ValueError:
        max_frames = 100

    print(f"\n正在启动摄像头 (最多{max_frames}帧)...")
    detector.detect_video(0, max_frames=max_frames)
    print("✅ 摄像头检测完成！")


def detect_video_mode(detector):
    """视频文件检测模式"""
    print("\n--- 视频文件检测模式 ---")
    video_path = input("请输入视频文件路径: ").strip()

    if not video_path or not Path(video_path).exists():
        print(f"❌ 错误：视频文件不存在 - {video_path}")
        return

    max_frames = input("输入最多处理的帧数 (默认: 100): ").strip()
    try:
        max_frames = int(max_frames) if max_frames else 100
    except ValueError:
        max_frames = 100

    print(f"\n正在检测视频: {video_path}...")
    detector.detect_video(video_path, max_frames=max_frames)
    print("✅ 视频检测完成！")


def print_help():
    """打印帮助信息"""
    print("""
【赛博宠物环境检测系统 - 帮助】

本系统用于为赛博宠物识别和记忆家庭环境。

【功能说明】
1. 物体检测：使用YOLOv8识别家里的物体（床、椅子、台灯等）
2. 位置追踪：记录物体在家中的位置
3. 环境地图：生成JSON格式的环境地图文件
4. 家庭布局：分析物体在房间中的分布

【使用流程】
1. 准备图像：
    - 将家里的照片放入项目 images/ 目录
    - 程序会自动检测其中的全部图片
    - 也可以使用摄像头或录制好的视频

2. 运行检测：
    - 选择检测方式（图像/摄像头/视频）
    - 图像模式自动读取 images/ 目录
    - 视频模式再输入视频文件路径
   - 设置置信度阈值（默认0.5）

3. 查看结果：
   - 控制台输出物体检测统计
   - 保存带注解的检测图片（results/目录）
   - 保存环境地图JSON文件（maps/目录）

【置信度阈值解释】
- 0.3-0.5: 宽松检测，包含低置信度物体
- 0.5-0.7: 标准检测（推荐）
- 0.7-1.0: 严格检测，只检测高置信度物体

【模型选择】
- nano (n): 最快，适合CPU实时推理
- small (s): 平衡速度和精度
- medium (m): 更好的精度
- large (l): 最高精度，需要GPU

【常见问题】
Q: 为什么第一次运行很慢？
A: 第一次运行会下载YOLOv8模型（~45MB），这是正常的。

Q: 如何提高检测准确率？
A: 尝试使用更大的模型或提高置信度阈值。

Q: 检测结果保存在哪里？
A: 
- 带注解的图片：results/ 目录
- 环境地图：maps/ 目录

【后续步骤】
- 这个模块的输出可以被赛博宠物的其他模块（如导航、行为决策）使用
- 参考README.md了解如何在代码中集成这个模块

【联系我们】
如有问题，请查看项目README或提交Issue

    """)


def main():
    """主函数"""
    print_banner()

    # 初始化检测器
    print("🚀 正在初始化赛博宠物环境检测系统...")
    try:
        detector = CyberPetEnvironmentDetector(model_size="n")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n💡 解决方案:")
        print("1. 确保已安装依赖: pip install -r requirements.txt")
        print("2. 确保网络连接正常（首次运行会下载模型）")
        print("3. 检查磁盘空间是否充足")
        return

    print("✅ 系统初始化完成！\n")

    # 交互式菜单
    while True:
        print_menu()
        choice = input("请输入选择 (1-5): ").strip().lower()

        if choice in ["1", "image"]:
            detect_image_mode(detector)
        elif choice in ["2", "camera"]:
            detect_camera_mode(detector)
        elif choice in ["3", "video"]:
            detect_video_mode(detector)
        elif choice in ["4", "help"]:
            print_help()
        elif choice in ["5", "exit"]:
            print("\n👋 感谢使用赛博宠物环境检测系统！")
            break
        else:
            print("❌ 无效选择，请重新输入")

        input("\n按Enter继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()
