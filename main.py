# main.py
import argparse
import os
import sys  # 用于在某些情况下退出

# 确保项目根目录在Python路径中，这样可以正确导入其他模块
# 如果 main.py 就在项目根目录，通常不需要这一步。
# 如果 main.py 在子目录，可能需要调整 sys.path。
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # 假设 main.py 在类似 src/ 的目录下
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

from config import Config


def run_eda():
    """执行探索性数据分析 (EDA)"""
    print("--- 开始数据探索性分析 (EDA) ---")
    try:
        from dataset_utils import load_image_paths_and_labels
        import matplotlib.pyplot as plt
        from PIL import Image
        import pandas as pd  # 确保导入
    except ImportError as e:
        print(f"错误: EDA 所需的库未能导入: {e}")
        print("请确保已安装 matplotlib, Pillow, pandas。")
        return

    print(f"从 {Config.TRAIN_DIR} 加载训练数据进行EDA...")
    # 注意：这里不创建label_to_int_map，因为我们只是想看看原始标签分布
    try:
        train_df_eda, _, _ = load_image_paths_and_labels(Config.TRAIN_DIR, is_test_set=False)
    except FileNotFoundError:
        print(f"错误: 训练目录 {Config.TRAIN_DIR} 未找到。EDA无法进行。")
        return
    except Exception as e:
        print(f"加载数据进行EDA时发生错误: {e}")
        return

    if train_df_eda.empty:
        print(f"未能从 {Config.TRAIN_DIR} 加载任何训练数据进行EDA。请检查目录和文件格式。")
        print(f"支持的图像格式: {Config.SUPPORTED_IMAGE_FORMATS}")
        print("--- EDA结束 (无数据) ---")
        return

    print("训练数据标签分布: ")
    print(train_df_eda['label_str'].value_counts())

    print("随机显示几张训练图片: ")
    # 修正：确保样本数量不超过 DataFrame 长度
    num_samples_to_show = min(9, len(train_df_eda))
    if num_samples_to_show == 0:
        print("没有样本可供显示。")
    print("--- EDA结束 ---")
    return

    sample_eda_df = train_df_eda.sample(n=num_samples_to_show, random_state=Config.RANDOM_SEED)

    cols = 3
    rows = (len(sample_eda_df) + cols - 1) // cols
    fig_width = 5 * cols
    fig_height = 5 * rows
    # 增加一个上限，避免figsize过大
    max_fig_dim = 20
    fig_width = min(fig_width, max_fig_dim)
    fig_height = min(fig_height, max_fig_dim)

    plt.figure(figsize=(fig_width, fig_height))

    for i, row_data in enumerate(sample_eda_df.itertuples()):
        plt.subplot(rows, cols, i + 1)
        try:
            img = Image.open(row_data.filename).convert('RGB')
            plt.imshow(img)
            title = f"Label: {row_data.label_str}Size: {img.size}"
            if hasattr(row_data, 'original_filename'):  # 如果有原始文件名
                title += f"File: {os.path.basename(row_data.original_filename)}"
            plt.title(title, fontsize=8)  # 调小字体以适应更多信息
        except FileNotFoundError:
            plt.text(0.5, 0.5, f'Img not found:{os.path.basename(row_data.filename)}', ha='center', va='center', color='red', fontsize=8)
            print(f"EDA: 图像文件未找到 {row_data.filename}")
        except Exception as e:
            plt.text(0.5, 0.5, 'Error loading img', ha='center', va='center', color='red', fontsize=8)
            print(f"EDA: 加载图像 {row_data.filename} 时发生错误: {e}")
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle("EDA: 随机样本图像", fontsize=14, y=1.02)  # y=1.02 避免与子图标题重叠
    plt.show()
    print("--- EDA结束 ---")


def run_training():
    """执行模型训练和验证"""
    print("--- 开始训练 ---")
    try:
        from train import train_and_validate
    except ImportError:
        print("错误: 'train.py' 或其依赖项未能导入。")
        return False  # 表示训练失败

    trained_model_path = train_and_validate()  # 假设函数返回模型路径或None
    if trained_model_path and os.path.exists(trained_model_path):
        return True  # 表示训练成功
    else:
        print("--- 训练失败或未生成有效模型文件 ---")
        return False  # 表示训练失败


def run_prediction():
    """执行模型预测"""
    # 检查必要文件
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"错误: 模型文件 {Config.MODEL_SAVE_PATH} 未找到。无法进行预测。")
        print("请先运行训练模式，或确保模型文件已放置在正确路径。")
        return False  # 表示预测失败

    if not os.path.exists(Config.LABEL_MAP_FILE):
        print(f"错误: 标签映射文件 {Config.LABEL_MAP_FILE} 未找到。无法进行预测。")
        print("请先运行训练模式，或确保标签映射文件已放置在正确路径。")
        return False  # 表示预测失败

    if not os.path.exists(Config.TEST_DIR) or not os.listdir(Config.TEST_DIR):
        print(f"错误: 测试数据目录 {Config.TEST_DIR} 不存在或为空。无法进行预测。")
        return False

    try:
        from predict import predict_on_test_data  # 假设 predict_on_test_data 在 predict.py 中
    except ImportError:
        print("错误: 'predict.py' 或其依赖项未能导入。")
        return False

    predictions_df = predict_on_test_data()  # 假设返回一个DataFrame或None
    if predictions_df is not None and not predictions_df.empty:
        print(f"--- 预测成功完成。结果已保存在 {Config.SUBMISSION_FILE} ---")
        print("部分预测结果预览:")
        print(predictions_df.head())
        return True  # 表示预测成功
    else:
        print("--- 预测失败或未返回结果 ---")
        return False  # 表示预测失败


def main():
    parser = argparse.ArgumentParser(description="图像分类脚本")
    parser.add_argument('--mode', type=str, default='train_predict',
                        choices=['train', 'predict', 'train_predict', 'eda'],
                        help="运行模式: 'train' (仅训练), 'predict' (仅预测), "
                             "'train_predict' (先训练后预测), 'eda' (数据探索分析)")
    args = parser.parse_args()

    print(f"--- 脚本启动，当前工作目录: {os.getcwd()} ---")
    print(f"选择模式: {args.mode}")
    print(f"使用配置:")
    print(f"  训练数据目录: {Config.TRAIN_DIR}")
    print(f"  测试数据目录: {Config.TEST_DIR}")
    print(f"  输出目录: {Config.OUTPUT_DIR}")
    print(f"  模型保存路径: {Config.MODEL_SAVE_PATH}")
    print(f"  标签映射文件: {Config.LABEL_MAP_FILE}")
    # 可以打印更多重要的Config项



    # 确保输出目录存在 (Config中应该已经处理，但这里再次确认无妨)
    Config.ensure_output_dirs()
    if args.mode == 'eda':
        run_eda()

    elif args.mode == 'train':
        training_successful = run_training()

    elif args.mode == 'predict':
        run_prediction()

    elif args.mode == 'train_predict':
        training_successful = run_training()
        if training_successful:
            run_prediction()
        else:
            print(" - -- 由于训练未成功，跳过预测步骤 - --")

    else:
        print(f"错误: 未知的模式 '{args.mode}'。请从 'train', 'predict', 'train_predict', 'eda' 中选择。")
        parser.print_help()  # 显示帮助信息
        sys.exit(1)  # 以错误码退出

    print(f"--- 脚本模式 '{args.mode}' 执行完毕 ---")

if __name__ == '__main__':
    main()
