# predict.py
import os
import json
import shutil

import pandas as pd
from PIL import Image
import paddle
from paddle.io import DataLoader

from config import Config
from Model import WeatherModel  # 需要模型定义
from dataset_utils import WeatherDataset, load_image_paths_and_labels  # 需要Dataset和加载函数
from paddle.vision import transforms



def predict_on_test_data(model_path_override=None,
                         label_map_path_override=None,
                         test_dir_override=None,
                         submission_file_override=None):
    """
    在测试集上进行预测并生成提交文件。
    """
    if model_path_override: Config.MODEL_SAVE_PATH = model_path_override
    if label_map_path_override: Config.LABEL_MAP_FILE = label_map_path_override
    if test_dir_override: Config.TEST_DIR = test_dir_override
    if submission_file_override: Config.SUBMISSION_FILE = submission_file_override

    print(f"--- 开始预测 ---")
    print(f"使用模型: {Config.MODEL_SAVE_PATH}")
    print(f"使用标签映射: {Config.LABEL_MAP_FILE}")
    print(f"使用测试数据: {Config.TEST_DIR}")
    print(f"测试提交文件将保存到: {Config.SUBMISSION_FILE}")
    print(f"使用设备: {Config.DEVICE}")

    # 1. 加载标签映射以确定类别数和 int_to_label
    if not os.path.exists(Config.LABEL_MAP_FILE):
        print(f"错误: 标签映射文件 {Config.LABEL_MAP_FILE} 未找到。无法进行预测。")
        return None
    try:
        with open(Config.LABEL_MAP_FILE, 'r', encoding='utf-8') as f:
            map_data = json.load(f)
            label_to_int = map_data.get('label_to_int')
            int_to_label_loaded = map_data.get('int_to_label')  # 直接加载 int_to_label
            if not label_to_int or not isinstance(label_to_int, dict) or \
                    not int_to_label_loaded or not isinstance(int_to_label_loaded, dict):
                print(f"错误: 标签映射文件 {Config.LABEL_MAP_FILE} 内容格式不正确。")
                return None
            # 对于 int_to_label，PaddlePaddle 的 Dataset 通常需要 int 类型的键
            # 但我们 json 保存时可能是字符串，需要转换
            int_to_label = {int(k): v for k, v in int_to_label_loaded.items()}

        num_classes = len(label_to_int)
        print(f"从 {Config.LABEL_MAP_FILE} 加载了标签映射。类别数: {num_classes}")
        Config.NUM_CLASSES = num_classes  # 更新Config，虽然模型加载时也会用，但保持一致性
    except Exception as e:
        print(f"加载标签映射 {Config.LABEL_MAP_FILE} 失败: {e}")
        return None

    # 2. 初始化模型
    print("初始化模型...")
    model = WeatherModel(
        model_name=Config.MODEL_NAME,  # <--- **关键修改：从 Config 传递 model_name**
        num_classes=Config.NUM_CLASSES,
        pretrained_flag=False  # 预测时加载自己的权重，所以 backbone 不需要预训练
    )

    # 3. 加载模型权重
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"错误: 模型权重文件 {Config.MODEL_SAVE_PATH} 未找到。")
        return None
    try:
        model_state_dict = paddle.load(Config.MODEL_SAVE_PATH)
        model.set_state_dict(model_state_dict)
        model.to(Config.DEVICE)
        model.eval()  # 设置为评估模式
        print(f"模型权重已从 {Config.MODEL_SAVE_PATH} 加载。")
    except Exception as e:
        print(f"加载模型权重 {Config.MODEL_SAVE_PATH} 失败: {e}")
        return None

    # 4. 定义测试数据转换
    test_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),  # 例如 (125, 125)
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])
    print(
        f"信息: 测试图像将使用以下变换: Resize to {Config.IMAGE_SIZE}, ToTensor, Normalize(mean={Config.MEAN}, std={Config.STD})")

    # 5. 加载测试数据集的路径信息
    print(f"从 {Config.TEST_DIR} 加载测试数据路径...")
    test_df, _, _ = load_image_paths_and_labels(
        Config.TEST_DIR,  # 使用配置中的测试目录
        label_to_int_map_to_use=None,
        is_test_set=True
    )

    if test_df is None or test_df.empty:
        print(f"未能从 {Config.TEST_DIR} 加载任何测试图像，或者目录为空。")
        # 可以选择创建一个空的 submission.json
        submission_data = []  # 空列表代表没有预测结果
        with open(Config.SUBMISSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, ensure_ascii=False, indent=4)
        print(f"已创建空的提交文件: {Config.SUBMISSION_FILE}，因为未找到测试图像。")
        print("--- 预测失败或未返回结果 ---")
        return None

    print(f"成功加载 {len(test_df)} 个测试图像的路径信息。")

    # 6. 创建测试数据加载器
    test_dataset = WeatherDataset(
        dataframe=test_df,
        label_to_int_map=None,  # 测试集不需要标签映射
        for_training=False,  # 不是训练
        is_test_set=True,  # 表明是测试集
        transform=test_transform  # 应用测试变换
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # 测试时不需要打乱
        num_workers=Config.NUM_WORKERS,
        drop_last=False  # 测试时通常处理所有样本
    )

    # 7. 进行预测
    predictions_indices = []  # 存储预测的类别索引
    actual_filenames = []  # 存储对应的文件名
    print("开始预测...")
    with paddle.no_grad():  # 推理时不需要计算梯度
        for batch_idx, (imgs, fns_batch) in enumerate(test_loader):  # WeatherDataset 返回 (图像, 文件名列表)
            imgs = imgs.to(Config.DEVICE)
            outputs = model(imgs)  # 模型前向传播

            preds_indices_batch = paddle.argmax(outputs, axis=1).numpy()

            predictions_indices.extend(preds_indices_batch)
            actual_filenames.extend(fns_batch)  # fns_batch 是一个批次的文件名列表

            if (batch_idx + 1) % Config.LOG_INTERVAL == 0 or batch_idx == len(test_loader) - 1:
                print(f"预测进度: Batch [{batch_idx + 1}/{len(test_loader)}]")

    # 8. 将整数标签索引转换回字符串标签
    predicted_labels_str = [int_to_label.get(p_idx, "unknown_prediction") for p_idx in predictions_indices]

    # 9. 准备提交结果
    submission_data = []
    if len(actual_filenames) != len(predicted_labels_str):
        print(f"错误: 文件名数量 ({len(actual_filenames)}) 与预测标签数量 ({len(predicted_labels_str)}) 不匹配！")
        # 这通常不应该发生，但以防万一
        return None

    for filename, predicted_label in zip(actual_filenames, predicted_labels_str):
        # filename 应该是从 WeatherDataset 返回的纯文件名
        submission_data.append({
            "filename": filename,  # 确保 filename 是 os.path.basename(original_path)
            "label": predicted_label
        })

    # 10. 保存提交文件
    try:
        with open(Config.SUBMISSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, ensure_ascii=False, indent=4)
        print(f"预测结果已保存到: {Config.SUBMISSION_FILE}")
        print(f"--- 预测完成 ---")
    except Exception as e:
        print(f"保存提交文件 {Config.SUBMISSION_FILE} 失败: {e}")
        return None


    print(f"DEBUG (predict.py): Type of submission_data before converting to DF: {type(submission_data)}")
    if isinstance(submission_data, list):
        print(f"DEBUG (predict.py): Length of submission_data list: {len(submission_data)}")

    df_to_return = pd.DataFrame(submission_data)  # 将列表转换为 DataFrame

    print(f"DEBUG (predict.py): Type of value being returned: {type(df_to_return)}")
    if isinstance(df_to_return, pd.DataFrame):
        print(f"DEBUG (predict.py): DataFrame is empty after conversion: {df_to_return.empty}")

    return df_to_return  # <--- 确保返回的是 DataFrame


