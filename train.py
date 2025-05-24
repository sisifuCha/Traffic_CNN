# train.py
import os
import json
import numpy as np
import paddle
import paddle.nn as nn
from paddle.optimizer import Adam
from paddle.io import DataLoader
from PIL import Image  # 用于在测试块中创建虚拟图片

from config import Config
from dataset_utils import load_image_paths_and_labels, prepare_dataloaders
from model_def import WeatherModel


def train_and_validate(existing_label_map_path=None,
                       train_dir_override=None,
                       epochs_override=None,
                       batch_size_override=None,
                       val_split_override=None):
    """
    执行模型训练和验证的主要函数。
    """
    # --- 0. (可选) 保存并根据 override 参数更新 Config ---
    # 保存原始 Config 值，以便在函数结束时恢复（尤其在测试时有用）
    original_config_values = {
        'TRAIN_DIR': Config.TRAIN_DIR,
        'EPOCHS': Config.EPOCHS,
        'BATCH_SIZE': Config.BATCH_SIZE,
        'VALIDATION_SPLIT_SIZE': Config.VALIDATION_SPLIT_SIZE,
        'LABEL_MAP_FILE': Config.LABEL_MAP_FILE, # 如果测试中修改，也应恢复
        'MODEL_SAVE_PATH': Config.MODEL_SAVE_PATH # 如果测试中修改，也应恢复
    }

    best_model_path = None
    if train_dir_override:
        Config.TRAIN_DIR = train_dir_override
        print(f"Config.TRAIN_DIR overridden to: {Config.TRAIN_DIR}")
    if epochs_override is not None:
        Config.EPOCHS = epochs_override
        print(f"Config.EPOCHS overridden to: {Config.EPOCHS}")
    if batch_size_override:
        Config.BATCH_SIZE = batch_size_override
        print(f"Config.BATCH_SIZE overridden to: {Config.BATCH_SIZE}")
    if val_split_override is not None:
        Config.VALIDATION_SPLIT_SIZE = val_split_override
        print(f"Config.VALIDATION_SPLIT_SIZE overridden to: {Config.VALIDATION_SPLIT_SIZE}")

    Config.ensure_output_dirs()  # 确保输出目录（可能基于覆盖的Config）存在
    print(f"使用设备: {Config.DEVICE}")

    # --- 1. 处理 existing_label_map_path 以获取 existing_label_to_int ---
    label_map_to_use = None # 初始化，这将传递给 prepare_dataloaders
    if existing_label_map_path and os.path.exists(existing_label_map_path):
        try:
            with open(existing_label_map_path, 'r', encoding='utf-8') as f:
                maps_from_file = json.load(f)
                # 假设文件总是包含 'label_to_int'
                loaded_l2i = maps_from_file.get('label_to_int')
            if loaded_l2i and isinstance(loaded_l2i, dict):
                label_map_to_use = loaded_l2i
                print(f"成功从 {existing_label_map_path} 加载预定义的标签映射。")
            else:
                if not loaded_l2i:
                    print(f"警告: 文件 {existing_label_map_path} 中未找到 'label_to_int' 键或其值无效。将动态生成标签映射。")
                elif not isinstance(loaded_l2i, dict):
                    print(f"警告: 从 {existing_label_map_path} 加载的标签映射不是字典类型。将动态生成标签映射。")
        except Exception as e:
            print(f"从 {existing_label_map_path} 加载标签映射失败: {e}。将动态生成标签映射。")
    elif existing_label_map_path: # 提供了路径但文件不存在
        print(f"警告: 提供的标签映射文件 {existing_label_map_path} 不存在。将动态生成标签映射。")
    else: # 没有提供路径
        print("未提供现有标签映射文件路径。将动态生成标签映射。")

    print(f"传递给 prepare_dataloaders 的 existing_label_to_int_map 是: {label_map_to_use}")

    # --- 2. 调用 prepare_dataloaders ---
    # 假设 prepare_dataloaders 的参数名是 existing_label_to_int_map
    # 并且它会:
    #   - 使用 label_map_to_use (如果提供) 或动态生成新的标签映射
    #   - 基于最终的标签映射更新 Config.NUM_CLASSES
    #   - 如果生成了新的或使用了不同的标签映射，则保存到 Config.LABEL_MAP_FILE
    #   - 返回: train_loader, val_loader, final_label_to_int_map
    train_loader, val_loader, final_label_to_int_map = prepare_dataloaders(
        existing_label_to_int_map=label_map_to_use # <--- 将加载的或None的映射传递过去
    )

    if not train_loader:
        print("错误: 训练 DataLoader 未能创建。训练中止。")
        # 恢复 Config
        for key, value in original_config_values.items(): setattr(Config, key, value)
        return None

    if not final_label_to_int_map:
        print("错误: 标签映射未能有效生成或加载。无法确定类别数。训练中止。")
        # 恢复 Config
        for key, value in original_config_values.items(): setattr(Config, key, value)
        return None

    # Config.NUM_CLASSES 应该已经在 prepare_dataloaders 内部被正确设置
    # 我们再次确认一下
    num_classes_from_map = len(final_label_to_int_map)
    if Config.NUM_CLASSES != num_classes_from_map:
        print(f"警告: Config.NUM_CLASSES ({Config.NUM_CLASSES}) 与从最终标签映射得到的类别数 ({num_classes_from_map}) 不一致。已自动校正为 {num_classes_from_map}。")
        Config.NUM_CLASSES = num_classes_from_map # 确保一致性

    if Config.NUM_CLASSES <= 0:
        print(f"错误: 最终确定的类别数 ({Config.NUM_CLASSES}) 无效。训练中止。")
        # 恢复 Config
        for key, value in original_config_values.items(): setattr(Config, key, value)
        return None

    print(f"数据加载和预处理完成。最终训练类别数: {Config.NUM_CLASSES}")
    print(f"最终使用的标签映射 (label_to_int): {final_label_to_int_map}")
    if os.path.exists(Config.LABEL_MAP_FILE):
        # 可以选择在这里验证 Config.LABEL_MAP_FILE 的内容是否与 final_label_to_int_map 一致
        print(f"标签映射已根据需要保存到/更新于 {Config.LABEL_MAP_FILE}")
    else:
        # 这种情况不应该发生，如果 prepare_dataloaders 逻辑正确且需要保存映射
        print(f"警告: 标签映射文件 {Config.LABEL_MAP_FILE} 未找到，即使数据加载完成。请检查 prepare_dataloaders 实现。")
    # 2. 初始化模型
    print("初始化模型...")
    model = WeatherModel(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(parameters=model.parameters(), learning_rate=Config.LEARNING_RATE)

    # 4. 训练循环
    best_val_accuracy = 0.0  # 用于保存最佳模型
    print(f"开始训练，共{Config.EPOCHS}个Epochs...")
    for epoch in range(Config.EPOCHS):
        model.train()  # 设置模型为训练模式
    epoch_train_loss = []
    epoch_train_acc = []

    # --- 训练阶段 ---
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE).squeeze()  # CrossEntropyLoss期望1D标签

    # 前向传播
    pred_logits = model(images)

    # 计算损失
    loss = criterion(pred_logits, labels)

    # 计算准确率
    # paddle.metric.accuracy 的输入 logits 和 1D 标签
    acc = paddle.metric.accuracy(input=pred_logits, label=labels.unsqueeze(1))  # label需要是[N,1]

    # 反向传播和优化
    optimizer.clear_grad()
    loss.backward()
    optimizer.step()

    epoch_train_loss.append(loss.item())
    epoch_train_acc.append(acc.item())

    if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
        print(f"Epoch [{epoch + 1}/{Config.EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], "
              f"训练损失: {loss.item():.4f}, 训练准确率: {acc.item():.4f}")

    avg_train_loss = np.mean(epoch_train_loss)
    avg_train_acc = np.mean(epoch_train_acc)
    print(f"--- Epoch [{epoch + 1}/{Config.EPOCHS}] 总结 ---")
    print(f"平均训练损失: {avg_train_loss:.4f}, 平均训练准确率: {avg_train_acc:.4f}")

    # --- 验证阶段 ---
    if val_loader and len(val_loader) > 0:  # 只有当有验证集且验证集非空时才进行验证
        model.eval()  # 设置模型为评估模式
        epoch_val_loss = []
        epoch_val_acc = []
        with paddle.no_grad():  # 验证时不需要计算梯度
            for images_val, labels_val in val_loader:
                images_val = images_val.to(Config.DEVICE)
                labels_val = labels_val.to(Config.DEVICE).squeeze()

                pred_logits_val = model(images_val)
                loss_val = criterion(pred_logits_val, labels_val)
                acc_val = paddle.metric.accuracy(input=pred_logits_val, label=labels_val.unsqueeze(1))

                epoch_val_loss.append(loss_val.item())
                epoch_val_acc.append(acc_val.item())

        avg_val_loss = np.mean(epoch_val_loss) if epoch_val_loss else 0
        avg_val_acc = np.mean(epoch_val_acc) if epoch_val_acc else 0
        print(f"平均验证损失: {avg_val_loss:.4f}, 平均验证准确率: {avg_val_acc:.4f}")

        # 保存最佳模型 (基于验证准确率)
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            paddle.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"在 Epoch {epoch + 1} 保存了新的最佳模型到 {Config.MODEL_SAVE_PATH} (验证准确率: {best_val_accuracy:.4f})")
    else:  # 如果没有验证集或验证集为空，则每轮都保存模型
        # 仅当训练数据存在时才保存
        if train_loader and len(train_loader) > 0:
            best_model_path = Config.MODEL_SAVE_PATH  # 记录路径
            paddle.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Epoch [{epoch + 1}/{Config.EPOCHS}] 结束, 模型已保存到 {Config.MODEL_SAVE_PATH} (无有效验证集)")
        else:
            print(f"Epoch [{epoch + 1}/{Config.EPOCHS}] 结束, 但无训练数据，未保存模型。")

    print("-" * 50)

    print("训练完成。")
    if val_loader and len(val_loader) > 0:
        print(f"最佳验证准确率: {best_val_accuracy:.4f}")
    if best_model_path and os.path.exists(best_model_path):  # 确保文件确实存在
        print(f"最终模型保存在: {best_model_path}")
        return best_model_path  # <--- 返回路径字符串
    elif os.path.exists(Config.MODEL_SAVE_PATH):  # 作为备选，如果逻辑是最终都会保存
        print(f"最终模型保存在: {Config.MODEL_SAVE_PATH}")
        return Config.MODEL_SAVE_PATH
    else:
        print("模型未被保存或保存失败。")
        return None


if __name__ == '__main__':
    print("训练脚本 (train.py) 测试:")

    # 确保输出目录存在
    Config.ensure_output_dirs()

    # 临时创建虚拟训练数据 (如果不存在)
    dummy_train_dir_for_train_test = './dummy_train_for_train_script'
    # 使用至少两个类别，每个类别有足够的样本以满足分层抽样
    labels_for_dummy_train = ['classAlpha', 'classBeta', 'classGamma']
    samples_per_class_dummy = 10  # 每个类别10张图

    if not os.path.exists(dummy_train_dir_for_train_test):
        print(f"为 train.py 测试创建虚拟数据于: {dummy_train_dir_for_train_test}")
        for lbl in labels_for_dummy_train:
            os.makedirs(os.path.join(dummy_train_dir_for_train_test, lbl), exist_ok=True)
            for i in range(samples_per_class_dummy):
                try:
                    img_color = 'red' if lbl == 'classAlpha' else ('blue' if lbl == 'classBeta' else 'green')
                    img = Image.new('RGB', (Config.CROP_SIZE[0] + 20, Config.CROP_SIZE[1] + 20), color=img_color)
                    img.save(os.path.join(dummy_train_dir_for_train_test, lbl, f'dummy_train_img_{i}.png'))
                except Exception as e:
                    print(f"创建虚拟图片失败: {e}")

    original_train_dir_config = Config.TRAIN_DIR
    original_epochs_config = Config.EPOCHS
    original_val_split_config = Config.VALIDATION_SPLIT_SIZE
    original_model_save_path_config = Config.MODEL_SAVE_PATH
    original_label_map_file_config = Config.LABEL_MAP_FILE
    original_batch_size_config = Config.BATCH_SIZE
    original_num_classes_config = Config.NUM_CLASSES

    # 修改Config以适应快速测试
    Config.TRAIN_DIR = dummy_train_dir_for_train_test
    Config.EPOCHS = 2
    # 确保验证集大小合理：如果为整数，至少为类别数；如果为比例，确保每类至少一个样本
    # 总样本数: len(labels_for_dummy_train) * samples_per_class_dummy = 3 * 10 = 30
    # 如果 Config.VALIDATION_SPLIT_SIZE 是整数，则其值应 >= len(labels_for_dummy_train)
    # 如果是比例，例如 0.2，则验证集大小为 30 * 0.2 = 6，每个类别约2个，是合理的。
    Config.VALIDATION_SPLIT_SIZE = 0.2  # 使用20%作为验证集 (即6个样本)
    # 或者 Config.VALIDATION_SPLIT_SIZE = len(labels_for_dummy_train) # 确保整数值至少为类别数

    Config.MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, 'test_train_model.pdparams')
    Config.LABEL_MAP_FILE = os.path.join(Config.OUTPUT_DIR, 'test_train_label_map.json')
    Config.BATCH_SIZE = 4  # 减小批量大小以便在小数据集上运行
    # Config.NUM_CLASSES 会由 prepare_dataloaders 根据实际数据设置，这里无需预设

    print(f"使用虚拟训练数据: {Config.TRAIN_DIR}")
    print(f"Epochs for test: {Config.EPOCHS}")
    print(f"Validation split for test: {Config.VALIDATION_SPLIT_SIZE}")
    print(f"Batch size for test: {Config.BATCH_SIZE}")

    trained_model = train_and_validate()

    if trained_model:
        print("train_and_validate函数执行完毕。")
        assert os.path.exists(Config.MODEL_SAVE_PATH), f"模型文件 {Config.MODEL_SAVE_PATH} 未创建!"
        assert os.path.exists(Config.LABEL_MAP_FILE), f"标签映射文件 {Config.LABEL_MAP_FILE} 未创建!"
        print(f"测试模型文件 {Config.MODEL_SAVE_PATH} 已创建。")
        print(f"测试标签映射文件 {Config.LABEL_MAP_FILE} 已创建。")

        # 检查标签映射文件内容
        try:
            with open(Config.LABEL_MAP_FILE, 'r') as f:
                label_map_content = json.load(f)
            assert "label_to_int" in label_map_content
            assert "int_to_label" in label_map_content
            assert len(label_map_content["label_to_int"]) == len(labels_for_dummy_train)
            print("标签映射文件内容基本正确。")
        except Exception as e:
            print(f"检查标签映射文件失败: {e}")

    else:
        print("\ntrain_and_validate 函数未能成功完成或未返回模型。")

    # 恢复原始配置 (重要，以防其他测试依赖原始配置)
    Config.TRAIN_DIR = original_train_dir_config
    Config.EPOCHS = original_epochs_config
    Config.VALIDATION_SPLIT_SIZE = original_val_split_config
    Config.MODEL_SAVE_PATH = original_model_save_path_config
    Config.LABEL_MAP_FILE = original_label_map_file_config
    Config.BATCH_SIZE = original_batch_size_config
    Config.NUM_CLASSES = original_num_classes_config  # 恢复 NUM_CLASSES

    # (可选) 清理为 train.py 测试创建的虚拟数据和文件
    # import shutil
    # if os.path.exists(dummy_train_dir_for_train_test):
    #     print(f"清理为train.py测试创建的虚拟数据: {dummy_train_dir_for_train_test}")
#     shutil.rmtree(dummy_train_dir_for_train_test)
# if os.path.exists(Config.MODEL_SAVE_PATH) and Config.MODEL_SAVE_PATH.startswith(os.path.join(Config.OUTPUT_DIR, 'test_train_')): # 清理测试模型
#     os.remove(Config.MODEL_SAVE_PATH)
# if os.path.exists(Config.LABEL_MAP_FILE) and Config.LABEL_MAP_FILE.startswith(os.path.join(Config.OUTPUT_DIR, 'test_train_')): # 清理测试标签映射
#     os.remove(Config.LABEL_MAP_FILE)

print("train.py测试结束。")

