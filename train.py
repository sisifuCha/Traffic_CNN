# train.py
import os
import json
import numpy as np
import paddle
import paddle.nn as nn
from paddle.optimizer import Adam
from paddle.io import DataLoader
from PIL import Image  # 用于在测试块中创建虚拟图片
import matplotlib.pyplot as plt # 新增：用于绘图
from paddle.metric import Accuracy, Precision, Recall # 新增：更多评估指标
from config import Config
from dataset_utils import load_image_paths_and_labels, prepare_dataloaders
from Model import WeatherModel


# --- NEW: Plotting Function ---
def plot_metrics(history, save_path, num_epochs):
    """
    绘制训练和验证过程中的损失、准确率和F1分数。
    """
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(18, 6)) # Adjusted figure size slightly

    # Plot Loss
    plt.subplot(1, 3, 1)
    if history['train_loss']:
        plt.plot(epochs_range, history['train_loss'], 'o-', label='Training Loss')
    if history['val_loss'] and any(v is not None for v in history['val_loss']): # Check if there's actual val data
        plt.plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    if history['train_accuracy']:
        plt.plot(epochs_range, history['train_accuracy'], 'o-', label='Training Accuracy')
    if history['val_accuracy'] and any(v is not None for v in history['val_accuracy']):
        plt.plot(epochs_range, history['val_accuracy'], 'o-', label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05) # Accuracy is between 0 and 1
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"训练指标图已保存到: {save_path}")
    except Exception as e:
        print(f"保存指标图失败: {e}")
    # plt.show() # Uncomment if you want to display the plot during script execution


def train_and_validate(existing_label_map_path=None,
                       train_dir_override=None,
                       epochs_override=None,
                       batch_size_override=None,
                       val_split_override=None):
    """
    执行模型训练和验证的主要函数。
    """
    # --- 0. (可选) 保存并根据 override 参数更新 Config ---
    original_config_values = {
        'TRAIN_DIR': Config.TRAIN_DIR,
        'EPOCHS': Config.EPOCHS,
        'BATCH_SIZE': Config.BATCH_SIZE,
        'VALIDATION_SPLIT_SIZE': Config.VALIDATION_SPLIT_SIZE,
        'LABEL_MAP_FILE': Config.LABEL_MAP_FILE,
        'MODEL_SAVE_PATH': Config.MODEL_SAVE_PATH,
        # 确保也保存和恢复学习率相关的配置
        'LEARNING_RATE': Config.LEARNING_RATE,
        'WEIGHT_DECAY': getattr(Config, 'WEIGHT_DECAY', 0.0)
    }
    final_model_save_path = Config.MODEL_SAVE_PATH
    best_model_explicit_path = getattr(Config, 'BESTMODEL_PATH', os.path.join(Config.OUTPUT_DIR, 'best_model.pdparams'))
    os.makedirs(os.path.dirname(best_model_explicit_path), exist_ok=True)

    metrics_plot_save_path = getattr(Config, 'METRICS_PLOT_SAVE_PATH',
                                     os.path.join(Config.OUTPUT_DIR, 'training_metrics.png'))
    os.makedirs(os.path.dirname(metrics_plot_save_path), exist_ok=True)
    Config.ensure_output_dirs()
    print(f"使用设备: {Config.DEVICE}")

    # --- 1. 处理 existing_label_map_path ---
    label_map_to_use = None
    if existing_label_map_path and os.path.exists(existing_label_map_path):
        try:
            with open(existing_label_map_path, 'r', encoding='utf-8') as f:
                maps_from_file = json.load(f)
                loaded_l2i = maps_from_file.get('label_to_int')
            if loaded_l2i and isinstance(loaded_l2i, dict):
                label_map_to_use = loaded_l2i
                print(f"成功从 {existing_label_map_path} 加载预定义的标签映射。")
            else:
                print(f"警告: 文件 {existing_label_map_path} 中的 'label_to_int' 无效。将动态生成。")
        except Exception as e:
            print(f"从 {existing_label_map_path} 加载标签映射失败: {e}。将动态生成。")
    elif existing_label_map_path:
        print(f"警告: 提供的标签映射文件 {existing_label_map_path} 不存在。将动态生成。")
    else:
        print("未提供现有标签映射文件路径。将动态生成标签映射。")
    # print(f"传递给 prepare_dataloaders 的 existing_label_to_int_map 是: {label_map_to_use}") # 可以按需保留或移除

    train_loader, val_loader, final_label_to_int_map = prepare_dataloaders(
        existing_label_to_int_map=label_map_to_use
    )
    if not train_loader or not final_label_to_int_map:
        print("错误: DataLoader 或标签映射未能创建。训练中止。")
        for key, value in original_config_values.items():
            if hasattr(Config, key): setattr(Config, key, value)
        return None
    print(f"数据加载和预处理完成。最终训练类别数: {Config.NUM_CLASSES}")
    print(f"最终使用的标签映射 (label_to_int): {final_label_to_int_map}")

    if os.path.exists(Config.LABEL_MAP_FILE):
        print(f"标签映射已根据需要保存到/更新于 {Config.LABEL_MAP_FILE}")
    else:
        print(f"警告: 标签映射文件 {Config.LABEL_MAP_FILE} 未找到。")

    # --- 2. 初始化模型 ---
    print(f"初始化模型 (模型名称: {Config.MODEL_NAME}, 类别数: {Config.NUM_CLASSES})...")
    model = WeatherModel(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained_flag=True
    )
    model.to(Config.DEVICE)

    # --- 3. 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    # 从 Config 获取 WEIGHT_DECAY，如果未定义则默认为 0.0
    weight_decay_val = getattr(Config, 'WEIGHT_DECAY', 0.0)
    optimizer = Adam(parameters=model.parameters(), learning_rate=Config.LEARNING_RATE, weight_decay=weight_decay_val)
    print(f"优化器 Adam 初始化完毕。学习率: {Config.LEARNING_RATE}, 权重衰减: {weight_decay_val}")

    # --- 4. 初始化指标记录器和最佳模型跟踪 ---
    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': []
    }
    best_validation_accuracy = 0.0

    print(f"开始训练，共{Config.EPOCHS}个Epochs...")
    for epoch_idx in range(Config.EPOCHS):
        epoch_num = epoch_idx + 1
        model.train()
        current_epoch_train_losses = []

        # 手动累积准确率指标
        epoch_train_correct_preds = 0
        epoch_train_total_samples = 0

        # --- 训练阶段 ---
        for batch_idx, (images, labels) in enumerate(train_loader):
            images_device = images.to(Config.DEVICE)
            labels_device_squeezed = labels.to(Config.DEVICE).squeeze()
            pred_logits = model(images_device)

            loss = criterion(pred_logits, labels_device_squeezed)

            # 手动计算 batch 准确率并累积
            with paddle.no_grad():
                preds_indices_batch = paddle.argmax(pred_logits, axis=1)
                correct_batch_count = paddle.sum(preds_indices_batch == labels_device_squeezed).item()
                epoch_train_correct_preds += correct_batch_count
                epoch_train_total_samples += len(labels_device_squeezed)
                current_batch_acc_val = correct_batch_count / len(labels_device_squeezed) if len(
                    labels_device_squeezed) > 0 else 0.0

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            current_epoch_train_losses.append(loss.item())

            if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
                print(f"Epoch [{epoch_num}/{Config.EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"训练损失: {loss.item():.4f}, 训练准确率 (batch MANUAL): {current_batch_acc_val:.4f}")

        # --- Epoch 训练结束 ---
        avg_epoch_train_loss = np.mean(current_epoch_train_losses) if current_epoch_train_losses else 0.0
        epoch_train_accuracy = epoch_train_correct_preds / epoch_train_total_samples if epoch_train_total_samples > 0 else 0.0

        history['train_loss'].append(avg_epoch_train_loss)
        history['train_accuracy'].append(epoch_train_accuracy)

        print(f"--- Epoch [{epoch_num}/{Config.EPOCHS}] 训练总结 ---")
        print(f"平均训练损失: {avg_epoch_train_loss:.4f}")
        print(f"平均训练准确率 (MANUAL): {epoch_train_accuracy:.4f}")

        # --- 验证阶段 ---
        current_epoch_val_acc_float = 0.0
        avg_epoch_val_loss = np.nan

        if val_loader and len(val_loader) > 0:
            model.eval()
            current_epoch_val_losses = []
            epoch_val_correct_preds = 0
            epoch_val_total_samples = 0

            with paddle.no_grad():
                for batch_val_idx, (images_val, labels_val) in enumerate(val_loader):
                    images_val_device = images_val.to(Config.DEVICE)
                    labels_val_device_squeezed = labels_val.to(Config.DEVICE).squeeze()
                    pred_logits_val = model(images_val_device)
                    loss_val = criterion(pred_logits_val, labels_val_device_squeezed)
                    current_epoch_val_losses.append(loss_val.item())

                    preds_indices_val_batch = paddle.argmax(pred_logits_val, axis=1)
                    correct_val_batch_count = paddle.sum(preds_indices_val_batch == labels_val_device_squeezed).item()
                    epoch_val_correct_preds += correct_val_batch_count
                    epoch_val_total_samples += len(labels_val_device_squeezed)

            avg_epoch_val_loss = np.mean(current_epoch_val_losses) if current_epoch_val_losses else 0.0
            current_epoch_val_acc_float = epoch_val_correct_preds / epoch_val_total_samples if epoch_val_total_samples > 0 else 0.0

            history['val_loss'].append(avg_epoch_val_loss)
            history['val_accuracy'].append(current_epoch_val_acc_float)

            print(f"--- Epoch [{epoch_num}/{Config.EPOCHS}] 验证总结 ---")
            print(f"平均验证损失: {avg_epoch_val_loss:.4f}")
            print(f"平均验证准确率 (MANUAL): {current_epoch_val_acc_float:.4f}")

            best_model_target_path = getattr(Config, 'BESTMODEL_PATH', Config.MODEL_SAVE_PATH)
            if current_epoch_val_acc_float > best_validation_accuracy:
                best_validation_accuracy = current_epoch_val_acc_float
                paddle.save(model.state_dict(), best_model_target_path)
                print(f"在 Epoch {epoch_num} 保存了新的最佳模型到 {best_model_target_path} "
                      f"(验证准确率 (MANUAL): {best_validation_accuracy:.4f})")
        else:  # 无验证集
            history['val_loss'].append(np.nan)
            history['val_accuracy'].append(np.nan)
            if train_loader and len(train_loader) > 0:  # 确保有训练数据才保存
                paddle.save(model.state_dict(), final_model_save_path)
                print(
                    f"Epoch [{epoch_num}/{Config.EPOCHS}] 结束, 模型已保存到 {final_model_save_path} (无有效验证集)")
            else:
                print(f"Epoch [{epoch_num}/{Config.EPOCHS}] 结束, 但无训练数据，未保存模型。")
        print("-" * 50)

    print("训练完成。")
    path_of_best_model_if_val_exists = getattr(Config, 'BESTMODEL_PATH', Config.MODEL_SAVE_PATH)

    if val_loader and len(val_loader) > 0:
        print(
            f"最佳验证准确率 (MANUAL): {best_validation_accuracy:.4f} (模型可能保存在: {path_of_best_model_if_val_exists})")
        if os.path.exists(path_of_best_model_if_val_exists) and best_validation_accuracy > 0:
            print(f"  确认最佳模型文件位于: {path_of_best_model_if_val_exists}")
        elif best_validation_accuracy == 0 and not os.path.exists(path_of_best_model_if_val_exists):
            print(f"  注意: 未达到任何验证准确率提升，未保存特定最佳模型或路径配置问题。")
        else:  # 包括 best_validation_accuracy > 0 但文件不存在的情况，或者 best_validation_accuracy == 0 但文件存在的情况
            print(
                f"  警告: 最佳模型文件 {path_of_best_model_if_val_exists} 可能未按预期保存，或准确率未提升但文件存在，或准确率提升但文件未找到。请检查配置和保存逻辑。")
    elif train_loader and len(train_loader) > 0 and os.path.exists(final_model_save_path):  # 检查最终模型是否已保存
        print(f"最终模型保存在 {final_model_save_path} (基于最后一个epoch，无验证)")
    elif train_loader and len(train_loader) > 0 and not os.path.exists(final_model_save_path):
        print(f"警告: 训练结束，但最终模型文件 {final_model_save_path} 未找到 (无验证集情况)。")
    else:  # 无训练数据也无验证数据
        print(f"警告: 训练结束，但无训练数据，未保存任何模型。")

    if any(history['train_loss']) or any(history['val_loss']):  # 仅当有数据时绘图
        plot_metrics(history, metrics_plot_save_path, Config.EPOCHS)
    else:
        print("无训练或验证损失数据，跳过绘图。")

    # 恢复原始 Config 值
    for key, value in original_config_values.items():
        if hasattr(Config, key):  # 确保属性存在以避免 AttributeError
            setattr(Config, key, value)
        # else: # 可选：如果希望对不存在的属性发出警告
        #     print(f"警告: 尝试恢复 Config 属性 '{key}'，但它在 Config 类中不存在。")

    path_to_return = None
    if val_loader and len(val_loader) > 0 and best_validation_accuracy > 0 and os.path.exists(
            path_of_best_model_if_val_exists):
        print(f"返回最佳模型文件: {path_of_best_model_if_val_exists}")
        path_to_return = path_of_best_model_if_val_exists
    elif train_loader and len(train_loader) > 0 and os.path.exists(final_model_save_path):  # 确保训练过且文件存在
        print(f"返回最终模型文件(无验证或最佳模型未保存): {final_model_save_path}")
        path_to_return = final_model_save_path
    else:
        print(f"警告: 模型文件均未找到，无法返回路径。请检查训练过程和保存逻辑。")
    return path_to_return



