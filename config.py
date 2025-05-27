# config.py
import os
import paddle
from paddle.vision import resnet18


class Config:
    # --- 路径配置 ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = 'train_dataset/train'
    TEST_DIR = 'test_dataset/test'
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'weather_classifier_model.pdparams')
    LABEL_MAP_FILE = os.path.join(OUTPUT_DIR, 'label_map.json')
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, 'submission.json')
    # BESTMODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pdparams')


    MODEL_NAME="googlenet"#----------也可以选择"resnet18"/"vgg16"/"resnet50"/"googlenet"
    NUM_CLASSES = 10
    # --- 模型与训练参数 ---
    PRETRAINED_MODEL = True
    EPOCHS = 4
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-2
    VALIDATION_SPLIT_SIZE = 0.2
    RANDOM_SEED = 42

    # --- 数据预处理参数 ---
    RESIZE_SHAPE = (125, 125)
    CROP_SIZE = (125, 125)
    IMAGE_SIZE = (224,224)
    # ImageNet 标准的均值和标准差
    NORM_MEAN = [0.5, 0.5, 0.5]  # <--- 添加这一行  //原始是0.5,0.5[0.485, 0.456, 0.406]
    NORM_STD = [0.5, 0.5, 0.5]   # <--- 添加这一行[0.229, 0.224, 0.225]
    SUPPORTED_IMAGE_FORMATS = '.jpg'  # 这是一个元组，包含所有支持的图片扩展名

    # --- 其他配置 ---
    LOG_INTERVAL = 10
    NUM_WORKERS = 0 # 根据你的CPU核心数调整, Windows下设为0通常更稳定

    # --- 设备配置 ---
    USE_GPU = paddle.is_compiled_with_cuda()
    DEVICE = paddle.set_device('gpu' if USE_GPU else 'cpu')

    @staticmethod
    def ensure_output_dirs():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

Config.ensure_output_dirs()

if __name__ == '__main__':
    print("Config 模块测试:")
    print(f"项目根目录: {Config.BASE_DIR}")
    print(f"训练数据目录: {Config.TRAIN_DIR}")
    print(f"输出目录: {Config.OUTPUT_DIR}")
    print(f"模型保存路径: {Config.MODEL_SAVE_PATH}")
    print(f"使用设备: {Config.DEVICE}")
    print(f"类别数 (初始): {Config.NUM_CLASSES}")
    print(f"随机种子: {Config.RANDOM_SEED}")
    print(f"归一化均值: {Config.NORM_MEAN}") # 测试新添加的属性
    print(f"归一化标准差: {Config.NORM_STD}") # 测试新添加的属性
    Config.ensure_output_dirs()
    print("输出目录检查/创建完毕。")
    print("Config 模块测试结束。")
