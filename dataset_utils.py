# dataset_utils.py
import os
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import paddle
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms
from config import Config  # 假设 Config.py 在同一目录下或PYTHONPATH中


class WeatherDataset(Dataset):
    """
    处理训练、验证和测试数据。
    """
    def __init__(self, dataframe,
                 label_to_int_map,  # 对于测试集，此参数可以是 None
                 for_training,      # 指示是否为训练模式（影响数据增强）
                 is_test_set=False,
                 transform=None):   # 新增 transform 参数
        self.df = dataframe
        self.label_to_int_map = label_to_int_map
        self.for_training = for_training # for_training 通常用于控制是否应用训练时的数据增强
        self.is_test_set = is_test_set
        self.transform = transform # 存储传入的 transform

    #     # 确保测试集DataFrame包含 'original_filename' 列
    #     if self.is_test_set:
    #         if 'original_filename' not in self.df.columns:
    #             if 'filename' in self.df.columns:
    #                 print("提示: 测试集DataFrame中未找到 'original_filename' 列，从 'filename' 列创建。")
    #                 self.df['original_filename'] = self.df['filename'].apply(os.path.basename)
    #             else:
    #                 raise ValueError("测试集DataFrame必须包含 'original_filename' 或 'filename' 列。")
    #
    #     # 定义图像变换
    #     # 如果是测试集，则 for_training 参数无效，总是使用非训练变换
    #     apply_training_transforms = for_training and not self.is_test_set
    #
    #     if apply_training_transforms:
    #         # 训练集变换 (通常包括数据增强)
    #         self.transform = transforms.Compose([
    #             transforms.Resize(size=[int(s * 1.125) for s in Config.CROP_SIZE]),  # 先放大一点
    #             transforms.RandomResizedCrop(Config.CROP_SIZE, scale=(0.8, 1.0)),
    #             transforms.RandomHorizontalFlip(prob=0.5),
    #             transforms.RandomRotation(degrees=10),
    #             # 可以考虑添加 ColorJitter 等
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    #         ])
    #     else:
    #         # 验证集或测试集变换 (通常不包括数据增强)
    #         self.transform = transforms.Compose([
    #             transforms.Resize(Config.RESIZE_SHAPE),  # 先resize到一个固定尺寸
    #             transforms.CenterCrop(Config.CROP_SIZE),  # 然后中心裁剪
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    #         ])
    #
    # def __getitem__(self, index):
    #     row = self.df.iloc[index]
    #     img_path = row['filename']
    #     image_load_failed = False
    #
    #     try:
    #         image = Image.open(img_path).convert('RGB')
    #     except FileNotFoundError:
    #         print(f"错误: 图像文件未找到 {img_path}。将使用占位符图像。")
    #         image = Image.new('RGB', Config.CROP_SIZE, (0, 0, 0))  # 黑色占位符
    #         image_load_failed = True
    #     except Exception as e:
    #         print(f"错误: 读取图像 {img_path} 失败 ({e})。将使用占位符图像。")
    #         image = Image.new('RGB', Config.CROP_SIZE, (0, 0, 0))
    #         image_load_failed = True
    #
    #     image_tensor = self.transform(image)
    #
    #     if self.is_test_set:
    #         # 测试集返回图像张量和原始文件名
    #         original_filename = row['original_filename']
    #         if image_load_failed:
    #             original_filename += "_LOAD_ERROR"  # 标记加载失败的图像
    #         return image_tensor, original_filename
    #     else:
    #         # 训练/验证集返回图像张量和标签张量
    #         if 'label_int' not in row or pd.isna(row['label_int']):
    #             print(f"警告: {img_path} 的标签 'label_int' 缺失或为NaN。使用占位符标签 -1。")
    #             label_tensor = paddle.to_tensor(-1, dtype='int64')  # 使用一个特殊值表示无效标签
    #         else:
    #             label_int = int(row['label_int'])
    #             label_tensor = paddle.to_tensor(label_int, dtype='int64')
    #         return image_tensor, label_tensor
    #
    # def __len__(self):
    #     return len(self.df)
        # 在 __init__ 中进行必要的检查
        if 'image_path' not in self.df.columns:
            raise ValueError("DataFrame 中必须包含 'image_path' 列。")
        if self.is_test_set and 'filename' not in self.df.columns:
            raise ValueError("对于测试集 (is_test_set=True)，DataFrame 中必须包含 'filename' 列。")
        if not self.is_test_set and 'label' not in self.df.columns and self.label_to_int_map is not None:
             # 对于训练/验证集，如果提供了label_map，则应该有label列
             pass # 或者添加更严格的检查

        if self.transform is None:
             # 理论上，即使是测试集，也应该有基础的变换（如 Resize, ToTensor, Normalize）
             print("警告: WeatherDataset 初始化时未提供 transform。图像将不会被预处理。")


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']

        try:
            # 你可以选择你的图像加载方式:
            img = Image.open(img_path).convert('RGB') # 使用 PIL
            # 或者:
            # img_bytes = paddle.vision.image.read_file(img_path) # 使用 paddle.vision.image
            # img = paddle.vision.image.decode_jpeg(img_bytes, mode='rgb') # 或 decode_png 等
        except Exception as e:
            print(f"错误: 加载图像失败 '{img_path}': {e}")
            # 可以返回一个占位符图像和文件名，或者重新抛出异常
            # 这里我们简单地重新抛出，让 DataLoader 处理 (如果它有错误处理机制)
            raise

        if self.transform:
            img = self.transform(img) # 应用图像变换

        if self.is_test_set:
            filename = row['filename'] # 从 DataFrame 获取文件名
            return img, filename
        else:
            # 训练或验证模式
            if self.label_to_int_map is None:
                 raise ValueError("对于训练/验证集，label_to_int_map 不能为空。")
            label_str = row['label']
            label_int = self.label_to_int_map.get(label_str)
            if label_int is None:
                raise ValueError(f"在标签映射中未找到标签 '{label_str}' (来自图像 '{img_path}')。")
            return img, paddle.to_tensor(label_int, dtype='int64')


    def __len__(self):
        return len(self.df)

def find_image_files_recursively_for_test(directory_path):
    """
    递归地在目录及其子目录中查找图像文件。
    返回两个列表：图像的完整路径和它们的文件名。
    """
    image_paths_list = []
    filenames_list = []
    # 根据你的图像文件类型调整扩展名
    valid_extensions = '.jpg'

    if not os.path.isdir(directory_path):
        print(f"警告: 测试数据目录 '{directory_path}' 不是一个有效的目录或不存在。")
        return [], [] # 返回空列表

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file_name)
                image_paths_list.append(full_path)
                filenames_list.append(file_name) # 只存储文件名
    return image_paths_list, filenames_list


def load_image_paths_and_labels(data_dir: str,
                                label_to_int_map_to_use=None,
                                is_test_set=False,
                                create_map_if_not_provided=False,
                                label_map_save_path=None
                                # --- 结束原有参数 ---
                                ):
    """
    从目录结构加载图像路径和标签。
    """
    image_paths = []
    original_filenames = []
    labels_str = []
    labels_int = []

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 未找到。")
        return pd.DataFrame(columns=['filename', 'original_filename', 'label_str', 'label_int']), {}, {}

    current_label_to_int = {}
    current_int_to_label = {}

    if is_test_set:
         # 测试集：图像直接在 data_dir 下，没有类别子目录
        # for item_name in sorted(os.listdir(data_dir)):
        #     full_item_path = os.path.join(data_dir, item_name)
        #     # 确保只处理文件，并且是支持的图像类型
        #     if os.path.isfile(full_item_path) and item_name.lower().endswith(Config.SUPPORTED_IMAGE_FORMATS):
        #         image_paths.append(full_item_path)
        #         original_filenames.append(item_name)  # item_name 就是原始文件名
        #         labels_str.append("unknown")  # 测试集标签未知
        #         labels_int.append(-1)  # 使用 -1 作为占位符
        # # 对于测试集，通常使用从训练集加载的标签映射（如果预测时需要类别名转换）
        # # 这里我们仅返回空的或传入的映射，因为测试数据本身不用于生成映射
        # current_label_to_int = label_to_int_map_to_use if label_to_int_map_to_use is not None else {}
        # if current_label_to_int:
        #     current_int_to_label = {str(v): k for k, v in current_label_to_int.items()}
         print(f"信息: 正在从 '{data_dir}' 递归加载测试图像...")
         # 使用新的辅助函数来获取图像路径和文件名
         image_paths, image_filenames = find_image_files_recursively_for_test(data_dir)

         if not image_paths:
             print(f"警告: 在 '{data_dir}' 及其子目录中未找到任何图像文件。")
             # 返回一个包含预期列的空 DataFrame
             return pd.DataFrame(columns=['image_path', 'filename']), None, None

             # 创建 DataFrame，包含 image_path 和 filename 列
         df = pd.DataFrame({'image_path': image_paths, 'filename': image_filenames})

         # 对于测试集，我们不从数据生成标签映射
         print(f"信息: 成功加载 {len(df)} 个测试图像路径。")
         return df, None, None  # 返回 DataFrame，标签映射相关的返回 None

    else:  # 训练集或验证集
        if label_to_int_map_to_use:
            current_label_to_int = label_to_int_map_to_use
            current_int_to_label = {str(v): k for k, v in current_label_to_int.items()}  # 确保键是字符串
        else:
            # 从目录名生成标签映射
            label_id_counter = 0
            for class_name in sorted(os.listdir(data_dir)):
                class_dir_path = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir_path):  # 确保是目录
                    if class_name not in current_label_to_int:
                        current_label_to_int[class_name] = label_id_counter
                        current_int_to_label[str(label_id_counter)] = class_name  # JSON键最好是字符串
                        label_id_counter += 1

        if not current_label_to_int:
            print(f"警告: 在 '{data_dir}' 中未找到类别子目录，或提供的标签映射为空。")

        # 遍历类别子目录加载图像
        for class_name, label_id in current_label_to_int.items():
            class_dir_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir_path):
                for image_file in sorted(os.listdir(class_dir_path)):
                    if image_file.lower().endswith(Config.SUPPORTED_IMAGE_FORMATS):
                        full_path = os.path.join(class_dir_path, image_file)
                        image_paths.append(full_path)
                        original_filenames.append(image_file)
                        labels_str.append(class_name)
                        labels_int.append(label_id)
            else:
                if not label_to_int_map_to_use:  # Only warn if map wasn't provided (i.e., we expected this dir)
                    print(f"警告: 标签映射中的类别 '{class_name}' 在 '{data_dir}' 中没有对应的子目录。")

    if not image_paths:
        df = pd.DataFrame(columns=['image_path', 'label'])  # 'label' 通常指字符串标签
    else:
        df = pd.DataFrame({
            'image_path': image_paths,  # 完整路径
            'label': labels_str,  # 字符串形式的标签，如 'GuideSign'
            # 'original_filename': original_filenames, # 可选，如果其他地方需要
            # 'label_int': labels_int,                 # 可选，如果已经生成了，可以保留，但 WeatherDataset 通常会自己映射
        })

    print("DEBUG: Columns in df before returning from load_image_paths_and_labels (train/val):")
    print(df.columns)
    print("DEBUG: First 5 rows of df:")
    print(df.head())
    return df, current_label_to_int, current_int_to_label

    return df, current_label_to_int, current_int_to_label


def prepare_dataloaders(existing_label_to_int_map: dict = None):
    """
    准备训练和验证的DataLoader。
    如果提供了 existing_label_to_int_map，则使用它；否则从训练数据生成并保存。
    """
    Config.ensure_output_dirs()  # 确保输出目录存在，用于保存标签映射

    # 训练集的数据增强和基础转换
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE), # 使用Config中的图像大小
        # 在这里可以添加数据增强，例如：
        # transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD) # 使用Config中的均值和标准差
    ])
    # 验证集/测试集通常只有基础转换
    val_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])

    if not os.path.exists(Config.TRAIN_DIR) or not os.listdir(Config.TRAIN_DIR):
        print(f"错误: 训练目录 '{Config.TRAIN_DIR}' 为空或不存在。")
        return None, None, {}  # 返回空的数据加载器和标签映射

    if existing_label_to_int_map:
        print("信息: 使用已提供的标签映射。")
        label_to_int = existing_label_to_int_map
        # 从 label_to_int 生成 int_to_label，确保键是字符串
        int_to_label = {str(v): k for k, v in label_to_int.items()}
        # 使用提供的映射加载训练数据
        full_train_df, _, _ = load_image_paths_and_labels(
            Config.TRAIN_DIR,
            label_to_int_map_to_use=label_to_int,
            is_test_set=False  # 明确这是训练数据加载
        )
    else:
        print(f"信息: 从训练目录 '{Config.TRAIN_DIR}' 生成新的标签映射。")
        full_train_df, label_to_int, int_to_label = load_image_paths_and_labels(
            Config.TRAIN_DIR,
            is_test_set=False  # 明确这是训练数据加载
        )
        if not label_to_int:  # 如果没有成功生成标签映射
            print(f"错误: 未能从 '{Config.TRAIN_DIR}' 生成标签。无法继续。")
            return None, None, {}

        # 保存新生成的标签映射
        try:
            with open(Config.LABEL_MAP_FILE, 'w', encoding='utf-8') as f:
                json.dump({'label_to_int': label_to_int, 'int_to_label': int_to_label}, f, ensure_ascii=False, indent=4)
            print(f"信息: 新的标签映射已保存到 '{Config.LABEL_MAP_FILE}'。")
        except IOError as e:
            print(f"错误: 无法保存标签映射到 '{Config.LABEL_MAP_FILE}': {e}。")
            # 即使保存失败，如果 full_train_df 和 label_to_int 有内容，也许还能继续（但不推荐）
            # return None, None, {} # 或者选择在这里终止

    if full_train_df.empty:
        print(f"错误: 从 '{Config.TRAIN_DIR}' 加载的训练数据为空。")
        return None, None, label_to_int  # 返回标签映射，即使数据为空

    if 'label' in full_train_df.columns and label_to_int:
        full_train_df['label_int'] = full_train_df['label'].map(label_to_int)
        # 处理可能出现的NaN (如果有些label不在label_to_int中)
        full_train_df.dropna(subset=['label_int'], inplace=True)
        full_train_df['label_int'] = full_train_df['label_int'].astype(int)
    else:
        if 'label_int' not in full_train_df.columns:
            print("警告: full_train_df 中缺少 'label' 或 'label_int' 列，可能影响分层抽样。")

    # 更新Config中的类别数
    Config.NUM_CLASSES = len(label_to_int)
    if Config.NUM_CLASSES == 0:
        print("错误: 类别数为0。请检查数据和标签映射。")
        return None, None, label_to_int  # 返回标签映射

    print(f"信息: 共找到/使用 {Config.NUM_CLASSES} 个类别。")
    print(f"信息: 标签到整数的映射: {label_to_int}")

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # 根据 Config.VALIDATION_SPLIT_SIZE 划分训练集和验证集
    if 0 < Config.VALIDATION_SPLIT_SIZE < 1:  # 按比例划分
        # 确保 'label_int' 列存在且没有NaN值才能进行分层抽样
        can_stratify = ('label_int' in full_train_df.columns and
                        not full_train_df['label_int'].isnull().any() and
                        Config.NUM_CLASSES > 1 and
                        full_train_df['label_int'].value_counts().min() >= 2)  # 每类至少2个样本才能分层

        if can_stratify:
            try:
                train_df, val_df = train_test_split(
                    full_train_df,
                    test_size=Config.VALIDATION_SPLIT_SIZE,
                    random_state=Config.RANDOM_SEED,
                    stratify=full_train_df['label_int']
                )
                print(f"信息: 数据已划分: {len(train_df)} 训练样本, {len(val_df)} 验证样本 (已分层)。")
            except ValueError as e:  # 分层抽样可能失败
                print(f"警告: 分层抽样失败 ({e})。将使用非分层抽样。")
                train_df, val_df = train_test_split(
                    full_train_df,
                    test_size=Config.VALIDATION_SPLIT_SIZE,
                    random_state=Config.RANDOM_SEED
                )
                print(f"信息: 数据已划分: {len(train_df)} 训练样本, {len(val_df)} 验证样本 (未分层)。")
        else:
            print("警告: 不满足分层抽样条件。将使用非分层抽样。")
            train_df, val_df = train_test_split(
                full_train_df,
                test_size=Config.VALIDATION_SPLIT_SIZE,
                random_state=Config.RANDOM_SEED
            )
            print(f"信息: 数据已划分: {len(train_df)} 训练样本, {len(val_df)} 验证样本 (未分层)。")

    elif Config.VALIDATION_SPLIT_SIZE >= 1:  # 按绝对数量划分
        val_abs_size = int(Config.VALIDATION_SPLIT_SIZE)
        if val_abs_size < len(full_train_df):
            # 同样可以考虑分层，逻辑同上
            train_df, val_df = train_test_split(
                full_train_df,
                test_size=val_abs_size,
                random_state=Config.RANDOM_SEED
                # stratify=full_train_df['label_int'] if can_stratify else None
            )
            print(f"信息: 数据已划分: {len(train_df)} 训练样本, {len(val_df)} 验证样本 (按绝对数量)。")
        else:
            print(f"警告: 请求的验证集大小 ({val_abs_size}) 大于或等于总样本数 ({len(full_train_df)})。"
                  "所有数据将用于训练，不进行验证集划分。")
            train_df = full_train_df
            # val_df 保持为空
    else:  # 不进行验证集划分
        train_df = full_train_df
        # val_df 保持为空
        print("信息: 未进行验证集划分。所有数据将用于训练。")

    # 创建训练数据加载器
    train_loader = None
    if not train_df.empty:
        train_dataset = WeatherDataset(
            train_df,
            label_to_int_map=label_to_int,
            for_training=True,
            is_test_set=False,
            transform=train_transform  # <--- 添加 transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True
        )
        print(f"信息: 训练数据加载器已创建，包含 {len(train_dataset)} 个样本。")
    else:
        print("错误: 划分后的训练DataFrame为空。无法创建训练数据加载器。")

    # 创建验证数据加载器
    val_loader = None
    if not val_df.empty:
        val_dataset = WeatherDataset(
            val_df,
            label_to_int_map=label_to_int,
            for_training=False,
            is_test_set=False,
            transform=val_transform  # <--- 添加 transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            drop_last=False
        )
        print(f"信息: 验证数据加载器已创建，包含 {len(val_dataset)} 个样本。")
    else:
        print("信息: 验证DataFrame为空，未创建验证数据加载器。")

    return train_loader, val_loader, label_to_int


if __name__ == '__main__':
    import shutil
    import numpy as np
    from PIL import Image # 确保 Image 被导入，因为 create_dummy_image 会用到

    print("--- [dataset_utils.py 简化自测开始 (假设只有JPG)] ---")

    # --- 0. 准备测试环境和临时Config ---
    print("--- 步骤 0: 设置测试环境 ---")
    _original_config_values = {
        k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and k.isupper()
    }

    TEMP_TEST_ROOT_DIR = "./temp_dataset_utils_test_jpg_only"
    TEMP_TRAIN_DIR = os.path.join(TEMP_TEST_ROOT_DIR, "train_data")
    TEMP_TEST_DIR = os.path.join(TEMP_TEST_ROOT_DIR, "test_data_jpg") # 用于测试 load_image_paths_and_labels 测试模式
    TEMP_OUTPUT_DIR = os.path.join(TEMP_TEST_ROOT_DIR, "output_data")
    TEMP_LABEL_MAP_FILE = os.path.join(TEMP_OUTPUT_DIR, "temp_label_map_jpg.json")

    if os.path.exists(TEMP_TEST_ROOT_DIR):
        shutil.rmtree(TEMP_TEST_ROOT_DIR)
    os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)
    os.makedirs(TEMP_TEST_DIR, exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    # 临时覆盖 Config 值
    Config.TRAIN_DIR = TEMP_TRAIN_DIR
    Config.LABEL_MAP_FILE = TEMP_LABEL_MAP_FILE
    Config.OUTPUT_DIR = TEMP_OUTPUT_DIR
    Config.CROP_SIZE = (64, 64) # 使用较小的尺寸以加快测试
    Config.RESIZE_SHAPE = (72, 72)
    Config.NORM_MEAN = [0.5, 0.5, 0.5] # 简单均值
    Config.NORM_STD = [0.5, 0.5, 0.5]   # 简单标准差
    Config.SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg') # 明确指定只支持JPG
    Config.BATCH_SIZE = 2
    Config.VALIDATION_SPLIT_SIZE = 0.2 # 或者设为0，如果不测试验证集分割
    Config.RANDOM_SEED = 42
    Config.NUM_WORKERS = 0

    def create_dummy_jpg_image(path, size=(72, 72), color_value=128):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 创建一个单通道灰度图，然后转为RGB再保存为JPG，简化颜色处理
        img_array = np.full((size[1], size[0]), color_value, dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L').convert('RGB')
        img.save(path, format='JPEG') # 明确保存为JPEG

    # --- 1. 测试 load_image_paths_and_labels ---
    print("--- 步骤 1: 测试 load_image_paths_and_labels (JPG Only) ---")
    # 1.1 为训练模式创建数据
    os.makedirs(os.path.join(TEMP_TRAIN_DIR, "cloudy"), exist_ok=True)
    os.makedirs(os.path.join(TEMP_TRAIN_DIR, "sunny"), exist_ok=True)
    create_dummy_jpg_image(os.path.join(TEMP_TRAIN_DIR, "cloudy", "c1.jpg"), color_value=100)
    create_dummy_jpg_image(os.path.join(TEMP_TRAIN_DIR, "cloudy", "c2.jpg"), color_value=110)
    create_dummy_jpg_image(os.path.join(TEMP_TRAIN_DIR, "sunny", "s1.jpg"), color_value=200)

    print("测试 load_image_paths_and_labels (训练模式)...")
    train_df_loaded, map_l2i, map_i2l = load_image_paths_and_labels(TEMP_TRAIN_DIR, is_test_set=False)
    assert len(train_df_loaded) == 3, f"期望加载3个训练图像，实际: {len(train_df_loaded)}"
    assert len(map_l2i) == 2, f"期望生成2个类别，实际: {len(map_l2i)}"
    assert 'cloudy' in map_l2i and 'sunny' in map_l2i, "标签映射不正确"
    print(f"训练数据加载成功: {len(train_df_loaded)} 文件, 标签映射: {map_l2i}")

    # 1.2 为测试模式创建数据
    create_dummy_jpg_image(os.path.join(TEMP_TEST_DIR, "test1.jpg"), color_value=50)
    create_dummy_jpg_image(os.path.join(TEMP_TEST_DIR, "test2.jpg"), color_value=60)

    print("测试 load_image_paths_and_labels (测试模式)...")
    test_df_loaded, _, _ = load_image_paths_and_labels(TEMP_TEST_DIR, label_to_int_map_to_use=map_l2i, is_test_set=True)
    assert len(test_df_loaded) == 2, f"期望加载2个测试图像，实际: {len(test_df_loaded)}"
    assert all(test_df_loaded['original_filename'].str.endswith('.jpg')), "所有测试文件名应为.jpg"
    print(f"测试数据加载成功: {len(test_df_loaded)} 文件.")

    # --- 2. 测试 WeatherDataset ---
    print("--- 步骤 2: 测试 WeatherDataset (JPG Only) ---")
    # 2.1 训练模式
    print("测试 WeatherDataset (训练模式)...")
    train_dataset_instance = WeatherDataset(train_df_loaded, map_l2i, for_training=True, is_test_set=False)
    assert len(train_dataset_instance) == 3
    img_tensor_train, label_tensor_train = train_dataset_instance[0]
    # assert img_tensor_train.shape == (3, Config.CROP_SIZE[0], Config.CROP_SIZE[1]), f"训练图像尺寸错误: {img_tensor_train.shape}"
    assert label_tensor_train.dtype == paddle.int64
    print("WeatherDataset (训练模式) 通过.")

    # 2.2 测试模式
    print("测试 WeatherDataset (测试模式)...")
    test_dataset_instance = WeatherDataset(test_df_loaded, map_l2i, for_training=False, is_test_set=True)
    assert len(test_dataset_instance) == 2
    img_tensor_test, filename_test = test_dataset_instance[0]
    # assert img_tensor_test.shape == (3, Config.CROP_SIZE[0], Config.CROP_SIZE[1]), f"测试图像尺寸错误: {img_tensor_test.shape}"
    assert isinstance(filename_test, str) and filename_test.endswith(".jpg")
    print(f"WeatherDataset (测试模式) 通过. 返回文件名: {filename_test}")

    # 2.3 测试图像文件不存在的情况 (仍然重要)
    print("测试 WeatherDataset (图像文件不存在)...")
    # 创建一个包含一个存在和一个不存在文件的DataFrame
    df_with_missing = pd.DataFrame({
        'filename': [os.path.join(TEMP_TRAIN_DIR, "sunny", "s1.jpg"), "non_existent.jpg"],
        'original_filename': ["s1.jpg", "non_existent.jpg"], # original_filename 必须提供
        'label_str': ["sunny", "unknown"], # 确保label_str 和 label_int 对齐
        'label_int': [map_l2i["sunny"], -1]  # label_int 必须提供
    })
    # 测试训练模式下的缺失文件
    missing_train_dataset = WeatherDataset(df_with_missing, map_l2i, for_training=True, is_test_set=False)
    img_valid_train, _ = missing_train_dataset[0] # 存在的
    img_placeholder_train, _ = missing_train_dataset[1] # 不存在的
    assert img_valid_train.shape[1:] == Config.CROP_SIZE
    assert img_placeholder_train.shape[1:] == Config.CROP_SIZE, "占位符图像尺寸应为CROP_SIZE"
    print("WeatherDataset (训练模式，部分文件缺失) 处理通过.")

    # 测试测试模式下的缺失文件
    df_test_missing_only = pd.DataFrame({
        'filename': ["non_existent_test.jpg"],
        'original_filename': ["non_existent_test.jpg"]
        # 测试集不需要label_str, label_int
    })
    missing_test_dataset = WeatherDataset(df_test_missing_only, map_l2i, for_training=False, is_test_set=True)
    img_placeholder_test, fname_error = missing_test_dataset[0]
    assert img_placeholder_test.shape[1:] == Config.CROP_SIZE
    assert fname_error.endswith("_LOAD_ERROR"), "测试集加载失败文件名应有 _LOAD_ERROR 后缀"
    print("WeatherDataset (测试模式，文件缺失) 处理通过.")


    # --- 3. 测试 prepare_dataloaders (如果需要) ---
    if Config.VALIDATION_SPLIT_SIZE > 0: # 只在需要验证集分割时测试
        print("--- 步骤 3: 测试 prepare_dataloaders (JPG Only) ---")
        Config.TRAIN_DIR = TEMP_TRAIN_DIR # 确保指向正确的临时训练目录
        train_loader, val_loader, generated_map_dl = prepare_dataloaders(existing_label_to_int_map=None)
        assert train_loader is not None, "训练DataLoader不应为None"
        assert val_loader is not None, "验证DataLoader不应为None"
        assert len(generated_map_dl) == 2, f"prepare_dataloaders应识别2个类别, 实际: {len(generated_map_dl)}"
        assert os.path.exists(Config.LABEL_MAP_FILE), "标签映射文件应已创建"
        # 简单的DataLoader迭代测试
        for i, (batch_images, batch_labels_or_fnames) in enumerate(train_loader):
            assert batch_images.shape[0] <= Config.BATCH_SIZE
            assert batch_images.shape[1:] == (3, Config.CROP_SIZE[0], Config.CROP_SIZE[1])
            print(f"Train loader batch {i} OK.")
            if i >= 1: break # 测试少量批次即可
        for i, (batch_images, batch_labels_or_fnames) in enumerate(val_loader):
            assert batch_images.shape[0] <= Config.BATCH_SIZE
            print(f"Val loader batch {i} OK.")
            if i >= 1: break
        print("prepare_dataloaders 测试通过.")
    else:
        print("\n--- 步骤 3: 跳过 prepare_dataloaders (VALIDATION_SPLIT_SIZE <= 0) ---")


    # --- 4. 清理测试环境 ---
    print("--- 步骤 4: 清理测试环境 ---")
    for key, value in _original_config_values.items(): # 恢复Config
        setattr(Config, key, value)
    if os.path.exists(TEMP_TEST_ROOT_DIR):
        shutil.rmtree(TEMP_TEST_ROOT_DIR)
        print(f"临时测试目录 '{TEMP_TEST_ROOT_DIR}' 已删除。")

    print("--- [dataset_utils.py 简化自测结束] ---")
