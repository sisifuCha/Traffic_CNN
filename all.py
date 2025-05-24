import io, cv2
import math, json
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

train_json = pd.read_json('train.json')
# pd: 这是 pandas 库的常用别名。Pandas 是一个非常强大的数据分析和处理库。
# read_json('train.json'): 这是 Pandas 提供的一个函数，用于读取 JSON (JavaScript Object Notation) 文件。
# 'train.json': 是要读取的 JSON 文件的路径或文件名。
# 作用: 这行代码会读取名为 train.json 的文件，并将其内容解析为一个 Pandas DataFrame。DataFrame 是一种二维的、
# 表格状的数据结构，类似于电子表格或 SQL 表。train_json 这个变量现在就指向了这个 DataFrame。
train_json = train_json.sample(frac=1.0)
# train_json.sample(...): 这是 DataFrame 的一个方法，用于从 DataFrame 中随机抽取样本。
# frac=1.0: frac 参数指定了要抽取的行的比例。1.0 表示抽取所有行 (100%)。
# 作用: 这行代码实际上是对整个 DataFrame train_json 进行随机打乱（shuffle）操作。
# 打乱数据对于训练机器学习模型通常是有益的，可以防止模型学到数据顺序带来的偏差。


# 读取数据集
train_json['filename'] = train_json['annotations'].apply(lambda x: x['filename'])
# train_json['annotations']: 这表示选择 train_json DataFrame 中名为 "annotations" 的那一列。
# 这一列的内容很可能是字典或者包含字典的列表。
# .apply(lambda x: x['filename']): 这是 Pandas Series (DataFrame 的一列就是一个 Series) 的一个方法，
# 它会将一个函数应用于 Series 中的每一个元素。
# lambda x: x['filename']: 这是一个匿名函数 (lambda function)。
# x: 代表 "annotations" 列中的每一个元素（假设每个元素是一个字典）。
# x['filename']: 从这个字典 x 中提取键为 'filename' 的值。
# train_json['filename'] = ...: 将 apply 方法返回的结果（一个包含所有提取出的文件名的新 Series）
# 赋值给 train_json DataFrame 中一个名为 "filename" 的新列。如果 "filename" 列已存在，它将被覆盖。
# 作用: 这行代码从 "annotations" 列的每个元素（字典）中提取出 'filename' 对应的值，
# 并创建一个新的名为 "filename" 的列来存储这些文件名。
train_json['label'] = train_json['annotations'].apply(lambda x: x['label'])
train_json.head()
# train_json.head(): 这是 DataFrame 的一个方法，用于显示 DataFrame 的前几行（默认是前 5 行）。
# 作用: 这行代码通常用于快速查看 DataFrame 的结构和内容，以确认前面的操作是否符合预期。


# 定义自定义数据集类
class WeatherDataset(Dataset):
    def __init__(self, df):
        # df: 是传递给构造函数的参数，预期是一个PandasDataFrame(包含了文件名和标签信息)。
        super(WeatherDataset, self).__init__()
        self.df = df

        # 数据扩增方法
        self.transform = T.Compose([
            T.Resize(size=(128, 128)),
            T.RandomCrop(size=(125, 125)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

    def __getitem__(self, index):
        # 这是一个特殊方法。如果一个类定义了这个方法，那么它的实例就可以像列表或字典一样使用方括号[]
        # 来获取元素（例如dataset_instance[i]）。对于Dataset子类，这个方法是必须实现的，它负责根据索引
        # index返回一个数据样本。
        file_name = self.df['filename'].iloc[index]
        img = Image.open(file_name)
        img = self.transform(img)
        return img, paddle.to_tensor(self.df['label'].iloc[index])

    def __len__(self):
        return len(self.df)

# 训练集
train_dataset = WeatherDataset(train_json.iloc[:-500])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 验证集
val_dataset = WeatherDataset(train_json.iloc[-500:])
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)



from paddle.vision.models import resnet18
# 定义模型
class WeatherModel(paddle.nn.Layer):
    # 表明 WeatherModel 类继承自 paddle.nn.Layer。
    # 在 PaddlePaddle 中，所有自定义的神经网络模型（或自定义层）都必须是 paddle.nn.Layer 的子类。
    # 这个基类提供了管理参数、子层和模型状态（如训练/评估模式）等基本功能。
    def __init__(self):
        super(WeatherModel, self).__init__()
        backbone = resnet18(pretrained=True)
        # 意味着 resnet18 模型将使用在大型数据集（通常是 ImageNet）上预训练过的权重进行初始化。
        # 使用预训练权重是一种常见且有效的迁移学习技术，它能让你的模型利用从海量数据中学到的特征，
        # 通常能加速收敛并提升性能，尤其是在你的特定数据集相对较小的情况下。
        backbone.fc = paddle.nn.Identity()
        # 通常有一个最终的全连接层（常命名为 fc），用于执行原始任务的分类（例如 ImageNet 的 1000 个类别）。
        # 这里，backbone.fc 访问了加载的 ResNet-18 的这个最终层。
        # 这是一个特殊的层，它只是将其输入原封不动地传递到其输出，不进行任何转换。
        # 作用: 通过将 paddle.nn.Identity() 赋值给 backbone.fc，
        # 你实际上移除了 ResNet-18 原始的分类头。现在 backbone 的输出将是原始 fc 层之前那一层产生的特征向量
        # （对于 ResNet-18 来说，这是一个 512 维的向量）。这是将预训练模型用作特征提取器时的常用策略。
        self.backbone = backbone
        self.fc1 = paddle.nn.Linear(512, 10)
        # paddle.nn.Linear(in_features, out_features): 创建一个新的全连接（线性）层。
        # 512: in_features，输入特征数。这是该层输入特征向量的大小。它必须与 self.backbone 的输出大小（即 512）相匹配。
        # 10: out_features，输出单元数。这是这个线性层的输出单元数量。它应该对应于你特定问题中的类别数量（例如，如果你要对 10 种天气类别进行分类，这里就是 10）。
        # 作用: 这在 ResNet-18 特征提取器的基础上添加了一个新的分类头，专门用于你特定任务的类别数量。

    def forward(self, x):
        out = self.backbone(x)
        logits1 = self.fc1(out)
        return logits1

model = WeatherModel()
model(paddle.to_tensor(np.random.rand(10, 3, 125, 125).astype(np.float32)))




#标签分析
train_json['label'].value_counts()
#数据可视化
plt.figure(figsize=(10, 10))
# plt: matplotlib.pyplot 的别名。
# figure(): 创建一个新的 Matplotlib 图形，它是所有绘图元素的顶层容器。
# figsize=(10, 10): 以英寸为单位设置图形的宽度和高度。
for idx in range(10):
    plt.subplot(1, 10, idx+1)
    img = cv2.imread(train_json['filename'].iloc[idx])
    plt.imshow(img)
    plt.xticks([]); plt.yticks([])

# 优化器与损失函数
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0001)
# 创建一个 Adam 优化器的实例。Adam 是一种流行且有效的优化算法，它可以为每个参数自适应地调整学习率。
# parameters=model.parameters(): 这是一个关键参数。model.parameters() 是由 paddle.nn.Layer（WeatherModel 继承自该类）提供的一个方法，它返回模型中所有可训练参数（权重和偏置）的迭代器。优化器需要知道它应该更新哪些参数。
# learning_rate=0.0001: 设置 Adam 优化器的初始学习率。学习率控制模型参数在每次更新时响应估计误差的调整幅度。
criterion = paddle.nn.CrossEntropyLoss()
# 建一个交叉熵损失函数的实例。

# 模型训练与验证
for epoch in range(0, 4):
    Train_Loss, Val_Loss = [], []
    Train_ACC1 = []
    Val_ACC1 = []

    model.train()
    # 将 model 设置为训练模式。这很重要，因为一些层（如 Dropout 和 BatchNorm）在训练和评估期间的行为不同。
    # 例如，Dropout 在训练期间激活以防止过拟合，但在评估期间不激活。
    # BatchNorm 在训练期间使用批次统计数据，在评估期间使用运行（总体）统计数据。
    for i, (x, y1) in enumerate(train_loader):
        pred1 = model(x)
        loss = criterion(pred1, y1)
        Train_Loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        Train_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())

    model.eval()
    for i, (x, y1) in enumerate(val_loader):
        pred1 = model(x)
        loss = criterion(pred1, y1)
        Val_Loss.append(loss.item())
        Val_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())

    if epoch % 1 == 0:
        print(f'\nEpoch: {epoch}')
        print(f'Loss {np.mean(Train_Loss):3.5f}/{np.mean(Val_Loss):3.5f}')
        print(f'ACC {np.mean(Train_ACC1):3.5f}/{np.mean(Val_ACC1):3.5f}')
# 模型预测
import glob
test_df = pd.DataFrame({'filename': glob.glob('./test/*/*.jpg')})
test_df['label'] = 0
test_df = test_df.sort_values(by='filename')
test_dataset = WeatherDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model.eval()
pred = []

# ---模型预测
for i, (x, y1) in enumerate(test_loader):
    pred1 = model(x)
    pred += pred1.argmax(1).numpy().tolist()

test_df['label'] = pred
test_df['label'].value_counts()


submit_json = {
    'annotations':[]
}

# 生成提交结果文件
for row in test_df.iterrows():
    submit_json['annotations'].append({
        'filename': 'test_images/' + row[1].filename.split('/')[-1],
        'label': row[1].label,
    })

with open('submit.json', 'w') as up:
    json.dump(submit_json, up)
