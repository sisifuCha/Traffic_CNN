# Model.py
import paddle
import paddle.nn as nn
import paddle.vision.models as models
from config import Config  # 导入配置

import paddle
import paddle.nn as nn
from paddle.vision.models import resnet18,resnet50


class WeatherModel(nn.Layer):
    def __init__(self, model_name="resnet18", num_classes=10,pretrained_flag=True):
        super(WeatherModel, self).__init__()

        self.model_name = model_name.lower()  # 保存模型名称以供参考
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.pretrained_flag = pretrained_flag

        # 使用（可能被强制修改的）pretrained_flag 加载 ResNet18
        if self.model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=self.pretrained_flag)
            classifier_attr_name = "fc"  # ResNet 系列的分类层属性名
            expected_in_features = 512  # ResNet18 fc 层期望的输入特征数
            print("ResNet18 主干网络加载完毕。")
        elif self.model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=self.pretrained_flag)
            classifier_attr_name = "fc"  # ResNet 系列的分类层属性名
            expected_in_features = 2048  # ResNet50 fc 层期望的输入特征数
            print("ResNet50 主干网络加载完毕。")
        else:
            # 列出所有支持的模型名称，方便用户排查错误
            supported_models = ["resnet18", "resnet50"]  # 当你添加新模型时，也在此处更新
            raise ValueError(f"不支持的模型名称: '{self.model_name}'. 支持的模型有: {', '.join(supported_models)}.")


        # --- 后续获取 in_features 和替换 fc 层的逻辑保持不变 ---
        # 最终 self.backbone.fc 会被替换为 nn.Linear(512, num_classes) )

        # --- 2. 获取原始的分类层 ---
        try:
            # 使用 getattr 动态获取分类层对象
            original_classifier_layer = getattr(self.backbone, classifier_attr_name)
        except AttributeError:
            print(f"错误: 模型 {self.model_name} 没有名为 '{classifier_attr_name}' 的分类层属性。")
            raise
        # --- 3. 确定新分类器的输入特征数 (in_features) ---
        in_features_found = None
        # 首先尝试从 .in_features 属性获取
        if hasattr(original_classifier_layer, 'in_features'):
            try:
                in_features_found = original_classifier_layer.in_features
                print(
                    f"成功通过属性 'original_classifier_layer.in_features' 获取到 in_features: {in_features_found}")
            except Exception as e_attr:
                print(f"尝试访问 original_classifier_layer.in_features 时出错: {e_attr}")

        # 如果属性获取失败，尝试从权重形状推断
        if in_features_found is None and hasattr(original_classifier_layer,
                                                     'weight') and original_classifier_layer.weight is not None:
            #print(f"未能从属性 'in_features' 获取。尝试从权重推断 (假定 shape=(in_features, out_features))。")
            print(
                f"  原始权重形状 (original_classifier_layer.weight.shape): {original_classifier_layer.weight.shape}")
            try:
                if len(original_classifier_layer.weight.shape) == 2:
                    # PaddlePaddle nn.Linear 的权重形状是 [in_features, out_features]
                    in_features_found = original_classifier_layer.weight.shape[0]
                    print(
                        f"  从 original_classifier_layer.weight.shape[0] 推断得到的 in_features: {in_features_found}")
                else:
                    print(f"  权重形状不是2维，无法推断 in_features。")
            except Exception as e_infer:
                print(f"  从权重推断 in_features (shape[0]) 失败: {e_infer}")

        # 使用探测到的 in_features，或者回退到期望值
        if in_features_found is not None:
            final_in_features = in_features_found
            if final_in_features != expected_in_features:
                # 这种情况可能是因为加载了不同变体或非标准预训练权重，或者模型定义本身就有差异
                print(
                    f"信息: 动态获取到的 {self.model_name} in_features ({final_in_features}) 与硬编码的预期值 ({expected_in_features}) 不同。")
                print(f"       将使用动态获取的值: {final_in_features}。这可能是由于不同的预训练权重或模型变体。")
        else:
            print(
                f"警告: 未能通过属性或权重形状确定 in_features。对于 {self.model_name}，将回退到已知的预期值: {expected_in_features}。")
            final_in_features = expected_in_features

        # --- 4. 替换分类层 ---
        print(
            f"准备将 {self.model_name} 的 {classifier_attr_name} 层替换为 nn.Linear(in_features={final_in_features}, out_features={self.num_classes})")
        new_classifier = nn.Linear(final_in_features, self.num_classes)
        # 使用 setattr 动态设置新的分类层
        setattr(self.backbone, classifier_attr_name, new_classifier)

        # --- 5. 调试打印 ---
        print(f"--- 模型 {self.model_name} 初始化完毕 (pretrained={self.pretrained_flag}) ---")
        current_classifier = getattr(self.backbone, classifier_attr_name)
        print(f"新的分类器 (self.backbone.{classifier_attr_name}): {current_classifier}")
        try:
            print(f"  新分类器的输入特征数 (尝试属性): {current_classifier.in_features}")
        except AttributeError:
            if hasattr(current_classifier, 'weight') and current_classifier.weight is not None:
                print(f"  新分类器的输入特征数 (从权重 shape[0] 推断): {current_classifier.weight.shape[0]}")
        try:
            print(f"  新分类器的输出类别数 (尝试属性): {current_classifier.out_features}")
        except AttributeError:
            if hasattr(current_classifier, 'weight') and current_classifier.weight is not None:
                print(f"  新分类器的输出类别数 (从权重 shape[1] 推断): {current_classifier.weight.shape[1]}")
        print(f"期望输出类别数: {self.num_classes}")
        print(f"--- 模型初始化结束 ---")

    def forward(self, x):
        # forward 方法直接使用已修改的主干网络
        try:
            return self.backbone(x)
        except Exception as e:
            # 加入模型名称，方便定位问题
            print(f"WeatherModel ({self.model_name}) 前向传播时发生错误: {e}")
            raise

if __name__ == '__main__':
    print("模型定义模块 (Model.py) 测试:")

    num_test_classes = Config.NUM_CLASSES if Config.NUM_CLASSES else 3
    # 根据你的错误日志，这里可能是10，如果Config.NUM_CLASSES被其他地方设置了
    # 如果想强制测试固定数量，可以直接写 num_test_classes = 10
    print(f"使用 {num_test_classes} 个类别实例化模型。")

    try:
        model = WeatherModel(num_classes=num_test_classes)
        print("模型 WeatherModel 实例化成功。")

        dummy_input = paddle.randn([2, 3, Config.CROP_SIZE[0], Config.CROP_SIZE[1]], dtype='float32')
        dummy_input = dummy_input.to(Config.DEVICE)
        model.to(Config.DEVICE)

        model.eval()
        with paddle.no_grad():
            output_logits = model(dummy_input)

        print(f"虚拟输入形状: {dummy_input.shape}")
        print(f"模型输出 Logits 形状: {output_logits.shape}")
        assert output_logits.shape == [2, num_test_classes], "模型输出形状不匹配！"
        print("模型前向传播测试通过。")

    except Exception as e:
        print(f"模型测试失败: {e}")

    print("Model.py 测试结束。")
