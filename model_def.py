# model_def.py
import paddle
import paddle.nn as nn
from paddle.vision.models import resnet18
from config import Config  # 导入配置

import paddle
import paddle.nn as nn
from paddle.vision.models import resnet18


# # from config import Config # 假设 Config.PRETRAINED_MODEL 已定义
#
# class WeatherModel(nn.Layer):
#     def __init__(self, num_classes, pretrained_flag=True):  # 直接传递预训练标志或从 Config 获取
#         super(WeatherModel, self).__init__()
#         if num_classes is None or num_classes <= 0:
#             raise ValueError("num_classes 必须是一个正整数。")
#
#         # 1. 加载预训练的 resnet18
#         print(f"尝试加载 ResNet18, pretrained={pretrained_flag}...")
#         self.backbone = resnet18(pretrained=pretrained_flag)  # 使用传入的标志
#         print("ResNet18 主干网络加载完毕。")
#
#         # 2. 获取原始 fc 层的输入特征维度
#         #    对于 ResNet18，这个值是 512。
#         #    标准方法是访问其 .in_features 属性。
#         original_fc_layer = self.backbone.fc
#         print(f"原始 ResNet18 FC 层: {original_fc_layer}")
#
#         # --- 简化和修正的 in_features 获取逻辑 ---
#         in_features = None
#         if hasattr(original_fc_layer, 'in_features'):
#             try:
#                 in_features = original_fc_layer.in_features
#                 print(f"成功通过属性 'original_fc_layer.in_features' 获取到 in_features: {in_features}")
#             except Exception as e_attr:
#                 print(f"尝试访问 original_fc_layer.in_features 时出错: {e_attr}")
#
#         if in_features is None and hasattr(original_fc_layer, 'weight') and original_fc_layer.weight is not None:
#             # 正确方式: paddle.nn.Linear 的 weight.shape 是 (out_features, in_features)
#             # 所以, in_features 是 original_fc_layer.weight.shape[1]
#             print(f"未能从属性 'in_features' 获取。尝试从 'original_fc_layer.weight.shape[1]' 推断。")
#             print(f"  原始权重形状 (original_fc_layer.weight.shape): {original_fc_layer.weight.shape}")
#             try:
#                 if len(original_fc_layer.weight.shape) == 2:
#                     in_features = original_fc_layer.weight.shape[1]  # 使用索引 1
#                     print(f"  从 original_fc_layer.weight.shape[1] 推断得到的 in_features: {in_features}")
#                 else:
#                     print(f"  权重形状不是2维，无法从中推断 in_features。")
#             except Exception as e_infer:
#                 print(f"  从权重推断 in_features (shape[1]) 失败: {e_infer}")
#
#         if in_features is None:
#             # 如果上述方法都失败了，对于ResNet18，我们可以回退到已知值 512
#             # （尽管上述方法对于标准模型来说不应该失败）
#             print("警告: 未能通过属性或权重形状确定 in_features。对于ResNet18，将回退到已知的 512。")
#             in_features = 512
#
#             # 验证获取到的 in_features 是否合理（特别是对于已知的 ResNet18）
#         if pretrained_flag and in_features != 512:
#             # 如果是预训练的ResNet18，但获取到的in_features不是512，这通常是个问题。
#             print(f"警告: 对于预训练的ResNet18，获取到的 in_features ({in_features}) 不是预期的 512。")
#             print(f"       这可能表示模型加载或 PaddlePaddle 版本存在问题。")
#             print(f"       将强制使用 512 作为 in_features 以继续，但请检查您的设置。")
#             in_features = 512  # 强制修正为ResNet18的正确值
#
#         # 3. 替换原始的 fc 层。
#         #    选项 A (更常见和直接): 直接替换 self.backbone.fc
#         print(f"准备将 ResNet18 的 fc 层替换为 nn.Linear(in_features={in_features}, out_features={num_classes})")
#         self.backbone.fc = nn.Linear(in_features, num_classes)
#
#         #    选项 B (你当前的方法 - 如果 self.fc1 在 forward 中使用，也是有效的):
#         #    self.backbone.fc = nn.Identity() # 使原始 fc 层成为一个直通层
#         #    self.fc1 = nn.Linear(in_features, num_classes) # 添加你自己的新 fc 层
#
#         print(f"--- 模型初始化完毕 ---")
#         if hasattr(self, 'fc1') and isinstance(self.fc1, nn.Linear):  # 如果你使用选项 B
#             print(f"新的分类器 (self.fc1): {self.fc1}")
#             # 改动点：如果 .in_features 不可用，从权重形状获取
#             try:
#                 print(f"  新分类器的输入特征数 (尝试属性): {self.fc1.in_features}")
#             except AttributeError:
#                 if hasattr(self.fc1, 'weight') and self.fc1.weight is not None:
#                     print(
#                         f"  新分类器的输入特征数 (从权重推断): {self.fc1.weight.shape[1]}")  # weight.shape[1] is in_features
#                 else:
#                     print(f"  新分类器的输入特征数: 未知 (无法从属性或权重获取)")
#             # 同样可以为 out_features 做类似处理
#             # print(f"  新分类器的输出类别数: {self.fc1.out_features}")
#         elif isinstance(self.backbone.fc, nn.Linear):  # 如果你使用选项 A (我们现在的方案)
#             print(f"新的分类器 (self.backbone.fc): {self.backbone.fc}")
#             # 改动点：如果 .in_features 不可用，从权重形状获取
#             try:
#                 print(f"  新分类器的输入特征数 (尝试属性): {self.backbone.fc.in_features}")  # 这是出错的地方
#             except AttributeError:
#                 if hasattr(self.backbone.fc, 'weight') and self.backbone.fc.weight is not None:
#                     print(f"  新分类器的输入特征数 (从权重推断): {self.backbone.fc.weight.shape[1]}")  # <--- 使用这个
#                 else:
#                     print(f"  新分类器的输入特征数: 未知 (无法从属性或权重获取)")
#
#             # 对于 out_features，通常 .out_features 属性是可用的，或者可以从 weight.shape[0] 获取
#             try:
#                 print(f"  新分类器的输出类别数 (尝试属性): {self.backbone.fc.out_features}")
#             except AttributeError:
#                 if hasattr(self.backbone.fc, 'weight') and self.backbone.fc.weight is not None:
#                     print(f"  新分类器的输出类别数 (从权重推断): {self.backbone.fc.weight.shape[0]}")
#                 else:
#                     print(f"  新分类器的输出类别数: 未知 (无法从属性或权重获取)")
#         print(f"期望输出类别数: {num_classes}")
#         print(f"--- 模型初始化结束 ---")


import paddle
import paddle.nn as nn
from paddle.vision.models import resnet18


class WeatherModel(nn.Layer):
    def __init__(self, num_classes, pretrained_flag=True):  # 外部仍然可以传入 True
        super(WeatherModel, self).__init__()
        if num_classes is None or num_classes <= 0:
            raise ValueError("num_classes 必须是一个正整数。")

        # 使用（可能被强制修改的）pretrained_flag 加载 ResNet18
        print(f"尝试加载 ResNet18, pretrained={pretrained_flag}...")
        self.backbone = resnet18(pretrained=pretrained_flag)
        print("ResNet18 主干网络加载完毕。")

        # --- 后续获取 in_features 和替换 fc 层的逻辑保持不变 ---
        # (我们之前的讨论已经确认这部分逻辑在模型结构上是正确的，
        #  即最终 self.backbone.fc 会被替换为 nn.Linear(512, num_classes) )

        original_fc_layer = self.backbone.fc
        # print(f"原始 ResNet18 FC 层 (在 pretrained={pretrained_flag} 后): {original_fc_layer}")

        in_features = None
        # 尝试直接访问 .in_features (对于随机初始化的模型，这个属性应该存在且为512)
        if hasattr(original_fc_layer, 'in_features'):
            try:
                in_features = original_fc_layer.in_features
                print(f"成功通过属性 'original_fc_layer.in_features' 获取到 in_features: {in_features}")
            except Exception as e_attr:
                print(f"尝试访问 original_fc_layer.in_features 时出错: {e_attr}")

        # 如果直接访问失败，或为了处理之前的奇怪权重形状，保留从权重推断的逻辑，但调整为 (in, out) 约定
        if in_features is None and hasattr(original_fc_layer, 'weight') and original_fc_layer.weight is not None:
            print(f"未能从属性 'in_features' 获取。尝试从权重推断 (假定 shape=(in,out))。")
            print(f"  原始权重形状 (original_fc_layer.weight.shape): {original_fc_layer.weight.shape}")
            try:
                if len(original_fc_layer.weight.shape) == 2:
                    # 假设 weight.shape 是 (in_features, out_features)
                    in_features = original_fc_layer.weight.shape[0]
                    print(f"  从 original_fc_layer.weight.shape[0] 推断得到的 in_features: {in_features}")
                else:
                    print(f"  权重形状不是2维。")
            except Exception as e_infer:
                print(f"  从权重推断 in_features (shape[0]) 失败: {e_infer}")

        if in_features is None:  # 理论上随机初始化时， .in_features 应该能取到
            print("警告: 未能通过属性或权重形状确定 in_features。对于ResNet18，将回退到已知的 512。")
            in_features = 512

            # 对于随机初始化的 ResNet18，其 fc 层的 in_features 应该是 512
        if in_features != 512:
            print(f"警告: 获取到的 ResNet18 in_features ({in_features}) 不是预期的 512。")
            print(f"       对于随机初始化的模型，这不应该发生。将强制使用 512。")
            in_features = 512

        print(f"准备将 ResNet18 的 fc 层替换为 nn.Linear(in_features={in_features}, out_features={num_classes})")
        self.backbone.fc = nn.Linear(in_features, num_classes)

        print(f"--- 模型初始化完毕 (强制 pretrained={pretrained_flag}) ---")
        new_fc_layer = self.backbone.fc
        print(f"新的分类器 (self.backbone.fc): {new_fc_layer}")
        # 打印新层的 in_features 和 out_features (仍使用 (in,out) 约定从权重推断)
        try:
            print(f"  新分类器的输入特征数 (尝试属性): {new_fc_layer.in_features}")
        except AttributeError:
            if hasattr(new_fc_layer, 'weight') and new_fc_layer.weight is not None:
                print(f"  新分类器的输入特征数 (从权重 shape[0] 推断): {new_fc_layer.weight.shape[0]}")
        try:
            print(f"  新分类器的输出类别数 (尝试属性): {new_fc_layer.out_features}")
        except AttributeError:
            if hasattr(new_fc_layer, 'weight') and new_fc_layer.weight is not None:
                print(f"  新分类器的输出类别数 (从权重 shape[1] 推断): {new_fc_layer.weight.shape[1]}")
        print(f"期望输出类别数: {num_classes}")
        print(f"--- 模型初始化结束 ---")

    def forward(self, x):
        # 在 forward 中也加入调试打印
        print(f"--- WeatherModel.forward ---")
        print(f"  Input x shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
        print(f"  Type of self.backbone: {type(self.backbone)}")
        print(f"  Is self.backbone a paddle.nn.Layer? {isinstance(self.backbone, nn.Layer)}")
        # print(f"  Attributes of self.backbone (first 20): {dir(self.backbone)[:20]}") # 可以暂时注释掉，如果输出太多
        print(f"  Does self.backbone have 'forward' attribute? {hasattr(self.backbone, 'forward')}")
        if hasattr(self.backbone, 'forward'):
            print(f"    Is self.backbone.forward callable? {callable(self.backbone.forward)}")

        print(f"  Attempting to call self.backbone(x)...")
        try:
            output_backbone = self.backbone(x)
            print(f"  Successfully called self.backbone(x). Output shape: {output_backbone.shape}")
            return output_backbone
        except NotImplementedError as e_ni:
            print(f"  Caught NotImplementedError when calling self.backbone(x): {e_ni}")
            raise e_ni  # 重新抛出，让程序停止
        except Exception as e:
            print(f"  Caught other Exception when calling self.backbone(x): {e}")
            raise e  # 重新抛出


if __name__ == '__main__':
    print("模型定义模块 (model_def.py) 测试:")

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

    print("model_def.py 测试结束。")
