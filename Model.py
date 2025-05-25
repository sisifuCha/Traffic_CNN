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
        elif self.model_name == "vgg16":
            self.backbone = models.vgg16(pretrained=self.pretrained_flag)
            # VGG 的分类器是一个 nn.Sequential，通常名为 'classifier'
            # 我们需要替换的是这个 Sequential 内部的最后一个 Linear 层
            # (6): Linear(in_features=4096, out_features=1000, ...) <--- 我们要替换这个
            classifier_attr_name = "classifier"
            # VGG16 最后一个 Linear 层的期望输入特征数
            expected_in_features = 4096
            print("VGG16 主干网络加载完毕。")
        elif self.model_name == "googlenet":  # <--- 新增 GoogLeNet
            self.backbone = models.googlenet(pretrained=self.pretrained_flag, num_classes=1000)
            print("信息: GoogLeNet 主干网络加载完毕。")
            classifier_attr_name = "_fc_out"
            expected_in_features = 1024  # GoogLeNet fc 层的输入特征数
            print("GoogLeNet 主干网络加载完毕。")
        else:
            # 列出所有支持的模型名称，方便用户排查错误
            supported_models = ["resnet18", "resnet50", "vgg16"]  # <--- 应该包含 "googlenet"
            raise ValueError(f"不支持的模型名称: '{self.model_name}'. 支持的模型有: {', '.join(supported_models)}.")


        # --- 2. 获取原始的分类层/分类器模块 ---
        try:
            original_classifier_module = getattr(self.backbone, classifier_attr_name)
            print(f"调试: 成功获取属性 '{classifier_attr_name}'")
        except AttributeError:
            print(f"错误: 模型 {self.model_name} 没有名为 '{classifier_attr_name}' 的分类层/模块属性。")
            print(f"错误详情: {e}")
            print(f"提示: 请检查上面 'GoogLeNet 实例的属性列表' 的输出，找到正确的分类层名称并更新 WeatherModel.py。")
            raise  # 重新抛出异常，因为没有分类层无法继续

        # --- 3. 确定新分类器的输入特征数 (in_features) 和替换逻辑 ---
        in_features_found = None

        if self.model_name.startswith("resnet") or  self.model_name == "googlenet":
            # 对于 ResNet 系列, original_classifier_module 就是一个 nn.Linear
            original_classifier_layer = original_classifier_module
            # 首先尝试从 .in_features 属性获取
            if hasattr(original_classifier_layer, 'in_features'):
                try:
                    in_features_found = original_classifier_layer.in_features
                    print(
                            f"成功通过属性 'original_classifier_layer.in_features' 获取到 in_features: {in_features_found}")
                except Exception as e_attr:
                    print(f"尝试访问 original_classifier_layer.in_features 时出错: {e_attr}")

            if in_features_found is None and hasattr(original_classifier_layer,
                                                         'weight') and original_classifier_layer.weight is not None:
                print(
                        f"  原始权重形状 (original_classifier_layer.weight.shape): {original_classifier_layer.weight.shape}")
                try:
                    if len(original_classifier_layer.weight.shape) == 2:
                        in_features_found = original_classifier_layer.weight.shape[0]
                        print(
                                f"  从 original_classifier_layer.weight.shape[0] 推断得到的 in_features: {in_features_found}")
                    else:
                        print(f"  权重形状不是2维，无法推断 in_features。")
                except Exception as e_infer:
                    print(f"  从权重推断 in_features (shape[0]) 失败: {e_infer}")

            final_in_features = in_features_found if in_features_found is not None else expected_in_features
            if in_features_found is not None and final_in_features != expected_in_features:
                print(
                        f"信息: 动态获取到的 {self.model_name} in_features ({final_in_features}) 与硬编码的预期值 ({expected_in_features}) 不同。将使用动态获取的值。")
            elif in_features_found is None:
                print(
                        f"警告: 未能通过属性或权重形状确定 in_features。对于 {self.model_name}，将回退到已知的预期值: {expected_in_features}。")

            print(
                    f"准备将 {self.model_name} 的 {classifier_attr_name} 层替换为 nn.Linear(in_features={final_in_features}, out_features={self.num_classes})")
            new_classifier = nn.Linear(final_in_features, self.num_classes)
            setattr(self.backbone, classifier_attr_name, new_classifier)

        elif self.model_name.startswith("vgg"):
            # 对于 VGG 系列, original_classifier_module 是一个 nn.Sequential
            # 我们需要找到这个 Sequential 内部的最后一个 nn.Linear 层
            last_linear_layer_index = -1
            for i, layer in reversed(list(enumerate(original_classifier_module))):
                if isinstance(layer, nn.Linear):
                    last_linear_layer_index = i
                    break

            if last_linear_layer_index == -1:
                raise ValueError(f"在 {self.model_name} 的 '{classifier_attr_name}' 模块中未找到 nn.Linear 层。")

            original_last_linear_layer = original_classifier_module[last_linear_layer_index]

            # 获取最后一个 Linear 层的输入特征
            if hasattr(original_last_linear_layer, 'in_features'):
                try:
                    in_features_found = original_last_linear_layer.in_features
                    print(
                            f"成功通过属性 'original_last_linear_layer.in_features' 获取到 in_features: {in_features_found}")
                except Exception as e_attr:
                    print(f"尝试访问 original_last_linear_layer.in_features 时出错: {e_attr}")

            if in_features_found is None and hasattr(original_last_linear_layer,
                                                         'weight') and original_last_linear_layer.weight is not None:
                print(
                        f"  VGG 最后一个 Linear 层的原始权重形状 (weight.shape): {original_last_linear_layer.weight.shape}")
                try:
                    if len(original_last_linear_layer.weight.shape) == 2:
                        in_features_found = original_last_linear_layer.weight.shape[0]
                        print(
                                f"  从 VGG 最后一个 Linear 层的 weight.shape[0] 推断得到的 in_features: {in_features_found}")
                    else:
                            print(f"  VGG 最后一个 Linear 层的权重形状不是2维，无法推断 in_features。")
                except Exception as e_infer:
                    print(f"  从 VGG 最后一个 Linear 层的权重推断 in_features (shape[0]) 失败: {e_infer}")

            final_in_features = in_features_found if in_features_found is not None else expected_in_features
            if in_features_found is not None and final_in_features != expected_in_features:
                print(
                        f"信息: 动态获取到的 {self.model_name} (最后一个Linear层) in_features ({final_in_features}) 与硬编码的预期值 ({expected_in_features}) 不同。将使用动态获取的值。")
            elif in_features_found is None:
                print(
                        f"警告: 未能通过属性或权重形状确定 {self.model_name} (最后一个Linear层) in_features。将回退到已知的预期值: {expected_in_features}。")

            print(
                    f"准备将 {self.model_name} 的 '{classifier_attr_name}' 模块中索引为 {last_linear_layer_index} 的 Linear 层替换为 nn.Linear(in_features={final_in_features}, out_features={self.num_classes})")
            # 创建新的 Linear 层
            new_last_linear_layer = nn.Linear(final_in_features, self.num_classes)
            # 替换 Sequential 中的层
            original_classifier_module[last_linear_layer_index] = new_last_linear_layer
            # 将修改后的 Sequential 重新赋值给 backbone 的 classifier 属性 (虽然原地修改通常也生效)
            setattr(self.backbone, classifier_attr_name, original_classifier_module)

        # --- 5. 调试打印 ---
        print(f"--- 模型 {self.model_name} 初始化完毕 (pretrained={self.pretrained_flag}) ---")
        current_classifier_wrapper = getattr(self.backbone, classifier_attr_name)

        if self.model_name.startswith("resnet")or self.model_name == "googlenet":
            print(f"新的分类器 (self.backbone.{classifier_attr_name}): {current_classifier_wrapper}")
            try:
                print(f"  新分类器的输入特征数 (尝试属性): {current_classifier_wrapper.in_features}")
            except AttributeError:
                if hasattr(current_classifier_wrapper, 'weight') and current_classifier_wrapper.weight is not None:
                    print(
                            f"  新分类器的输入特征数 (从权重 shape[0] 推断): {current_classifier_wrapper.weight.shape[0]}")
            try:
                print(f"  新分类器的输出类别数 (尝试属性): {current_classifier_wrapper.out_features}")
            except AttributeError:
                if hasattr(current_classifier_wrapper, 'weight') and current_classifier_wrapper.weight is not None:
                    print(
                            f"  新分类器的输出类别数 (从权重 shape[1] 推断): {current_classifier_wrapper.weight.shape[1]}")

        elif self.model_name.startswith("vgg"):
            print(f"新的分类器模块 (self.backbone.{classifier_attr_name}):")
            # 打印 VGG classifier 模块的结构
            for i, layer in enumerate(current_classifier_wrapper):
                print(f"  ({i}): {layer}")
            # 特别打印我们替换的最后一个 Linear 层的信息
            final_linear_layer_in_vgg = None
            for layer in reversed(list(current_classifier_wrapper)):  # 从后往前找最后一个Linear
                if isinstance(layer, nn.Linear):
                    final_linear_layer_in_vgg = layer
                    break
            if final_linear_layer_in_vgg:
                print(f"  VGG 中最终的 Linear 分类器: {final_linear_layer_in_vgg}")
                try:
                    print(f"    输入特征数 (尝试属性): {final_linear_layer_in_vgg.in_features}")
                except AttributeError:
                    if hasattr(final_linear_layer_in_vgg,
                                   'weight') and final_linear_layer_in_vgg.weight is not None:
                        print(f"    输入特征数 (从权重 shape[0] 推断): {final_linear_layer_in_vgg.weight.shape[0]}")
                try:
                    print(f"    输出类别数 (尝试属性): {final_linear_layer_in_vgg.out_features}")
                except AttributeError:
                    if hasattr(final_linear_layer_in_vgg,
                                   'weight') and final_linear_layer_in_vgg.weight is not None:
                        print(f"    输出类别数 (从权重 shape[1] 推断): {final_linear_layer_in_vgg.weight.shape[1]}")

        print(f"期望输出类别数: {self.num_classes}")
        print(f"--- 模型初始化结束 ---")

    def forward(self, x):
        output = self.backbone(x)

        if isinstance(output, tuple) and len(output) > 0:  # 检查是否是元组并且非空
            if self.training and self.model_name == "googlenet":  # 只在训练时打印此警告
                # 这个警告在评估/预测时不需要，因为那时也可能返回元组但我们只用第一个
                pass  # 可以选择在这里打印一个更简洁的日志，或者不打印
                # print(f"信息 (GoogLeNet 前向传播): 模型返回了多个输出，将使用第一个（主）输出。")
            return output[0]  # 返回主输出
        return output

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
