# RMBG 2.0 升级指南

## 概述

本指南将帮助您将 HivisionIDPhotos 项目从 RMBG 1.4 升级到 RMBG 2.0，并根据 ComfyUI 参数进行配置。

## RMBG 2.0 的改进

<mcreference link="https://huggingface.co/briaai/RMBG-2.0" index="1">1</mcreference> RMBG v2.0 是基于 BiRefNet 架构的新一代背景移除模型，相比 RMBG v1.4 有显著改进：

- **更高精度**：基于 BiRefNet (Bilateral Reference Network) 架构
- **更大训练集**：使用超过 15,000 张高质量、高分辨率的手工标注图像训练
- **更好的边缘处理**：双向参考机制提供更精确的边界分割
- **更强的泛化能力**：支持复杂背景和多对象场景

## 升级步骤

### 1. 下载 RMBG 2.0 模型

由于 RMBG 2.0 需要登录 HuggingFace 才能下载，您需要：

1. 访问 <mcreference link="https://huggingface.co/briaai/RMBG-2.0" index="1">1</mcreference>
2. 登录您的 HuggingFace 账户
3. 同意许可协议（仅限非商业用途）
4. 下载 ONNX 模型文件
5. 将模型文件重命名为 `rmbg-2.0.onnx`
6. 放置到 `hivision/creator/weights/` 目录下

### 2. 代码修改

项目已经完成了以下代码修改：

#### 2.1 添加模型配置
- 在 `human_matting.py` 中添加了 RMBG 2.0 的权重路径
- 添加了全局会话变量 `RMBG_2_SESS`

#### 2.2 新增处理函数
- `extract_human_rmbg_2()`: RMBG 2.0 的主要抠图函数
- `get_rmbg_2_matting()`: RMBG 2.0 的具体处理逻辑

#### 2.3 更新模型选择器
- 在 `choose_handler.py` 中添加了 "rmbg-2.0" 选项
- 添加了对应的处理器映射

### 3. ComfyUI 参数配置

根据您提供的 ComfyUI 参数截图，RMBG 2.0 的配置参数如下：

```python
# 模型配置
model_name = "RMBG-2.0"
sensitivity = 0.95  # 灵敏度
processing_resolution = 1024  # 处理分辨率
mask_blur = 0  # 遮罩模糊
mask_offset = 0  # 遮罩偏移
reverse_output = False  # 反转输出
optimize_foreground = True  # 精细前景优化
background_type = "Alpha"  # 背景类型
```

### 4. 使用方法

#### 4.1 在代码中使用

```python
from hivision.creator.choose_handler import choose_handler
from hivision.creator.context import Context

# 创建上下文
ctx = Context()
ctx.processing_image = your_image  # 您的输入图像

# 选择 RMBG 2.0 模型
creator = YourCreatorClass()
choose_handler(creator, matting_model_option="rmbg-2.0")

# 执行抠图
creator.matting_handler(ctx)

# 获取结果
result_image = ctx.processing_image
```

#### 4.2 在 Gradio 界面中使用

在模型选择下拉菜单中选择 "rmbg-2.0" 即可使用新模型。

### 5. 性能对比

| 特性 | RMBG 1.4 | RMBG 2.0 |
|------|----------|----------|
| 架构 | IS-Net | BiRefNet |
| 训练数据 | 12,000+ 图像 | 15,000+ 图像 |
| 输入分辨率 | 1024x1024 | 1024x1024 |
| 预处理 | [-1, 1] 归一化 | ImageNet 标准化 |
| 边缘质量 | 良好 | 优秀 |
| 复杂场景 | 一般 | 优秀 |

### 6. 注意事项

1. **许可证**：RMBG 2.0 使用 CC BY-NC 4.0 许可证，仅限非商业用途
2. **模型大小**：RMBG 2.0 模型文件较大，请确保有足够的存储空间
3. **性能要求**：建议使用 GPU 加速以获得最佳性能
4. **兼容性**：新模型与现有的 API 接口完全兼容

### 7. 故障排除

#### 7.1 模型加载失败
- 检查模型文件是否正确放置在 `hivision/creator/weights/` 目录
- 确认文件名为 `rmbg-2.0.onnx`
- 检查文件是否损坏

#### 7.2 内存不足
- 尝试使用 CPU 模式：设置环境变量 `ONNX_DEVICE=CPU`
- 减少批处理大小
- 关闭其他占用内存的程序

#### 7.3 精度问题
- 确保输入图像质量良好
- 尝试调整处理分辨率
- 检查图像预处理步骤

## 总结

RMBG 2.0 提供了更高的抠图精度和更好的边缘处理能力。通过本指南的步骤，您可以轻松升级到新版本并享受改进的性能。如果遇到问题，请参考故障排除部分或查看项目文档。