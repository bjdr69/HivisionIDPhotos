#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 21:21
@File: human_matting.py
@IDE: pycharm
@Description:
    人像抠图
"""
import numpy as np
from PIL import Image
import onnxruntime
from .tensor2numpy import NNormalize, NTo_Tensor, NUnsqueeze
from .context import Context
import cv2
import os
from time import time


WEIGHTS = {
    "hivision_modnet": os.path.join(
        os.path.dirname(__file__), "weights", "hivision_modnet.onnx"
    ),
    "modnet_photographic_portrait_matting": os.path.join(
        os.path.dirname(__file__),
        "weights",
        "modnet_photographic_portrait_matting.onnx",
    ),
    "mnn_hivision_modnet": os.path.join(
        os.path.dirname(__file__),
        "weights",
        "mnn_hivision_modnet.mnn",
    ),
    "rmbg-1.4": os.path.join(os.path.dirname(__file__), "weights", "rmbg-1.4.onnx"),
    "rmbg-2.0": os.path.join(os.path.dirname(__file__), "weights", "rmbg-2.0.onnx"),
    "birefnet-v1-lite": os.path.join(
        os.path.dirname(__file__), "weights", "birefnet-v1-lite.onnx"
    ),
}

ONNX_DEVICE = onnxruntime.get_device()
ONNX_PROVIDER = (
    "CUDAExecutionProvider" if ONNX_DEVICE == "GPU" and "CUDAExecutionProvider" in onnxruntime.get_available_providers() else "CPUExecutionProvider"
)
print(f"ONNX Device: {ONNX_DEVICE}, Provider: {ONNX_PROVIDER}, Available providers: {onnxruntime.get_available_providers()}")

# GPU idle timeout: release unused GPU model memory after inactivity
_LAST_GPU_USE = {}
IDLE_TIMEOUT = 300  # 5 minutes

def _check_gpu_idle(model_key):
    """Check if GPU model has been idle too long; release if so."""
    global RMBG_SESS, RMBG_2_SESS, BIREFNET_V1_LITE_SESS
    now = time()
    last = _LAST_GPU_USE.get(model_key, 0)
    if last > 0 and (now - last) > IDLE_TIMEOUT:
        if model_key == "rmbg" and RMBG_SESS is not None:
            print(f"[GPU Idle] Releasing rmbg model after {(now - last):.0f}s idle")
            RMBG_SESS = None
        elif model_key == "rmbg2" and RMBG_2_SESS is not None:
            print(f"[GPU Idle] Releasing rmbg2 model after {(now - last):.0f}s idle")
            RMBG_2_SESS = None
        elif model_key == "birefnet" and BIREFNET_V1_LITE_SESS is not None:
            print(f"[GPU Idle] Releasing birefnet model after {(now - last):.0f}s idle")
            BIREFNET_V1_LITE_SESS = None

def _record_gpu_use(model_key):
    """Update last-use timestamp for a GPU model."""
    _LAST_GPU_USE[model_key] = time()

HIVISION_MODNET_SESS = None
MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS = None
RMBG_SESS = None
RMBG_2_SESS = None
BIREFNET_V1_LITE_SESS = None


def load_onnx_model(checkpoint_path, set_cpu=False):
    # Small model optimization: models under 50MB run faster on CPU than GPU
    if not set_cpu and os.path.exists(checkpoint_path):
        model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        if model_size_mb < 50:
            print(f"Model {os.path.basename(checkpoint_path)} is {model_size_mb:.1f}MB (<50MB), using CPU for better performance")
            set_cpu = True
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ONNX_PROVIDER == "CUDAExecutionProvider"
        else ["CPUExecutionProvider"]
    )

    if set_cpu:
        sess = onnxruntime.InferenceSession(
            checkpoint_path, providers=["CPUExecutionProvider"]
        )
    else:
        try:
            sess = onnxruntime.InferenceSession(checkpoint_path, providers=providers)
            print(f"Model {os.path.basename(checkpoint_path)} loaded with providers: {sess.get_providers()}")
        except Exception as e:
            if ONNX_PROVIDER == "CUDAExecutionProvider":  # 修复：使用ONNX_PROVIDER而不是ONNX_DEVICE
                print(f"Failed to load model with CUDAExecutionProvider: {e}")
                print("Falling back to CPUExecutionProvider")
                # 尝试使用CPU加载模型
                sess = onnxruntime.InferenceSession(
                    checkpoint_path, providers=["CPUExecutionProvider"]
                )
                print(f"Model {os.path.basename(checkpoint_path)} loaded with providers: {sess.get_providers()}")
            else:
                raise e  # 如果是CPU执行失败，重新抛出异常

    return sess


def extract_human(ctx: Context):
    """
    人像抠图
    :param ctx: 上下文
    """
    # 抠图
    matting_image = get_modnet_matting(ctx.processing_image, WEIGHTS["hivision_modnet"])
    if matting_image is None:
        raise RuntimeError("HivisionIDPhotos ModNet 模型加载失败或模型文件不存在，请检查模型文件路径")
    # 修复抠图
    ctx.processing_image = hollow_out_fix(matting_image)
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_modnet_photographic_portrait_matting(ctx: Context):
    """
    人像抠图
    :param ctx: 上下文
    """
    # 抠图
    matting_image = get_modnet_matting_photographic_portrait_matting(
        ctx.processing_image, WEIGHTS["modnet_photographic_portrait_matting"]
    )
    if matting_image is None:
        raise RuntimeError("ModNet Photographic Portrait 模型加载失败或模型文件不存在，请检查模型文件路径")
    # 修复抠图
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_mnn_modnet(ctx: Context):
    matting_image = get_mnn_modnet_matting(
        ctx.processing_image, WEIGHTS["mnn_hivision_modnet"]
    )
    if matting_image is None:
        raise RuntimeError("MNN ModNet 模型加载失败或模型文件不存在，请检查模型文件路径")
    ctx.processing_image = hollow_out_fix(matting_image)
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_rmbg(ctx: Context):
    matting_image = get_rmbg_matting(ctx.processing_image, WEIGHTS["rmbg-1.4"])
    if matting_image is None:
        raise RuntimeError("RMBG 1.4 模型加载失败或模型文件不存在，请检查模型文件路径")
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_rmbg_2(ctx: Context):
    # 获取抠图参数
    sensitivity = getattr(ctx, 'matting_sensitivity', 0.95)
    resolution = getattr(ctx, 'matting_resolution', 1024)
    
    matting_image = get_rmbg_2_matting(ctx.processing_image, WEIGHTS["rmbg-2.0"], 
                                      ref_size=resolution, sensitivity=sensitivity)
    if matting_image is None:
        raise RuntimeError("RMBG 2.0 模型加载失败或模型文件不存在，请检查模型文件路径")
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()


# def extract_human_birefnet_portrait(ctx: Context):
#     matting_image = get_birefnet_portrait_matting(
#         ctx.processing_image, WEIGHTS["birefnet-portrait"]
#     )
#     ctx.processing_image = matting_image
#     ctx.matting_image = ctx.processing_image.copy()


def extract_human_birefnet_lite(ctx: Context):
    matting_image = get_birefnet_portrait_matting(
        ctx.processing_image, WEIGHTS["birefnet-v1-lite"]
    )
    if matting_image is None:
        raise RuntimeError("BiRefNet Lite 模型加载失败或模型文件不存在，请检查模型文件路径")
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()


def hollow_out_fix(src: np.ndarray) -> np.ndarray:
    """
    修补抠图区域，作为抠图模型精度不够的补充
    :param src:
    :return:
    """
    b, g, r, a = cv2.split(src)
    src_bgr = cv2.merge((b, g, r))
    # -----------padding---------- #
    add_area = np.zeros((10, a.shape[1]), np.uint8)
    a = np.vstack((add_area, a, add_area))
    add_area = np.zeros((a.shape[0], 10), np.uint8)
    a = np.hstack((add_area, a, add_area))
    # -------------end------------ #
    _, a_threshold = cv2.threshold(a, 127, 255, 0)
    a_erode = cv2.erode(
        a_threshold,
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=3,
    )
    contours, hierarchy = cv2.findContours(
        a_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [x for x in contours]
    # contours = np.squeeze(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    a_contour = cv2.drawContours(np.zeros(a.shape, np.uint8), contours[0], -1, 255, 2)
    # a_base = a_contour[1:-1, 1:-1]
    h, w = a.shape[:2]
    mask = np.zeros(
        [h + 2, w + 2], np.uint8
    )  # mask 必须行和列都加 2，且必须为 uint8 单通道阵列
    cv2.floodFill(a_contour, mask=mask, seedPoint=(0, 0), newVal=255)
    a = cv2.add(a, 255 - a_contour)
    return cv2.merge((src_bgr, a[10:-10, 10:-10]))


def image2bgr(input_image):
    if len(input_image.shape) == 2:
        input_image = input_image[:, :, None]
    if input_image.shape[2] == 1:
        result_image = np.repeat(input_image, 3, axis=2)
    elif input_image.shape[2] == 4:
        result_image = input_image[:, :, 0:3]
    else:
        result_image = input_image

    return result_image


def read_modnet_image(input_image, ref_size=512):
    im = Image.fromarray(np.uint8(input_image))
    width, length = im.size[0], im.size[1]
    im = np.asarray(im)
    im = image2bgr(im)
    im = cv2.resize(im, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    im = NNormalize(im, mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5]))
    im = NUnsqueeze(NTo_Tensor(im))

    return im, width, length


def get_modnet_matting(input_image, checkpoint_path, ref_size=512):
    global HIVISION_MODNET_SESS

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None

    # 如果RUN_MODE不是野兽模式，则不加载模型
    if HIVISION_MODNET_SESS is None:
        HIVISION_MODNET_SESS = load_onnx_model(checkpoint_path)

    input_name = HIVISION_MODNET_SESS.get_inputs()[0].name
    output_name = HIVISION_MODNET_SESS.get_outputs()[0].name

    im, width, length = read_modnet_image(input_image=input_image, ref_size=ref_size)

    matte = HIVISION_MODNET_SESS.run([output_name], {input_name: im})
    matte = (matte[0] * 255).astype("uint8")
    matte = np.squeeze(matte)
    mask = cv2.resize(matte, (width, length), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(np.uint8(input_image))

    output_image = cv2.merge((b, g, r, mask))
    
    # 如果RUN_MODE不是野兽模式，则释放模型
    if os.getenv("RUN_MODE") != "beast":
        HIVISION_MODNET_SESS = None

    return output_image


def get_modnet_matting_photographic_portrait_matting(
    input_image, checkpoint_path, ref_size=512
):
    global MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None

    # 如果RUN_MODE不是野兽模式，则不加载模型
    if MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS is None:
        MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS = load_onnx_model(
            checkpoint_path
        )

    input_name = MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS.get_inputs()[0].name
    output_name = MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS.get_outputs()[0].name

    im, width, length = read_modnet_image(input_image=input_image, ref_size=ref_size)

    matte = MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS.run(
        [output_name], {input_name: im}
    )
    matte = (matte[0] * 255).astype("uint8")
    matte = np.squeeze(matte)
    mask = cv2.resize(matte, (width, length), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(np.uint8(input_image))

    output_image = cv2.merge((b, g, r, mask))
    
    # 如果RUN_MODE不是野兽模式，则释放模型
    if os.getenv("RUN_MODE") != "beast":
        MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS = None

    return output_image


def get_rmbg_matting(input_image: np.ndarray, checkpoint_path, ref_size=1024):
    global RMBG_SESS

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None

    def resize_rmbg_image(image):
        image = image.convert("RGB")
        model_input_size = (ref_size, ref_size)
        image = image.resize(model_input_size, Image.BILINEAR)
        return image

    _check_gpu_idle("rmbg")
    if RMBG_SESS is None:
        RMBG_SESS = load_onnx_model(checkpoint_path)

    orig_image = Image.fromarray(input_image)
    image = resize_rmbg_image(orig_image)
    im_np = np.array(image).astype(np.float32)
    im_np = im_np.transpose(2, 0, 1)  # Change to CxHxW format
    im_np = np.expand_dims(im_np, axis=0)  # Add batch dimension
    im_np = im_np / 255.0  # Normalize to [0, 1]
    im_np = (im_np - 0.5) / 0.5  # Normalize to [-1, 1]

    # Inference with I/O binding for GPU (avoids per-call memory allocation overhead)
    if "CUDAExecutionProvider" in RMBG_SESS.get_providers():
        io_binding = RMBG_SESS.io_binding()
        input_ort = onnxruntime.OrtValue.ortvalue_from_numpy(im_np, 'cuda', 0)
        io_binding.bind_ortvalue_input(RMBG_SESS.get_inputs()[0].name, input_ort)
        io_binding.bind_output(RMBG_SESS.get_outputs()[0].name, 'cuda')
        RMBG_SESS.run_with_iobinding(io_binding)
        result = io_binding.get_outputs()[0].numpy()
    else:
        result = RMBG_SESS.run(None, {RMBG_SESS.get_inputs()[0].name: im_np})[0]

    # Post process
    result = np.squeeze(result)
    ma = np.max(result)
    mi = np.min(result)
    result = (result - mi) / (ma - mi)  # Normalize to [0, 1]

    # Convert to PIL image with enhanced edge processing
    im_array = (result * 255).astype(np.uint8)
    pil_im = Image.fromarray(
        im_array, mode="L"
    )  # Ensure mask is single channel (L mode)

    # Resize the mask to match the original image size
    # 使用LANCZOS插值以获得更好的边缘质量，特别是对细微发丝
    pil_im = pil_im.resize(orig_image.size, Image.LANCZOS)

    # Paste the mask on the original image
    new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    new_im.paste(orig_image, mask=pil_im)
    
    # 如果RUN_MODE不是野兽模式，则释放模型
    if os.getenv("RUN_MODE") != "beast":
        RMBG_SESS = None
    else:
        _record_gpu_use("rmbg")

    return np.array(new_im)


def get_rmbg_2_matting(input_image: np.ndarray, checkpoint_path, ref_size=1024, sensitivity=0.95):
    """
    RMBG 2.0 抠图处理函数
    基于BiRefNet架构，使用不同的预处理参数
    """
    global RMBG_2_SESS

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None

    def resize_rmbg_2_image(image):
        image = image.convert("RGB")
        model_input_size = (ref_size, ref_size)
        image = image.resize(model_input_size, Image.BILINEAR)
        return image

    # 记录加载onnx模型的开始时间
    load_start_time = time()

    # 如果RUN_MODE不是野兽模式，则不加载模型
    _check_gpu_idle("rmbg2")
    if RMBG_2_SESS is None:
        # print("首次加载rmbg-2.0模型...")
        if ONNX_DEVICE == "GPU":
            print("onnxruntime-gpu已安装，尝试使用CUDA加载RMBG 2.0模型")
            try:
                import torch
            except ImportError:
                print(
                    "torch未安装，尝试直接使用onnxruntime-gpu加载模型，这需要配置好CUDA和cuDNN"
                )
            RMBG_2_SESS = load_onnx_model(checkpoint_path)
        else:
            RMBG_2_SESS = load_onnx_model(checkpoint_path)

    # 记录加载onnx模型的结束时间
    load_end_time = time()

    # 打印加载onnx模型所花的时间
    print(f"Loading RMBG 2.0 ONNX model took {load_end_time - load_start_time:.4f} seconds")

    orig_image = Image.fromarray(input_image)
    image = resize_rmbg_2_image(orig_image)
    
    # RMBG 2.0 使用标准的ImageNet预处理参数
    im_np = np.array(image).astype(np.float32)
    im_np = im_np.transpose(2, 0, 1)  # Change to CxHxW format
    im_np = np.expand_dims(im_np, axis=0)  # Add batch dimension
    im_np = im_np / 255.0  # Normalize to [0, 1]
    
    # 使用ImageNet标准化参数 (与BiRefNet架构一致)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    im_np = (im_np - mean) / std

    input_name = RMBG_2_SESS.get_inputs()[0].name
    print(onnxruntime.get_device(), RMBG_2_SESS.get_providers())

    time_st = time()
    # RMBG 2.0 has dynamic internal shapes, use standard run() instead of IO binding
    result = RMBG_2_SESS.run(None, {input_name: im_np.astype(np.float32)})[0]
    print(f"RMBG 2.0 Inference time: {time() - time_st:.4f} seconds")

    # Post process - RMBG 2.0 后处理
    result = np.squeeze(result)
    
    # 检查输出范围，如果在[-inf, inf]范围内则需要sigmoid，如果在[0,1]则已经过sigmoid
    if np.min(result) < 0 or np.max(result) > 1:
        # 应用sigmoid函数（模型输出未经过sigmoid）
        result = 1 / (1 + np.exp(-result))  # Sigmoid function
    
    # 应用敏感度调整 - 改善细微发丝处理
    # 敏感度越高，保留更多细节；敏感度越低，更加平滑
    if sensitivity != 1.0:
        # 使用幂函数调整敏感度，保持细微发丝的细节
        gamma = 2.0 - sensitivity  # sensitivity=0.95时，gamma=1.05；sensitivity=0.1时，gamma=1.9
        result = np.power(result, gamma)
    
    # 确保结果在[0,1]范围内
    result = np.clip(result, 0, 1)

    # Convert to PIL image with enhanced edge processing
    im_array = (result * 255).astype(np.uint8)
    pil_im = Image.fromarray(
        im_array, mode="L"
    )  # Ensure mask is single channel (L mode)

    # Resize the mask to match the original image size
    # 使用LANCZOS插值以获得更好的边缘质量，特别是对细微发丝
    pil_im = pil_im.resize(orig_image.size, Image.LANCZOS)

    # Paste the mask on the original image
    new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    new_im.paste(orig_image, mask=pil_im)
    
    # 如果RUN_MODE不是野兽模式，则释放模型
    if os.getenv("RUN_MODE") != "beast":
        RMBG_2_SESS = None
    else:
        _record_gpu_use("rmbg2")

    return np.array(new_im)


def get_mnn_modnet_matting(input_image, checkpoint_path, ref_size=512):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None

    try:
        import MNN.expr as expr
        import MNN.nn as nn
    except ImportError as e:
        raise ImportError(
            "The MNN module is not installed or there was an import error. Please ensure that the MNN library is installed by using the command 'pip install mnn'."
        ) from e

    config = {}
    config["precision"] = "low"  # 当硬件支持（armv8.2）时使用fp16推理
    config["backend"] = 0  # CPU
    config["numThread"] = 4  # 线程数
    im, width, length = read_modnet_image(input_image, ref_size=512)
    rt = nn.create_runtime_manager((config,))
    net = nn.load_module_from_file(
        checkpoint_path, ["input1"], ["output1"], runtime_manager=rt
    )
    input_var = expr.convert(im, expr.NCHW)
    output_var = net.forward(input_var)
    matte = expr.convert(output_var, expr.NCHW)
    matte = matte.read()  # var转换为np
    matte = (matte * 255).astype("uint8")
    matte = np.squeeze(matte)
    mask = cv2.resize(matte, (width, length), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(np.uint8(input_image))

    output_image = cv2.merge((b, g, r, mask))

    return output_image


def get_birefnet_portrait_matting(input_image, checkpoint_path, ref_size=512):
    global BIREFNET_V1_LITE_SESS

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None

    def transform_image(image):
        image = image.resize((1024, 1024))  # Resize to 1024x1024
        image = (
            np.array(image, dtype=np.float32) / 255.0
        )  # Convert to numpy array and normalize to [0, 1]
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Normalize
        image = np.transpose(image, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image.astype(np.float32)  # Ensure the output is float32

    orig_image = Image.fromarray(input_image)
    input_images = transform_image(
        orig_image
    )  # This will already have the correct shape

    # 记录加载onnx模型的开始时间
    load_start_time = time()

    # 如果RUN_MODE不是野兽模式，则不加载模型
    _check_gpu_idle("birefnet")
    if BIREFNET_V1_LITE_SESS is None:
        # print("首次加载birefnet-v1-lite模型...")
        if ONNX_DEVICE == "GPU":
            print("onnxruntime-gpu已安装，尝试使用CUDA加载模型")
            try:
                import torch
            except ImportError:
                print(
                    "torch未安装，尝试直接使用onnxruntime-gpu加载模型，这需要配置好CUDA和cuDNN"
                )
            BIREFNET_V1_LITE_SESS = load_onnx_model(checkpoint_path)
        else:
            BIREFNET_V1_LITE_SESS = load_onnx_model(checkpoint_path)

    # 记录加载onnx模型的结束时间
    load_end_time = time()

    # 打印加载onnx模型所花的时间
    print(f"Loading ONNX model took {load_end_time - load_start_time:.4f} seconds")

    input_name = BIREFNET_V1_LITE_SESS.get_inputs()[0].name
    print(onnxruntime.get_device(), BIREFNET_V1_LITE_SESS.get_providers())

    time_st = time()
    # Inference with I/O binding for GPU
    if "CUDAExecutionProvider" in BIREFNET_V1_LITE_SESS.get_providers():
        io_binding = BIREFNET_V1_LITE_SESS.io_binding()
        input_ort = onnxruntime.OrtValue.ortvalue_from_numpy(input_images, 'cuda', 0)
        io_binding.bind_ortvalue_input(input_name, input_ort)
        io_binding.bind_output(BIREFNET_V1_LITE_SESS.get_outputs()[0].name, 'cuda')
        BIREFNET_V1_LITE_SESS.run_with_iobinding(io_binding)
        pred_onnx = io_binding.get_outputs()[0].numpy()
    else:
        pred_onnx = BIREFNET_V1_LITE_SESS.run(None, {input_name: input_images})[-1]
    pred_onnx = np.squeeze(pred_onnx)  # Use numpy to squeeze
    result = 1 / (1 + np.exp(-pred_onnx))  # Sigmoid function using numpy
    print(f"Inference time: {time() - time_st:.4f} seconds")

    # Convert to PIL image
    im_array = (result * 255).astype(np.uint8)
    pil_im = Image.fromarray(
        im_array, mode="L"
    )  # Ensure mask is single channel (L mode)

    # Resize the mask to match the original image size
    pil_im = pil_im.resize(orig_image.size, Image.BILINEAR)

    # Paste the mask on the original image
    new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    new_im.paste(orig_image, mask=pil_im)
    
    # 如果RUN_MODE不是野兽模式，则释放模型
    if os.getenv("RUN_MODE") != "beast":
        BIREFNET_V1_LITE_SESS = None
    else:
        _record_gpu_use("birefnet")

    return np.array(new_im)
