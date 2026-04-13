from common_paths import mmseg_config, weight_path, BACKBONE_SUPPORT_ROOT
# import json
# import shutil
# import requests
# import pandas as pd
# import tqdm
# import argparse
# from fastapi import FastAPI, File, UploadFile,Form
# import uvicorn
# import os
# import base64
# from fastapi.responses import JSONResponse
# import numpy as np
# import matplotlib.pyplot as plt
# from mmengine.config import Config
# from mmengine.runner import Runner
# from mmcv.image import imread, imwrite,imresize
# from mmseg.registry import MODELS, DATASETS, TRANSFORMS
# # from mmseg.apis import inference_segmentor, init_segmentor
# from mmseg.apis import inference_model, init_model
# import sys
# from PIL import Image
# sys.path.append("../../../MTP-main/MTP-main/RS_Tasks_Finetune/Semantic_Segmentation/mmseg/models/backbones")
# from vit_rvsa_mtp import RVSA_MTP
# import io
# import numpy as np
# import cv2
# from collections import defaultdict

# def detect_image_sat(img):
#     class_list = ['background','building','road','water','barren','forest','agriculture']#'no_data',
#         # args = parse_args()

#     num_classes = 7  # 假设有7类
#     config_path = mmseg_config('rvsa-b-upernet-512-mae-mtp-loveda.py')
#     checkpoint_path = weight_path('loveda-rvsa-b-mae-mtp-iter_80000.pth')#spacenetv1

#     cfg = Config.fromfile(config_path)

#     # Initialize the segmentation model
#     model = init_model(cfg, checkpoint_path, device = 'cuda')#'cpu') #device='cuda:0')
#     set_width = 1024

#     image = img #= Image.fromarray(img)
#     image= imresize(image, (set_width, set_width))
#         # Perform inference
#     result = inference_model(model, image)

#         # Save the segmentation result
#     if isinstance(result, list):
#     # If the result contains multiple outputs (e.g., TTA), use the first one
#         result = result[0]
#     pred_seg = result.pred_sem_seg.data.squeeze(0).cpu().numpy()
#     num_classes = 7  # 假设有7类

#     # 初始化统计结果
#     height, width = pred_seg.shape
#     total_pixels = height * width
#     class_pixel_count = defaultdict(int)
#     class_contours = defaultdict(list)

#     # 简化轮廓的阈值
#     epsilon_factor = 0.1  # 轮廓边界简化程度，越大点越少
#     ans_str = ''

#     for cls in [1,3,4,5,6]:#range(1,num_classes):
#         # 提取当前类别的二值掩码
#         mask = (pred_seg == cls).astype(np.uint8)

#         # 计算像素占比
#         class_pixel_count[cls] = np.sum(mask)
#         pixel_ratio = class_pixel_count[cls] / total_pixels

#         # 提取轮廓
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # 简化轮廓
#         simplified_contours = []
#         for contour in contours:
#             if len(contour) > 0:
#                 # 根据轮廓的周长和 epsilon_factor 简化轮廓
#                 epsilon = epsilon_factor * cv2.arcLength(contour, True)
#                 approx = cv2.approxPolyDP(contour, epsilon, True)
#                 simplified_contours.append(approx.squeeze(1).tolist())

#         # 保存简化后的轮廓
#         class_contours[cls] = simplified_contours

#         resized_data = [[[coord[0] / set_width, coord[1] / set_width] for coord in sublist] for sublist in class_contours[cls]]
#         ans_str = ans_str + str(class_list[cls]) + ' pixel percentage: '+ str(pixel_ratio)[:6] +'; ' #+ ', normalized boundary: '+  str(resized_data) 
#     color_map = {
#             0: [219, 65, 154],      # background -> [219, 65, 154]
#             1: [56, 148, 175],      # building -> [56, 148, 175]
#             2: [143, 251, 128],     # road -> [143, 251, 128]
#             3: [219, 154, 225],     # water -> [219, 154, 225]
#             4: [146, 179, 117],     # barren -> [146, 179, 117]
#             5: [29, 196, 92],       # forest -> [29, 196, 92]
#             6: [124, 79, 78],       # agriculture -> [124, 79, 78]
#             # 如果类别更多，可以继续添加颜色映射
#         }

#         # 初始化彩色分割图
#     colored_seg = np.zeros((*pred_seg.shape, 3), dtype=np.uint8)

#     # 映射每个类别到对应颜色
#     for class_id in range(num_classes):
#         if class_id in color_map:
#             colored_seg[pred_seg == class_id] = color_map[class_id]
#     overlay = (colored_seg).astype(np.uint8) #0.5 * original_image + 0.5 * colored_seg).astype(np.uint8)
#         # 将 NumPy 数组转换为 PIL 图像

#     generated_pil_image = Image.fromarray(overlay)

#     # 6. 将 PIL Image 保存到内存中的字节流，并进行 Base64 编码
#     image_bytes_io = io.BytesIO()
#     generated_pil_image.save(image_bytes_io, format="PNG") # 指定输出格式为 PNG
#     image_bytes_io.seek(0) # 将游标移到文件开头

#     # 获取图片的原始字节数据
#     image_raw_bytes = image_bytes_io.getvalue()
#     # 对字节数据进行 Base64 编码
#     encoded_image_string = base64.b64encode(image_raw_bytes).decode('utf-8')

#     # 7. 构建 JSON 响应
#     response_data = {
#         "message": ans_str,
#         "generated_image_png_base64": encoded_image_string,
#         "image_format": "png", # 告知客户端图像格式
#         "image_encoding": "base64" # 告知客户端编码方式
#     }

#     return JSONResponse(content=response_data)

# # detect_image_sat("49617_107871.png")
# # ########################################################################################################################
# app = FastAPI()

# @app.post("/inference_loveda_one_image")
# async def inference_loveda_one_image_endpoint(file: UploadFile = File(...)):
#     """
#     接收图像文件，进行分割并返回结果
#     """
#     try:
#         # 1. 读取上传文件的内容
#         contents = await file.read() # UploadFile 的 read() 方法是异步的，返回字节流

#         # 2. 将字节流读取为 PIL 图像
#         image = Image.open(io.BytesIO(contents)) # PIL 期望一个文件状对象，BytesIO 提供了这个

#         # 3. 将图像转换为 numpy 数组
#         image_array = np.array(image)
        
#         # 调用分割模型
#         segmentation_result = detect_image_sat(image_array)
        
#         return segmentation_result
#         # # 返回结果
#         # return {
#         #     "status": "success",
#         #     "object": segmentation_result
#         # }
#         # return segmentation_result
#     except Exception as e:
#         import traceback
#         traceback.print_exc() # 这行会打印详细的错误堆栈信息
#         return {"status": "error!!!!", "error": f"处理文件时发生错误: {str(e)}"}

import json
import shutil
import requests
import pandas as pd
import tqdm
import argparse
from fastapi import FastAPI, File, UploadFile,Form
import uvicorn
import os
import base64
from fastapi.responses import JSONResponse
import numpy as np
import matplotlib.pyplot as plt
import torch # 导入 torch 用于检查 cuda
from mmengine.config import Config
from mmengine.runner import Runner
from mmcv.image import imread, imwrite,imresize # 保持 mmcv.image 的导入
from mmseg.registry import MODELS, DATASETS, TRANSFORMS
from mmseg.apis import inference_model, init_model
import sys
from PIL import Image
import io
import cv2
from collections import defaultdict

# 确保导入路径正确，这里使用 os.path.abspath 和 os.path.join 确保路径的鲁棒性
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../../../MTP-main/MTP-main/RS_Tasks_Finetune/Semantic_Segmentation/mmseg/models/backbones"))
# 导入模型，确保 RVSA_MTP 被正确注册到 MMSegmentation 的 MODELS 注册表中
from vit_rvsa_mtp import RVSA_MTP 

# 全局变量来存储加载的模型
global_loveda_model = None

class_list = ['background','building','road','water','barren','forest','agriculture']

def load_loveda_model(config_file, checkpoint_file):
    """
    加载语义分割模型。
    :param config_file: 配置文件路径
    :param checkpoint_file: 检查点文件路径
    :return: 加载后的模型
    """
    print("Loading Loveda semantic segmentation model...")
    cfg = Config.fromfile(config_file)
    # 对于 MMSegmentation，init_model 会自动处理 checkpoint_file 的加载，
    # 通常不需要在 cfg 中设置 pretrained 参数
    
    device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(cfg, checkpoint_file, device=device)
    model.eval() # 将模型设置为评估模式
    print(f"Loveda Model loaded on {device}.")
    return model

def detect_image_sat(img_array):
    """
    使用 Loveda 模型进行语义分割。
    :param img_array: 输入图像的 numpy 数组 (H, W, C)
    :return: JSONResponse 包含分割结果文本和 Base64 编码的彩色分割图
    """
    # 确保模型已经加载
    if global_loveda_model is None:
        raise RuntimeError("Loveda Model not loaded. Please restart the application.")

    model = global_loveda_model
    
    set_width = 1024

    # 确保输入图像是 NumPy 数组，并且是 RGB 格式
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array.convert("RGB"))
    elif len(img_array.shape) == 2: # 灰度图
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4: # RGBA图
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    
    # 调整图像大小
    # imresize 期望 BGR 格式，如果你的图像是 RGB，需要转换
    # 或者直接使用 PIL 或 OpenCV 的 resize，它们对 RGB 格式更友好
    # 这里我们假设 imresize 也能处理 RGB，或者你确保输入到imresize的是BGR
    # 最佳实践是将 PIL.Image 转换为 NumPy 数组后，直接用 cv2.resize
    image_resized = cv2.resize(img_array, (set_width, set_width), interpolation=cv2.INTER_LINEAR)
    
    # 执行推理
    with torch.no_grad():
        result = inference_model(model, image_resized)

    # 处理分割结果
    if isinstance(result, list):
        result = result[0] # 如果是列表，取第一个
    pred_seg = result.pred_sem_seg.data.squeeze(0).cpu().numpy()

    num_classes = 7
    height, width = pred_seg.shape
    total_pixels = height * width
    class_pixel_count = defaultdict(int)
    class_contours = defaultdict(list)

    epsilon_factor = 0.1
    ans_str = ''

    for cls in [1,3,4,5,6]: # 遍历指定的类别
        mask = (pred_seg == cls).astype(np.uint8)
        class_pixel_count[cls] = np.sum(mask)
        pixel_ratio = class_pixel_count[cls] / total_pixels

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = []
        for contour in contours:
            if len(contour) > 0:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                simplified_contours.append(approx.squeeze(1).tolist())
        class_contours[cls] = simplified_contours

        # 注意这里 bbox 没有加入到 ans_str 中，如果你需要，可以加入
        # ans_str += f'{class_list[cls]} pixel percentage: {pixel_ratio:.4f}; '
        ans_str += f'{class_list[cls]} pixel percentage: {pixel_ratio:.2%}; '

    color_map = {
        0: [219, 65, 154],       # background
        1: [56, 148, 175],       # building
        2: [143, 251, 128],      # road
        3: [219, 154, 225],      # water
        4: [146, 179, 117],      # barren
        5: [29, 196, 92],        # forest
        6: [124, 79, 78],        # agriculture
    }

    # 初始化彩色分割图
    colored_seg = np.zeros((*pred_seg.shape, 3), dtype=np.uint8)

    # 映射每个类别到对应颜色
    for class_id in range(num_classes):
        if class_id in color_map:
            colored_seg[pred_seg == class_id] = color_map[class_id]
            
    # OpenCV 图像是 BGR 格式，而 PIL 默认是 RGB。如果 colored_seg 是 OpenCV 处理的，
    # 并且你需要用 PIL 保存，可能需要转换颜色通道。
    # 这里假设 color_map 中的颜色是 RGB 顺序，所以直接创建 PIL Image
    generated_pil_image = Image.fromarray(colored_seg.astype(np.uint8))

    # 将 PIL Image 保存到内存中的字节流，并进行 Base64 编码
    image_bytes_io = io.BytesIO()
    generated_pil_image.save(image_bytes_io, format="PNG") # 指定输出格式为 PNG
    image_bytes_io.seek(0) # 将游标移到文件开头

    image_raw_bytes = image_bytes_io.getvalue()
    encoded_image_string = base64.b64encode(image_raw_bytes).decode('utf-8')

    response_data = {
        "message": ans_str,
        "generated_image_png_base64": encoded_image_string,
        "image_format": "png",
        "image_encoding": "base64"
    }

    return JSONResponse(content=response_data)

# ########################################################################################################################
app = FastAPI()

# FastAPI 应用启动时加载 Loveda 模型
@app.on_event("startup")
async def startup_event():
    global global_loveda_model
    config_path = mmseg_config('rvsa-b-upernet-512-mae-mtp-loveda.py')
    checkpoint_path = weight_path('loveda-rvsa-b-mae-mtp-iter_80000.pth')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    global_loveda_model = load_loveda_model(config_path, checkpoint_path)


@app.post("/inference_loveda_one_image")
async def inference_loveda_one_image_endpoint(file: UploadFile = File(...)):
    """
    接收图像文件，进行语义分割并返回结果
    """
    try:
        # 确保模型已经加载
        if global_loveda_model is None:
            return JSONResponse(status_code=500, content={"status": "error", "error": "Loveda Model not loaded. Please restart the application."})

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB") # 确保图像是RGB格式
        image_array = np.array(image)
        
        # 调用分割模型
        segmentation_response = detect_image_sat(image_array)
        
        # detect_image_sat 已经返回 JSONResponse，直接返回即可
        return segmentation_response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "error": f"处理文件时发生错误: {str(e)}"})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
    #10.112.27.164

