from common_paths import mmseg_config, weight_path, BACKBONE_SUPPORT_ROOT
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
from mmengine.config import Config
from mmengine.runner import Runner
from mmcv.image import imread, imwrite,imresize
from mmseg.registry import MODELS, DATASETS, TRANSFORMS
# from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis import inference_model, init_model
import sys
from PIL import Image
sys.path.append(BACKBONE_SUPPORT_ROOT)`r`nfrom vit_rvsa_mtp import RVSA_MTP
import io
import numpy as np
import cv2
from collections import defaultdict

def detect_image_sat(img):
    class_list = ['background','building','road','water','barren','forest','agriculture']#'no_data',
        # args = parse_args()

    num_classes = 7  # 假设有7类
    config_path = mmseg_config('rvsa-b-upernet-512-mae-mtp-loveda.py')
    checkpoint_path = weight_path('loveda-rvsa-b-mae-mtp-iter_80000.pth')#spacenetv1

    cfg = Config.fromfile(config_path)

    # Initialize the segmentation model
    model = init_model(cfg, checkpoint_path, device = 'cuda')#'cpu') #device='cuda:0')
    set_width = 1024

    image = img #= Image.fromarray(img)
    image= imresize(image, (set_width, set_width))
        # Perform inference
    result = inference_model(model, image)

        # Save the segmentation result
    if isinstance(result, list):
    # If the result contains multiple outputs (e.g., TTA), use the first one
        result = result[0]
    pred_seg = result.pred_sem_seg.data.squeeze(0).cpu().numpy()
    num_classes = 7  # 假设有7类

    # 初始化统计结果
    height, width = pred_seg.shape
    total_pixels = height * width
    class_pixel_count = defaultdict(int)
    class_contours = defaultdict(list)

    # 简化轮廓的阈值
    epsilon_factor = 0.1  # 轮廓边界简化程度，越大点越少
    ans_str = ''

    for cls in [1,3,4,5,6]:#range(1,num_classes):
        # 提取当前类别的二值掩码
        mask = (pred_seg == cls).astype(np.uint8)

        # 计算像素占比
        class_pixel_count[cls] = np.sum(mask)
        pixel_ratio = class_pixel_count[cls] / total_pixels

        # 提取轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 简化轮廓
        simplified_contours = []
        for contour in contours:
            if len(contour) > 0:
                # 根据轮廓的周长和 epsilon_factor 简化轮廓
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                simplified_contours.append(approx.squeeze(1).tolist())

        # 保存简化后的轮廓
        class_contours[cls] = simplified_contours

        resized_data = [[[coord[0] / set_width, coord[1] / set_width] for coord in sublist] for sublist in class_contours[cls]]
        ans_str = ans_str + str(class_list[cls]) + ' pixel percentage: '+ str(pixel_ratio)[:6] +'; ' #+ ', normalized boundary: '+  str(resized_data) 
    color_map = {
            0: [219, 65, 154],      # background -> [219, 65, 154]
            1: [56, 148, 175],      # building -> [56, 148, 175]
            2: [143, 251, 128],     # road -> [143, 251, 128]
            3: [219, 154, 225],     # water -> [219, 154, 225]
            4: [146, 179, 117],     # barren -> [146, 179, 117]
            5: [29, 196, 92],       # forest -> [29, 196, 92]
            6: [124, 79, 78],       # agriculture -> [124, 79, 78]
            # 如果类别更多，可以继续添加颜色映射
        }

        # 初始化彩色分割图
    colored_seg = np.zeros((*pred_seg.shape, 3), dtype=np.uint8)

    # 映射每个类别到对应颜色
    for class_id in range(num_classes):
        if class_id in color_map:
            colored_seg[pred_seg == class_id] = color_map[class_id]
    overlay = (colored_seg).astype(np.uint8) #0.5 * original_image + 0.5 * colored_seg).astype(np.uint8)
        # 将 NumPy 数组转换为 PIL 图像

    generated_pil_image = Image.fromarray(overlay)

    # 6. 将 PIL Image 保存到内存中的字节流，并进行 Base64 编码
    image_bytes_io = io.BytesIO()
    generated_pil_image.save(image_bytes_io, format="PNG") # 指定输出格式为 PNG
    image_bytes_io.seek(0) # 将游标移到文件开头

    # 获取图片的原始字节数据
    image_raw_bytes = image_bytes_io.getvalue()
    # 对字节数据进行 Base64 编码
    encoded_image_string = base64.b64encode(image_raw_bytes).decode('utf-8')

    # 7. 构建 JSON 响应
    response_data = {
        "message": ans_str,
        "generated_image_png_base64": encoded_image_string,
        "image_format": "png", # 告知客户端图像格式
        "image_encoding": "base64" # 告知客户端编码方式
    }

    return JSONResponse(content=response_data)

# detect_image_sat("49617_107871.png")
# ########################################################################################################################
app = FastAPI()

@app.post("/inference_loveda_one_image")
async def inference_loveda_one_image_endpoint(file: UploadFile = File(...)):
    """
    接收图像文件，进行分割并返回结果
    """
    try:
        # 1. 读取上传文件的内容
        contents = await file.read() # UploadFile 的 read() 方法是异步的，返回字节流

        # 2. 将字节流读取为 PIL 图像
        image = Image.open(io.BytesIO(contents)) # PIL 期望一个文件状对象，BytesIO 提供了这个

        # 3. 将图像转换为 numpy 数组
        image_array = np.array(image)
        
        # 调用分割模型
        segmentation_result = detect_image_sat(image_array)
        
        return segmentation_result
        # # 返回结果
        # return {
        #     "status": "success",
        #     "object": segmentation_result
        # }
        # return segmentation_result
    except Exception as e:
        import traceback
        traceback.print_exc() # 这行会打印详细的错误堆栈信息
        return {"status": "error!!!!", "error": f"处理文件时发生错误: {str(e)}"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
    #10.112.27.164

