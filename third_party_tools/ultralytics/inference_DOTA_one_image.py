from common_paths import weight_path
import json
import pandas as pd
import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 加载模型
from fastapi import FastAPI, File, UploadFile,Form
import uvicorn
import matplotlib.pyplot as plt
import os
from PIL import Image
import uuid
import io
import numpy as np


def detect_image_sat(IMAGE_PATH):
    # 1. 定义一个临时目录，确保它存在
    TEMP_DIR = "temp_images"
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 2. 生成一个唯一的文件名
    # 可以根据需要选择图片格式，例如 .png, .jpg
    unique_filename = f"{uuid.uuid4()}.png"
    temp_file_path = os.path.join(TEMP_DIR, unique_filename)

    # 3. 将 NumPy 数组保存为临时文件
    # 使用 PIL.Image.fromarray 将 NumPy 数组转为 PIL Image 对象
    pil_image = Image.fromarray(IMAGE_PATH)
    pil_image.save(temp_file_path)

    DOTA_class_list = ['plane','ship','storage tank','baseball diamond','tennis court','basketball court','ground track field','harbor',
                'bridge','large vehicle','small vehicle','helicopter','roundabout','soccer ball field','swimming pool']
    model = YOLO(weight_path('yolo11l-obb.pt'))#.to('cuda')
    ans_str = ''
    results = model(
        source=temp_file_path  # 输入图像路径
        )
    for res in results[0].obb: 
        # resized_data = [[[coord[0] / 1024, coord[1] / 1024] for coord in sublist] for sublist in res.xyxyxyxyn]
        ans_str = ans_str + 'object: '+ str(DOTA_class_list[int(res.cls.cpu().numpy())]) + ', bbox: '+  str(res.xyxyxyxyn.cpu().numpy())+ ', scores: '+str(res.conf.cpu().numpy()) +'; '
    
    return ans_str

# detect_image_sat("49617_107871.png")
# ########################################################################################################################
app = FastAPI()

@app.post("/inference_DOTA_one_image")
async def inference_DOTA_one_image_endpoint(file: UploadFile = File(...)):
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
    uvicorn.run(app, host="127.0.0.1", port=8003)
    #10.112.27.164

