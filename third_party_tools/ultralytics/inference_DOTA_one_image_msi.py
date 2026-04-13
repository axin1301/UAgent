from common_paths import weight_path
# import json
# import pandas as pd
# import tqdm
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# # 加载模型
# from fastapi import FastAPI, File, UploadFile,Form
# import uvicorn
# import matplotlib.pyplot as plt
# import os
# from PIL import Image
# import uuid
# import io
# import numpy as np


# def detect_image_sat(IMAGE_PATH):
#     # 1. 定义一个临时目录，确保它存在
#     TEMP_DIR = "temp_images"
#     os.makedirs(TEMP_DIR, exist_ok=True)

#     # 2. 生成一个唯一的文件名
#     # 可以根据需要选择图片格式，例如 .png, .jpg
#     unique_filename = f"{uuid.uuid4()}.png"
#     temp_file_path = os.path.join(TEMP_DIR, unique_filename)

#     # 3. 将 NumPy 数组保存为临时文件
#     # 使用 PIL.Image.fromarray 将 NumPy 数组转为 PIL Image 对象
#     pil_image = Image.fromarray(IMAGE_PATH)
#     pil_image.save(temp_file_path)

#     DOTA_class_list = ['plane','ship','storage tank','baseball diamond','tennis court','basketball court','ground track field','harbor',
#                 'bridge','large vehicle','small vehicle','helicopter','roundabout','soccer ball field','swimming pool']
#     model = YOLO(weight_path('yolo11l-obb.pt'))#.to('cuda')
#     ans_str = ''
#     results = model(
#         source=temp_file_path  # 输入图像路径
#         )
#     for res in results[0].obb: 
#         # resized_data = [[[coord[0] / 1024, coord[1] / 1024] for coord in sublist] for sublist in res.xyxyxyxyn]
#         # ans_str = ans_str + 'object: '+ str(DOTA_class_list[int(res.cls.cpu().numpy())]) + ', bbox: '+  str(res.xyxyxyxyn.cpu().numpy())+ ', scores: '+str(res.conf.cpu().numpy()) +'; '
#         ans_str = ans_str + 'object: '+ str(DOTA_class_list[int(res.cls.cpu().numpy())]) + ', bbox: '+  str(res.xyxyxyxyn.cpu().numpy()[0])+'; '
    
#     return ans_str

# # detect_image_sat("49617_107871.png")
# # ########################################################################################################################
# app = FastAPI()

# @app.post("/inference_DOTA_one_image")
# async def inference_DOTA_one_image_endpoint(file: UploadFile = File(...)):
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
import pandas as pd
import tqdm
from ultralytics import YOLO # YOLOv8 for object detection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import matplotlib.pyplot as plt
import os
from PIL import Image
import uuid
import io
import numpy as np
import torch # 导入 torch 用于检查 cuda

# 全局变量来存储加载的模型
global_dota_model = None

DOTA_class_list = [
    'plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
    'basketball court', 'ground track field', 'harbor', 'bridge',
    'large vehicle', 'small vehicle', 'helicopter', 'roundabout',
    'soccer ball field', 'swimming pool'
]

def load_dota_model(model_path):
    """
    加载 YOLOv8 OBB 模型。
    :param model_path: 模型权重文件路径
    :return: 加载后的 YOLO 模型
    """
    print("Loading DOTA YOLOv8 OBB model...")
    # 判断是否有 GPU，并选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device) # 加载模型并移动到指定设备
    model.eval() # 将模型设置为评估模式
    print(f"DOTA YOLOv8 OBB Model loaded on {device}.")
    return model

def detect_objects_dota(model, image_array, score_thr=0.2):
    """
    使用 DOTA 模型检测图像中的物体。
    :param model: 加载的 YOLOv8 模型
    :param image_array: 图像的 numpy 数组
    :param score_thr: 置信度阈值
    :return: 检测结果字符串
    """
    if image_array is None:
        raise ValueError("输入图像数组为空。")
    
    # YOLOv8 模型可以直接接收 numpy 数组作为 source
    # results = model(image_array) 
    # 对于批处理推理，可以传入列表
    results = model([image_array], verbose=False) # verbose=False 减少控制台输出

    ans_str = ''
    # YOLOv8 的 OBB (Oriented Bounding Box) 结果存储在 .obb 属性中
    # 每个检测结果是一个 Results 对象，其 .obb 属性包含了所有旋转框信息
    
    # 确保 results 列表不为空
    if results and len(results) > 0:
        for res in results[0].obb: # 通常批处理只有一张图，所以取 results[0]
            score = res.conf.item() # 获取置信度
            if score >= score_thr:
                label_id = int(res.cls.item()) # 获取类别ID
                
                # 检查标签ID是否在 DOTA_class_list 范围内
                if 0 <= label_id < len(DOTA_class_list):
                    object_name = DOTA_class_list[label_id]
                else:
                    object_name = f"Unknown_Label_{label_id}"

                # 获取 OBB 坐标，通常是 [x_center, y_center, width, height, angle]
                # 或者直接获取 xyxyxyxy (8个点坐标)
                # 你原始代码中用的是 res.xyxyxyxyn，这里保留
                # 注意：res.xyxyxyxyn 是归一化坐标，如果需要原始像素坐标，可能是 res.xyxyxyxy
                # obb_coords = res.xyxyxyxyn.cpu().numpy() # 假设这里是归一化坐标
                obb_coords = res.xyxyxyxy.cpu().numpy() # 假设这里是归一化坐标
                
                # 你的原始代码取了 [0] 并且没有对宽度进行归一化，这里保持原样，但再次提醒可能需要调整
                # 假设你希望输出的是归一化后的 8 个顶点坐标
                # ans_str += f'object: {object_name}, bbox: {obb_coords[0].tolist()}; ' 
                # 如果需要scores，可以加回去
                ans_str += f'object: {object_name}, bbox: {obb_coords[0].tolist()}, scores: [{score}]; '

    return ans_str

# ########################################################################################################################
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    FastAPI 应用启动时加载 DOTA YOLOv8 OBB 模型
    """
    global global_dota_model
    model_weights_path = weight_path('yolo11l-obb.pt') # 模型权重文件路径

    # 检查模型权重文件是否存在，通常 yolo11l-obb.pt 会在首次运行时自动下载到 ~/.cache/ultralytics/assets/ 或当前目录
    if not os.path.exists(model_weights_path):
        print(f"Warning: Model weights '{model_weights_path}' not found locally. YOLO will attempt to download it.")
        # YOLO会自动处理下载，这里不需要手动下载逻辑

    global_dota_model = load_dota_model(model_weights_path)


@app.post("/inference_DOTA_one_image")
async def inference_DOTA_one_image_endpoint(file: UploadFile = File(...)):
    """
    接收图像文件，进行 DOTA OBB 检测并返回结果
    """
    try:
        # 确保模型已经加载
        if global_dota_model is None:
            return {"status": "error", "error": "DOTA YOLOv8 OBB Model not loaded. Please restart the application."}

        # 1. 读取上传文件的内容
        contents = await file.read() # UploadFile 的 read() 方法是异步的，返回字节流

        # 2. 将字节流读取为 PIL 图像
        image = Image.open(io.BytesIO(contents)).convert("RGB") # 确保图像是RGB格式

        # 3. 将图像转换为 numpy 数组 (H, W, C)
        image_array = np.array(image)
        
        # 调用检测模型，直接传入已加载的 global_dota_model 和 NumPy 数组
        detection_result = detect_objects_dota(global_dota_model, image_array)
        
        # 打印到控制台，以便调试
        print('final_detection_results_DOTA: ', detection_result)
        
        # 返回结果
        return {"status": "success", "result": detection_result}

    except Exception as e:
        import traceback
        traceback.print_exc() # 这行会打印详细的错误堆栈信息
        return {"status": "error", "error": f"处理文件时发生错误: {str(e)}"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
    #10.112.27.164

