from common_paths import weight_path
import os
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from fastapi import FastAPI, UploadFile, File
import cv2

# 全局变量来存储加载的模型
global_dota_model = None
global_model_imgsz = None # 在此需求下，主要用于加载模型时的信息输出

DOTA_class_list = [
    'plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
    'basketball court', 'ground track field', 'harbor', 'bridge',
    'large vehicle', 'small vehicle', 'helicopter', 'roundabout',
    'soccer ball field', 'swimming pool'
]

def load_dota_model(model_path):
    global global_model_imgsz
    print("Loading DOTA YOLOv8 OBB model...")
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device)
    model.eval()
    
    try:
        if isinstance(model.model.yaml.get('imgsz'), int):
            global_model_imgsz = (model.model.yaml.get('imgsz'), model.model.yaml.get('imgsz'))
        elif isinstance(model.model.yaml.get('imgsz'), (list, tuple)) and len(model.model.yaml.get('imgsz')) == 2:
            global_model_imgsz = tuple(model.model.yaml.get('imgsz'))
        else:
            print("Warning: Could not determine model's imgsz from model.yaml. Defaulting to (1024, 1024).")
            global_model_imgsz = (1024, 1024) 
    except Exception as e:
        print(f"Error getting imgsz from model.yaml: {e}. Defaulting to (1024, 1024).")
        global_model_imgsz = (1024, 1024)
    
    print(f"DOTA YOLOv8 OBB Model loaded on {device}. Model input size (imgsz): {global_model_imgsz}.")
    return model

def detect_objects_dota(model, image_array, score_thr=0.2):
    if image_array is None:
        raise ValueError("输入图像数组为空。")
    
    original_img_height, original_img_width, _ = image_array.shape
    
    results = model([image_array], verbose=False, conf=score_thr)

    ans_str = ''
    
    if results and len(results) > 0:
        for res in results[0].obb:
            score = res.conf.item()
            
            if score >= score_thr:
                label_id = int(res.cls.item())
                
                if 0 <= label_id < len(DOTA_class_list):
                    object_name = DOTA_class_list[label_id]
                else:
                    object_name = f"Unknown_Label_{label_id}"

                # 获取 OBB 8个顶点的像素坐标，并确保是 1D NumPy 数组
                obb_points_flat = res.xyxyxyxy.cpu().numpy().flatten() 
                
                is_out_of_bounds = False
                for i in range(0, len(obb_points_flat), 2):
                    x_coord = obb_points_flat[i]
                    y_coord = obb_points_flat[i+1]
                    
                    if not (0 <= x_coord < original_img_width and 0 <= y_coord < original_img_height):
                        is_out_of_bounds = True
                        break

                if is_out_of_bounds:
                    continue

                # === 修改点：将 1D 8个点的列表重塑为 4个2D点列表 ===
                # obb_points_flat 现在是 [x1, y1, x2, y2, x3, y3, x4, y4]
                # 需要变成 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                reshaped_bbox_coords = obb_points_flat.reshape(4, 2).tolist()
                
                ans_str += f'object: {object_name}, bbox: {reshaped_bbox_coords}, scores: [{score}]; '

    return ans_str

# ########################################################################################################################
app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     global global_dota_model, global_model_imgsz
#     model_weights_path = weight_path('yolo11l-obb.pt')
#     if not os.path.exists(model_weights_path):
#         print(f"Warning: Model weights '{model_weights_path}' not found locally. YOLO will attempt to download it.")
#     global_dota_model = load_dota_model(model_weights_path)


# @app.post("/inference_DOTA_one_image")
# async def inference_DOTA_one_image_endpoint(file: UploadFile = File(...)):
#     try:
#         if global_dota_model is None:
#             return {"status": "error", "error": "DOTA YOLOv8 OBB Model not loaded. Please restart the application."}

#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         image_array = np.array(image)

#         detection_result = detect_objects_dota(global_dota_model, image_array)
        
#         print('final_detection_results_DOTA: ', detection_result)
        
#         return {"status": "success", "result": detection_result}

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return {"status": "error", "error": f"处理文件时发生错误: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    global global_dota_model, global_model_imgsz
    model_weights_path = weight_path('yolo11l-obb.pt')
    if not os.path.exists(model_weights_path):
        print(f"Warning: Model weights '{model_weights_path}' not found locally. YOLO will attempt to download it.")
    global_dota_model = load_dota_model(model_weights_path)


@app.post("/inference_DOTA_one_image")
async def inference_DOTA_one_image_endpoint(file: UploadFile = File(...)):
    try:
        if global_dota_model is None:
            return {"status": "error", "error": "DOTA YOLOv8 OBB Model not loaded. Please restart the application."}

        contents = await file.read() 
        
        # --- 关键修改：使用 OpenCV 读取图像 ---
        # 1. 将字节流转换为 NumPy 数组
        np_array = np.frombuffer(contents, np.uint8)
        
        # 2. 使用 OpenCV 从 NumPy 数组中解码图像
        # cv2.imdecode 能够智能识别图像格式（JPEG, PNG等）并解码
        image_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR) # IMREAD_COLOR 会读取为 BGR 三通道
        
        if image_array is None:
            raise ValueError("无法解码图像。请检查上传文件是否为有效图像。")
        
        # YOLOv8 模型的预处理通常在内部处理 BGR 到 RGB 或其他格式的转换。
        # 如果模型训练时是期望 RGB 输入，则可能需要 cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # 但通常 Ultralytics 在接收 cv2 读取的 BGR 图像时会正确处理。
        
        # 确保图像的通道顺序与模型训练时一致
        # 大多数情况下，YOLOv8 在其内部预处理会处理好这个，直接传入 BGR 即可
        # 如果你特别担心，可以手动转为 RGB：
        # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) 
        # 但通常保持 BGR 传入给 YOLO 会更稳定

        # 调用检测模型
        detection_result = detect_objects_dota(global_dota_model, image_array)
        
        print('final_detection_results_DOTA: ', detection_result)
        
        return {"status": "success", "result": detection_result}

    except Exception as e:
        # traceback.print_exc()
        return {"status": "error", "error": f"处理文件时发生错误: {str(e)}"}
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
    #10.112.27.164
