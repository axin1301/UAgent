from common_paths import mmdet_config, weight_path, BACKBONE_SUPPORT_ROOT, DATASET_SUPPORT_ROOT
# import torch
# import json
# import matplotlib.patches as patches
# import sys
# sys.path.append("../../../mmdetection-main/mmdetection-main")
# # from mmcv import Config
# from mmengine.config import Config
# from mmengine.runner import load_checkpoint#,get_dist_info,init_dist
# # from mmcv.runner import load_checkpoint
# from mmdet.apis import inference_detector, init_detector
# import cv2
# from mmdet.registry import MODELS
# import sys
# sys.path.append("../../../MTP-main/MTP-main/RS_Tasks_Finetune/Horizontal_Detection/mmdet/models/backbones")
# from vit_rvsa_mtp_branches import RVSA_MTP_branches
# from fastapi import FastAPI, File, UploadFile,Form
# import uvicorn
# import numpy as np
# import cv2
# from collections import defaultdict
# import pandas as pd
# import sys
# sys.path.append("../../../MTP-main/MTP-main/RS_Tasks_Finetune/Horizontal_Detection/mmdet/datasets")
# from dior import DIORDataset
# sys.path.append("../../../MTP-main/MTP-main/RS_Tasks_Finetune/Horizontal_Detection/mmdet/datasets")
# from xview import XviewDataset
# import torchvision
# torchvision.disable_beta_transforms_warning()
# from PIL import Image
# import io
# # if 'RVSA_MTP_branches' in MODELS:
# #     print("RVSA_MTP_branches 已注册")
# # else:
# #     print("RVSA_MTP_branches 未注册")


# DIOR_classes = [
#     "Airplane",  # 类别 0
#     "Airport",
#     "Baseball field",  # 类别 2
#     "Basketball court",  # 类别 1
#     "Bridge",  # 类别 3
#     "Chimney",  # 类别 6
#     "Dam",  # 类别 7
#     "Expressway Service area",  # 类别 8
#     "Expressway toll station",
#     "Golf course",
#     "Ground track field",
#     "Harbor",  # 类别 9
#     "Overpass",  # 类别 10
#     "Ship",  # 类别 22
#     "Stadium",  # 类别 23
#     "Storage tank",  # 类别 26
#     "Tennis court",  # 类别 28
#     "Train station",  # 类别 29
#     "Vehicle",  # 类别 30
#     "Windmill"
# ]

# Xview_class = [
#     'Fixed-wing Aircraft',  'Small Aircraft',  'Cargo Plane',  'Helicopter',  
#          'Passenger Vehicle',  'Small Car',  'Bus',  'Pickup Truck',  
#          'Utility Truck',  'Truck',  'Cargo Truck',  'Truck w/Box',  
#          'Truck Tractor',  'Trailer',  'Truck w/Flatbed',  'Truck w/Liquid',  
#          'Crane Truck',  'Railway Vehicle',  'Passenger Car',  'Cargo Car',  
#          'Flat Car',  'Tank car',  'Locomotive',  'Maritime Vessel',  
#          'Motorboat',  'Sailboat',  'Tugboat',  'Barge',  
#          'Fishing Vessel',  'Ferry',  'Yacht',  'Container Ship',  
#          'Oil Tanker',  'Engineering Vehicle',  'Tower crane',  'Container Crane',  
#          'Reach Stacker',  'Straddle Carrier',  'Mobile Crane',  'Dump Truck',  
#          'Haul Truck',  'Scraper/Tractor',  'Front loader/Bulldozer',  'Excavator',  
#          'Cement Mixer',  'Ground Grader',  'Hut/Tent',  'Shed',  
#          'Building',  'Aircraft Hangar',  'Damaged Building',  'Facility',  
#          'Construction Site',  'Vehicle Lot',  'Helipad',  'Storage Tank',  
#          'Shipping container lot',  'Shipping Container',  'Pylon',  'Tower'
# ]

# def load_model(config_file, checkpoint_file):
#     """
#     加载检测模型。
#     :param config_file: 配置文件路径
#     :param checkpoint_file: 检查点文件路径
#     :return: 加载后的模型
#     """
#     cfg = Config.fromfile(config_file)
#     cfg.model.pretrained = None  # 禁用预训练模型加载
#     model = init_detector(config_file, checkpoint_file, device='cuda:0')
#     # model = init_detector(config_file, checkpoint_file, device='cpu')
#     return model

# import matplotlib.pyplot as plt

# def detect_objects(model, image_path, score_thr=0.2):
#     """
#     使用模型检测图像中的物体。
#     :param model: 加载的检测模型
#     :param image_path: 图像文件路径
#     :param score_thr: 置信度阈值
#     :return: 检测结果
#     """
#     # 读取图像
#     # img = cv2.imread(image_path)
#     img = image_path

#     if img is None:
#         raise FileNotFoundError(f"图像 {image_path} 不存在或无法读取。")
    
#     height, width, channels = img.shape    
#     # 推理
#     results = inference_detector(model, img)

#     filtered_results = []
#     # print(results.pred_instances)
#     # for label_id, bboxes in enumerate(results):
#     #     for bbox in bboxes:
#     #         score = bbox[4]
#     #         if score >= score_thr:
#     #             filtered_results.append({
#     #                 "label": model.CLASSES[label_id],
#     #                 "bbox": bbox[:4].tolist(),
#     #                 "score": score
#     #             })
#     # 假设 detect_objects 返回的是 DetDataSample 对象
#     for label_id, bboxes in enumerate(results.pred_instances):
#     # 如果你想获取每个检测框的标签
#         # print(label_id,bboxes)
#         filtered_results.append((label_id,bboxes))
#         # print(f"Label: {bboxes.pred_classes[label_id]}, Bounding Box: {bboxes.bboxes[label_id]}")
    
#     results =  filtered_results
    
#     ans_str = ''
#     for _,res in results:
#         # ans_str = ans_str + 'object: '+ str(Xview_class[int(res.labels.cpu().numpy())]) + ', bbox: '+  str(res.bboxes.cpu().numpy()/height)+ ', scores: '+str(res.scores.cpu().numpy()) +'; '
#         ans_str = ans_str + 'object: '+ str(Xview_class[int(res.labels.cpu().numpy())]) + ', bbox: '+  str(res.bboxes.cpu().numpy()[0]/height)+ '; '
#     # item['DIOR_info_sat_'+str(0)] = ans_str
    
#     return ans_str

# def detect_image_sat(IMAGE_PATH):
#     # 配置文件和权重文件路径
#     config_path = '../../../MTP-main/MTP-main/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/xview/retinanet_rvsa_l_416_mae_mtp_xview.py'
#     checkpoint_path = weight_path('xview-rvsa-l-mae-mtp_epoch_12.pth')
#     # config_path = mmdet_config('faster_rcnn_rvsa_b_800_mae_mtp_dior.py')
#     # checkpoint_path = weight_path('dior-rvsa-b-mae-mtp-epoch_12.pth')
    
#     # 输入图像路径
#     image_path = IMAGE_PATH # 'DIOR_test.png'
    
#     # 加载模型
#     model = load_model(config_path, checkpoint_path)
    
#     # 检测物体
#     results = detect_objects(model, image_path)
#     print('final_detection_results_xview: ',results)

#     return results

# # detect_image_sat("49617_107871.png")
# # ########################################################################################################################
# app = FastAPI()

# @app.post("/inference_xview_one_image")
# async def inference_xview_one_image_endpoint(file: UploadFile = File(...)):
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

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8002)
#     #10.112.27.164


import torch
import json
import matplotlib.patches as patches
import sys
import os # 导入 os 模块用于路径操作

# 确保导入路径正确，这里使用 os.path.abspath 和 os.path.join 确保路径的鲁棒性
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BACKBONE_SUPPORT_ROOT)`r`nsys.path.append(DATASET_SUPPORT_ROOT)`r`n

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import MODELS
from vit_rvsa_mtp_branches import RVSA_MTP_branches # 确保这个模块被正确导入和注册
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import numpy as np
import cv2
from collections import defaultdict
import pandas as pd
from dior import DIORDataset # 确保 DIORDataset 被注册
from xview import XviewDataset # 确保 XviewDataset 被注册
import torchvision
torchvision.disable_beta_transforms_warning()
from PIL import Image
import io

# 全局变量来存储加载的模型
global_xview_model = None

# DIOR_classes 定义不变，省略...
DIOR_classes = [
    "Airplane",  # 类别 0
    "Airport",
    "Baseball field",  # 类别 2
    "Basketball court",  # 类别 1
    "Bridge",  # 类别 3
    "Chimney",  # 类别 6
    "Dam",  # 类别 7
    "Expressway Service area",  # 类别 8
    "Expressway toll station",
    "Golf course",
    "Ground track field",
    "Harbor",  # 类别 9
    "Overpass",  # 类别 10
    "Ship",  # 类别 22
    "Stadium",  # 类别 23
    "Storage tank",  # 类别 26
    "Tennis court",  # 类别 28
    "Train station",  # 类别 29
    "Vehicle",  # 类别 30
    "Windmill"
]

Xview_class = [
    'Fixed-wing Aircraft',   'Small Aircraft',   'Cargo Plane',   'Helicopter',
    'Passenger Vehicle',   'Small Car',   'Bus',   'Pickup Truck',
    'Utility Truck',   'Truck',   'Cargo Truck',   'Truck w/Box',
    'Truck Tractor',   'Trailer',   'Truck w/Flatbed',   'Truck w/Liquid',
    'Crane Truck',   'Railway Vehicle',   'Passenger Car',   'Cargo Car',
    'Flat Car',   'Tank car',   'Locomotive',   'Maritime Vessel',
    'Motorboat',   'Sailboat',   'Tugboat',   'Barge',
    'Fishing Vessel',   'Ferry',   'Yacht',   'Container Ship',
    'Oil Tanker',   'Engineering Vehicle',   'Tower crane',   'Container Crane',
    'Reach Stacker',   'Straddle Carrier',   'Mobile Crane',   'Dump Truck',
    'Haul Truck',   'Scraper/Tractor',   'Front loader/Bulldozer',   'Excavator',
    'Cement Mixer',   'Ground Grader',   'Hut/Tent',   'Shed',
    'Building',   'Aircraft Hangar',   'Damaged Building',   'Facility',
    'Construction Site',   'Vehicle Lot',   'Helipad',   'Storage Tank',
    'Shipping container lot',   'Shipping Container',   'Pylon',   'Tower'
]

def load_model(config_file, checkpoint_file):
    """
    加载检测模型。
    :param config_file: 配置文件路径
    :param checkpoint_file: 检查点文件路径
    :return: 加载后的模型
    """
    print("Loading Xview model...")
    cfg = Config.fromfile(config_file)
    # 移除 cfg.model.pretrained = None，因为 MMDetection 3.x 不再接受此参数
    # cfg.model.pretrained = None 
    
    device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_detector(cfg, checkpoint_file, device=device) # init_detector 可以直接接受 Config 对象
    model.eval() # 将模型设置为评估模式
    print(f"Xview Model loaded on {device}.")
    return model

def detect_objects(model, image_array, score_thr=0.2):
    """
    使用模型检测图像中的物体。
    :param model: 加载的检测模型
    :param image_array: 图像的 numpy 数组
    :param score_thr: 置信度阈值
    :return: 检测结果字符串
    """
    if image_array is None:
        raise ValueError("输入图像数组为空。")
    
    # 确保图像是三通道的，即使是灰度图也转为RGB
    if len(image_array.shape) == 2: # 灰度图
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 4: # RGBA图
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

    height, width, _ = image_array.shape
    
    # 推理
    with torch.no_grad(): # 推理时禁用梯度计算，节省内存和加速
        results = inference_detector(model, image_array)

    ans_str = ''
    # results 是一个 DetDataSample 对象，其 pred_instances 包含了检测到的所有实例信息
    # .scores, .bboxes, .labels 是其属性
    
    if hasattr(results, 'pred_instances') and results.pred_instances is not None:
        for i in range(len(results.pred_instances)):
            label_id = results.pred_instances.labels[i].item() # 获取标签ID
            score = results.pred_instances.scores[i].item() # 获取分数
            bbox = results.pred_instances.bboxes[i].cpu().numpy() # 获取边界框

            if score >= score_thr: # 根据置信度阈值过滤
                # 检查标签ID是否在 Xview_class 范围内
                if 0 <= label_id < len(Xview_class):
                    object_name = Xview_class[label_id]
                else:
                    object_name = f"Unknown_Label_{label_id}" # 处理未知标签

                # 格式化输出字符串。注意：bbox 的归一化方式可能需要你再次确认。
                # 原始代码是 bbox[0]/height, 这看起来只取了第一个坐标且只除了height，
                # 如果是 [x1, y1, x2, y2] 形式，通常会是 x1/width, y1/height, x2/width, y2/height
                # 我这里先保持原样，但建议检查模型输出和预期归一化方式。
                # 如果你想要 [x1, y1, x2, y2] 的列表形式，且不归一化，就用 bbox.tolist()
                ans_str += f'object: {object_name}, bbox: {bbox.tolist()}, scores: [{score}]; ' 
                # ans_str += f'object: {object_name}, bbox: {bbox.tolist()}; ' # 如果不需要score

    return ans_str

# ########################################################################################################################
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    FastAPI 应用启动时加载 Xview 模型
    """
    global global_xview_model
    config_path = mmdet_config('retinanet_rvsa_l_416_mae_mtp_xview.py')
    checkpoint_path = weight_path('xview-rvsa-l-mae-mtp_epoch_12.pth')
    
    # 检查文件路径是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    global_xview_model = load_model(config_path, checkpoint_path)


@app.post("/inference_xview_one_image")
async def inference_xview_one_image_endpoint(file: UploadFile = File(...)):
    """
    接收图像文件，进行目标检测并返回结果
    """
    try:
        # 确保模型已经加载
        if global_xview_model is None:
            return {"status": "error", "error": "Xview Model not loaded. Please restart the application."}

        # 1. 读取上传文件的内容
        contents = await file.read() # UploadFile 的 read() 方法是异步的，返回字节流

        # 2. 将字节流读取为 PIL 图像
        image = Image.open(io.BytesIO(contents)).convert("RGB") # 确保图像是RGB格式

        # 3. 将图像转换为 numpy 数组 (H, W, C)
        image_array = np.array(image)
        
        # 调用检测模型，直接传入已加载的 global_xview_model
        detection_result = detect_objects(global_xview_model, image_array)
        
        # 打印到控制台，以便调试
        print('final_detection_results_xview: ', detection_result)
        
        # 返回结果
        return {"status": "success", "result": detection_result}

    except Exception as e:
        import traceback
        traceback.print_exc() # 这行会打印详细的错误堆栈信息
        return {"status": "error", "error": f"处理文件时发生错误: {str(e)}"}

if __name__ == '__main__':
    # 运行 FastAPI 应用
    # host 设为 "0.0.0.0" 可以从外部访问，如果仅本机访问则用 "127.0.0.1"
    uvicorn.run(app, host="127.0.0.1", port=8002)
