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
#         # ans_str = ans_str + 'object: '+ str(DIOR_classes[int(res.labels.cpu().numpy())]) + ', bbox: '+  str(res.bboxes.cpu().numpy()/height)+ ', scores: '+str(res.scores.cpu().numpy()) +'; '
#         ans_str = ans_str + 'object: '+ str(DIOR_classes[int(res.labels.cpu().numpy())]) + ', bbox: '+  str(res.bboxes.cpu().numpy()[0]/height)+ '; '
#     # item['DIOR_info_sat_'+str(0)] = ans_str
    
#     return ans_str

# def detect_image_sat(IMAGE_PATH):
#     # 配置文件和权重文件路径
#     # config_path = '/data3/xiyanxin/RVSA/Remote-Sensing-RVSA/Object Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.py'
#     # checkpoint_path = '/data3/xiyanxin/RVSA/Remote-Sensing-RVSA/Object Detection/vitae_rvsa_kvdiff_new.pth'
#     config_path = '../../../MTP-main/MTP-main/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/faster_rcnn_rvsa_b_800_mae_mtp_dior.py'
#     checkpoint_path = weight_path('dior-rvsa-b-mae-mtp-epoch_12.pth')
    
#     # 输入图像路径
#     image_path = IMAGE_PATH # 'DIOR_test.png'
    
#     # 加载模型
#     model = load_model(config_path, checkpoint_path)
    
#     # 检测物体
#     results = detect_objects(model, image_path)
#     print('final_detection_results_DIOR: ',results)

#     return results

# # detect_image_sat("49617_107871.png")
# # ########################################################################################################################
# app = FastAPI()

# @app.post("/inference_DIOR_one_image")
# async def inference_DIOR_one_image_endpoint(file: UploadFile = File(...)):
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
global_model = None

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

# Xview_class 定义不变，省略...
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
    print("Loading model...")
    cfg = Config.fromfile(config_file)
    
    # 移除或注释掉这一行，因为 MMDetection 3.x 不再接受 'pretrained' 参数
    # cfg.model.pretrained = None  # 禁用预训练模型加载
    
    # 确保设备设置正确，如果使用GPU，确认CUDA可用
    device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # init_detector 会负责加载 checkpoint_file
    model = init_detector(cfg, checkpoint_file, device=device) 
    model.eval() # 将模型设置为评估模式
    print(f"Model loaded on {device}.")
    return model

def detect_objects(model, image_array, score_thr=0.2):
    """
    使用模型检测图像中的物体。
    :param model: 加载的检测模型
    :param image_array: 图像的 numpy 数组
    :param score_thr: 置信度阈值
    :return: 检测结果
    """
    if image_array is None:
        raise ValueError("输入图像数组为空。")

    height, width, _ = image_array.shape 
    
    # 推理
    # inference_detector 期望模型处于评估模式
    with torch.no_grad(): # 推理时禁用梯度计算，节省内存和加速
        results = inference_detector(model, image_array)

    ans_str = ''
    # 遍历 results.pred_instances 来获取每个检测到的对象
    # results 是一个 DetDataSample 对象
    # .pred_instances 包含了检测到的所有实例信息
    # .scores, .bboxes, .labels 是其属性
    
    # 确保 results.pred_instances 存在且不为空
    if hasattr(results, 'pred_instances') and results.pred_instances is not None:
        for i in range(len(results.pred_instances)):
            label_id = results.pred_instances.labels[i].item() # 获取标签ID
            score = results.pred_instances.scores[i].item() # 获取分数
            bbox = results.pred_instances.bboxes[i].cpu().numpy() # 获取边界框

            if score >= score_thr: # 根据置信度阈值过滤
                # 检查标签ID是否在DIOR_classes范围内
                if 0 <= label_id < len(DIOR_classes):
                    object_name = DIOR_classes[label_id]
                else:
                    object_name = f"Unknown_Label_{label_id}" # 处理未知标签

                # 格式化输出字符串
                # 注意：这里 /height 的处理似乎只针对了y轴（height）的归一化，x轴（width）没有，
                # 如果你的bbox是 [x1, y1, x2, y2] 并且需要归一化，通常是 x/width, y/height
                # 但根据你原始代码的 /height，我保持了它。请根据实际模型输出和需求调整归一化。
                # ans_str += f'object: {object_name}, bbox: {bbox.tolist()}; ' 
                # 如果需要scores，可以加回去
                ans_str += f'object: {object_name}, bbox: {bbox.tolist()}, scores: [{score}]; ' 
    
    return ans_str

# ########################################################################################################################
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    FastAPI 应用启动时加载模型
    """
    global global_model
    config_path = mmdet_config('faster_rcnn_rvsa_b_800_mae_mtp_dior.py')
    checkpoint_path = weight_path('dior-rvsa-b-mae-mtp-epoch_12.pth')
    
    # 检查文件路径是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    global_model = load_model(config_path, checkpoint_path)


@app.post("/inference_DIOR_one_image")
async def inference_DIOR_one_image_endpoint(file: UploadFile = File(...)):
    """
    接收图像文件，进行目标检测并返回结果
    """
    try:
        # 确保模型已经加载
        if global_model is None:
            return {"status": "error", "error": "Model not loaded. Please restart the application."}

        # 1. 读取上传文件的内容
        contents = await file.read() # UploadFile 的 read() 方法是异步的，返回字节流

        # 2. 将字节流读取为 PIL 图像
        image = Image.open(io.BytesIO(contents)).convert("RGB") # 确保图像是RGB格式

        # 3. 将图像转换为 numpy 数组 (H, W, C)
        image_array = np.array(image)
        
        # 调用检测模型，直接传入已加载的 global_model
        # 注意：这里传递的是 numpy 数组，而不是文件路径
        detection_result = detect_objects(global_model, image_array)
        
        # 打印到控制台，以便调试
        print('final_detection_results_DIOR: ', detection_result)
        
        # 返回结果
        return {"status": "success", "result": detection_result}

    except Exception as e:
        import traceback
        traceback.print_exc() # 这行会打印详细的错误堆栈信息
        return {"status": "error", "error": f"处理文件时发生错误: {str(e)}"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
    #10.112.27.164

