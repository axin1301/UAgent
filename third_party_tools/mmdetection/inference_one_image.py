from common_paths import mmdet_config, weight_path, BACKBONE_SUPPORT_ROOT, DATASET_SUPPORT_ROOT
import torch
import sys
# from mmcv import Config
from mmengine.config import Config
from mmengine.runner import load_checkpoint#,get_dist_info,init_dist
# from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector
import cv2
from mmdet.registry import MODELS
import sys
sys.path.append(BACKBONE_SUPPORT_ROOT)`r`nfrom vit_rvsa_mtp_branches import RVSA_MTP_branches

import sys
sys.path.append(DATASET_SUPPORT_ROOT)`r`nfrom dior import DIORDataset


if 'RVSA_MTP_branches' in MODELS:
    print("RVSA_MTP_branches 已注册")
else:
    print("RVSA_MTP_branches 未注册")


def load_model(config_file, checkpoint_file):
    """
    加载检测模型。
    :param config_file: 配置文件路径
    :param checkpoint_file: 检查点文件路径
    :return: 加载后的模型
    """
    cfg = Config.fromfile(config_file)
    cfg.model.pretrained = None  # 禁用预训练模型加载
    #model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model = init_detector(config_file, checkpoint_file, device='cpu')
    return model

def detect_objects(model, image_path, score_thr=0.3):
    """
    使用模型检测图像中的物体。
    :param model: 加载的检测模型
    :param image_path: 图像文件路径
    :param score_thr: 置信度阈值
    :return: 检测结果
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像 {image_path} 不存在或无法读取。")
    
    # 推理
    results = inference_detector(model, img)
    
    # 过滤低置信度的结果
    filtered_results = []
    print(results.pred_instances)
    # for label_id, bboxes in enumerate(results):
    #     for bbox in bboxes:
    #         score = bbox[4]
    #         if score >= score_thr:
    #             filtered_results.append({
    #                 "label": model.CLASSES[label_id],
    #                 "bbox": bbox[:4].tolist(),
    #                 "score": score
    #             })
    # 假设 detect_objects 返回的是 DetDataSample 对象
    for label_id, bboxes in enumerate(results.pred_instances):
    # 如果你想获取每个检测框的标签
        print(label_id,bboxes)
        filtered_results.append((label_id,bboxes))
        # print(f"Label: {bboxes.pred_classes[label_id]}, Bounding Box: {bboxes.bboxes[label_id]}")
    
    return filtered_results

if __name__ == '__main__':
    # 配置文件和权重文件路径
    # config_path = '/data3/xiyanxin/RVSA/Remote-Sensing-RVSA/Object Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.py'
    # checkpoint_path = '/data3/xiyanxin/RVSA/Remote-Sensing-RVSA/Object Detection/vitae_rvsa_kvdiff_new.pth'
    config_path = mmdet_config('faster_rcnn_rvsa_b_800_mae_mtp_dior.py')
    checkpoint_path = weight_path('dior-rvsa-b-mae-mtp-epoch_12.pth')
    
    # 输入图像路径
    image_path = '../../../UrbanAgent/ImageData/NewYork_urbanllava/49228_38630.png'
    # 'DIOR_test.png'
    
    # 加载模型
    model = load_model(config_path, checkpoint_path)
    
    # 检测物体
    results = detect_objects(model, image_path)
    
    # 输出结果
    print("检测结果:")
    for res in results:
        print(f"物体: {res['label']}, 坐标: {res['bbox']}, 置信度: {res['score']:.2f}")

