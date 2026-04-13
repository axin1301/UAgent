from common_paths import mmseg_config, weight_path, BACKBONE_SUPPORT_ROOT
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner
from mmcv.image import imread, imwrite,imresize
from mmseg.registry import MODELS, DATASETS, TRANSFORMS
# from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis import inference_model, init_model
import sys
sys.path.append(BACKBONE_SUPPORT_ROOT)`r`nfrom vit_rvsa_mtp import RVSA_MTP
from fastapi import FastAPI, File, UploadFile,Form
import uvicorn
import numpy as np
import cv2
from collections import defaultdict

# def parse_args():
#     parser = argparse.ArgumentParser(description='Semantic Segmentation for a Single Image')
#     parser.add_argument('config', help='Config file path')
#     parser.add_argument('checkpoint', help='Checkpoint file path')
#     parser.add_argument('image', help='Path to the input image')
#     parser.add_argument('output', help='Path to save the output segmented image')
#     args = parser.parse_args()
#     return args


def segment_image_sat(IMAGE_PATH):
    class_list = [' ',' ','building','road','water','barren','forest','agriculture']
    # args = parse_args()

    # Load configuration
    # config_path = 'D:/OneDrive - University of Helsinki/MLLM-next/MTP-main/MTP-main/RS_Tasks_Finetune/Semantic_Segmentation/configs/mtp/spacenetv1/rvsa-b-upernet-384-mae-mtp-spacenetv1.py'
    # # IMAGE_PATH = 'DIOR_test.png'
    # IMAGE_PATH = '49608_107898.png'
    # checkpoint_path = 'D:/OneDrive - University of Helsinki/MLLM-next/MTP-main/MTP-main/RS_Tasks_Finetune/Semantic_Segmentation/spacenetv1-rvsa-b-mae-mtp-iter_80000.pth'#spacenetv1

    config_path = mmseg_config('rvsa-b-upernet-512-mae-mtp-loveda.py')
    # IMAGE_PATH = 'DIOR_test.png'
    # IMAGE_PATH = '49608_107898.png'
    checkpoint_path = weight_path('loveda-rvsa-b-mae-mtp-iter_80000.pth')#spacenetv1

    cfg = Config.fromfile(config_path)

    # Initialize the segmentation model
    model = init_model(cfg, checkpoint_path, device = 'cpu') #device='cuda:0')

    # Read the input image
    image = imread(IMAGE_PATH)
    image= imresize(image, (512, 512))

    # Perform inference
    result = inference_model(model, image)

    # Save the segmentation result
    if isinstance(result, list):
        # If the result contains multiple outputs (e.g., TTA), use the first one
        result = result[0]
    # print(result)



    # # 提取分割结果
    # pred_seg = result.pred_sem_seg.data.squeeze(0).numpy()  # [H, W]

    # # 定义颜色映射
    # num_classes = 7  # 类别数
    # colors = np.random.randint(0, 255, size=(num_classes, 3))  # 随机颜色

    # # 映射到伪彩图
    # colored_seg = np.zeros((*pred_seg.shape, 3), dtype=np.uint8)
    # for class_id in range(num_classes):
    #     colored_seg[pred_seg == class_id] = colors[class_id]

    # # 显示结果
    # original_image = image
    # overlay = (0.5 * original_image + 0.5 * colored_seg).astype(np.uint8)
    # plt.imshow(overlay)
    # plt.axis('off')
    # plt.show()
    # # plt.imshow(colored_seg)
    # # plt.axis('off')
    # # plt.show()
    # # imwrite(result, args.output)

    # # print(f"Segmentation result saved to: {args.output}")

    # 假设 pred_seg 是分割结果 [H, W]，每个像素值是类别编号
    # 假设 pred_seg 是分割结果 [H, W]，每个像素值是类别编号
    pred_seg = result.pred_sem_seg.data.squeeze(0).numpy()
    num_classes = 8  # 假设有7类

    # 初始化统计结果
    height, width = pred_seg.shape
    total_pixels = height * width
    class_pixel_count = defaultdict(int)
    class_contours = defaultdict(list)

    # 简化轮廓的阈值
    epsilon_factor = 0.02  # 轮廓边界简化程度，越大点越少
    ans_str = ''

    for cls in range(1,num_classes):
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

        # 输出当前类别信息
        print(class_list[cls])
        print(f"  Pixel Count: {class_pixel_count[cls]} ({pixel_ratio:.2%})")
        print(f"  Simplified Contours: {class_contours[cls]}")
        ans_str = ans_str + str(class_list[cls]) + ' landuse, pixel percentage: '+ str(pixel_ratio)[:6] + ', boundary: '+  str(class_contours[cls]) +'; '
    return ans_str

app = FastAPI()

# # 假设这是你的分割模型逻辑
# def segment_image(image: np.ndarray) -> str:
#     """
#     图像分割逻辑（占位示例）
#     输入：numpy 格式的图像
#     输出：分割结果（字符串形式）
#     """
#     # 这里可以调用你的模型进行推断
#     # 示例输出
#     return "Segmentation result: example segmentation data"

@app.post("/segment_image_sat")
async def segment_image_endpoint(file_path: str = Form(...)):#(file: UploadFile = File(...)):
    """
    接收图像文件，进行分割并返回结果
    """
    try:
        # # 将上传的文件读取为 PIL 图像
        # image = Image.open(file.file)
        
        # # 将图像转换为 numpy 数组
        # image_array = np.array(image)
        
        # 调用分割模型
        segmentation_result = segment_image_sat(file_path)#image_array)
        
        # 返回结果
        return {
            "status": "success",
            "segmentation_result": segmentation_result
        }
        # return segmentation_result
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    #10.112.27.164

