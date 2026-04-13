from common_paths import mmseg_config, weight_path, BACKBONE_SUPPORT_ROOT
import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner
from mmcv.image import imread, imwrite,imresize
from mmseg.registry import MODELS, DATASETS, TRANSFORMS
# from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis import inference_model, init_model
import sys
sys.path.append(BACKBONE_SUPPORT_ROOT)`r`nfrom vit_rvsa_mtp import RVSA_MTP

import numpy as np
import cv2
from collections import defaultdict
import glob

# def parse_args():
#     parser = argparse.ArgumentParser(description='Semantic Segmentation for a Single Image')
#     parser.add_argument('config', help='Config file path')
#     parser.add_argument('checkpoint', help='Checkpoint file path')
#     parser.add_argument('image', help='Path to the input image')
#     parser.add_argument('output', help='Path to save the output segmented image')
#     args = parser.parse_args()
#     return args

import json

def main():
    # with open('Beijing_STV_SAT_mapping_test_loveda.json', "r") as file:
    #     data = json.load(file)
    # with open('Beijing_STV_SAT_location_test_loveda.json', "r") as file:
    #     data = json.load(file)
    with open('Beijing_sat_landuse_mc_zl17_test_loveda.json', "r") as file:
        data = json.load(file)
    # with open('Beijing_SAT_count_pois_zl17_test_loveda.json', "r") as file:
    #     data = json.load(file)
    # with open('Beijing_SAT_count_buildings_zl17_test_loveda.json', "r") as file:
    #     data = json.load(file)
    # with open('Beijing_sat_address_mc_zl17_test_loveda.json', "r") as file:
        # data = json.load(file)
    image_name_list = [image_path.split('/')[-1].split('.')[0] for item in data for image_path in [item["image"]]]#[1:]
# # os.makedirs('sat_BJ_mix',exist_ok=True)
# # for item in data:
# #     image_paths = item["image"]  # 提取图像路径
# #     # print(image_path)
# #     for image_path in image_paths[1:]:
# #         shutil.copy(image_path,'sat_BJ_mix/'+image_path.split('/')[-1])
    class_list = ['background','building','road','water','barren','forest','agriculture']#'no_data',
    # args = parse_args()

    # Load configuration
    
    set_width = 1024 #2048 #512 #1024  #12395_26956
        
        # num_classes = 2  # 假设有7类
        # config_path = 'D:/OneDrive - University of Helsinki/MLLM-next/MTP-main/MTP-main/RS_Tasks_Finetune/Semantic_Segmentation/configs/mtp/spacenetv1/rvsa-b-upernet-384-mae-mtp-spacenetv1.py'
        # # IMAGE_PATH = 'DIOR_test.png'
        # # IMAGE_PATH = '49608_107898.png'
        # checkpoint_path = 'D:/OneDrive - University of Helsinki/MLLM-next/MTP-main/MTP-main/RS_Tasks_Finetune/Semantic_Segmentation/spacenetv1-rvsa-b-mae-mtp-iter_80000.pth'#spacenetv1

    num_classes = 7  # 假设有7类
    config_path = mmseg_config('rvsa-b-upernet-512-mae-mtp-loveda.py')
    # IMAGE_PATH = 'DIOR_test.png'
    # IMAGE_PATH = '49608_107898.png'
    checkpoint_path = weight_path('loveda-rvsa-b-mae-mtp-iter_80000.pth')#spacenetv1

    cfg = Config.fromfile(config_path)

    # Initialize the segmentation model
    model = init_model(cfg, checkpoint_path, device = 'cuda:0')
    # image_name_list = ['12395_26956']
    # image_name_list = list(pd.read_csv('../../../reproduce_citybench/BJ_urbanllava_zl17.csv')['img_name'])
    image_name_list = glob.glob('../../../London_urbanllava_zl17_merge/*.png') + glob.glob('../../../London_citybench_zl17_merge/*.png') + glob.glob('../../../London/*.png') + glob.glob('../../../London_image/*.png')
    for img_name in image_name_list:
        plt.close()
    # if 1:
        # img_name = '49617_107871'#'49616_107912'#'49631_107902'
        # IMAGE_PATH= '../../../reproduce_citybench/merged_zl17_2020/'+img_name+'.png' #12398_26956.png'#test_resampled_sat_image/12395_26958_bicubic_4.png'#_crop_bl.png'#12396_26972.png'
        # IMAGE_PATH= '../../../reproduce_citybench/merged_zl17_2020_BJ_urbanllava/'+img_name
        # IMAGE_PATH= '../../../reproduce_citybench/merged_zl17_2020_BJ_urbanllava/'+img_name
        # IMAGE_PATH= '../../../sat_BJ_mix/sat_BJ_mix/'+img_name+'.png'
    # image_name_list = glob.glob('../../../reproduce_citybench/merged_zl17/*.png')
    # for IMAGE_PATH in image_name_list:
        IMAGE_PATH = img_name
        img_name = IMAGE_PATH.split('\\')[-1].split('.')[0]
        # img_name = img_name.split('.')[0]
        print(img_name)
        # if not os.path.exists(IMAGE_PATH):
        #     continue

        if os.path.exists('D:/OneDrive - University of Helsinki/MLLM-next/API_test/mapping_stv_sat_example_images/'+img_name+'_semseg.png'):
            continue

        # Read the input image
        image = imread(IMAGE_PATH)
        image= imresize(image, (set_width, set_width))

        # Perform inference
        result = inference_model(model, image)

        # Save the segmentation result
        if isinstance(result, list):
            # If the result contains multiple outputs (e.g., TTA), use the first one
            result = result[0]
        # print(result)

        # 假设 pred_seg 是分割结果 [H, W]，每个像素值是类别编号
        # 假设 pred_seg 是分割结果 [H, W]，每个像素值是类别编号
        # pred_seg = result.pred_sem_seg.data.squeeze(0).numpy()
        
        # # 初始化统计结果
        # height, width = pred_seg.shape
        # total_pixels = height * width
        # class_pixel_count = defaultdict(int)
        # class_contours = defaultdict(list)

        # # 简化轮廓的阈值
        # epsilon_factor = 0.01  # 轮廓边界简化程度，越大点越少

        # for cls in range(num_classes):
        #     # 提取当前类别的二值掩码
        #     mask = (pred_seg == cls).astype(np.uint8)

        #     # 计算像素占比
        #     class_pixel_count[cls] = np.sum(mask)
        #     pixel_ratio = class_pixel_count[cls] / total_pixels

        #     # 提取轮廓
        #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #     # 简化轮廓
        #     simplified_contours = []
        #     for contour in contours:
        #         if len(contour) > 0:
        #             # 根据轮廓的周长和 epsilon_factor 简化轮廓
        #             epsilon = epsilon_factor * cv2.arcLength(contour, True)
        #             approx = cv2.approxPolyDP(contour, epsilon, True)
        #             simplified_contours.append(approx.squeeze(1).tolist())

        #     # 保存简化后的轮廓
        #     class_contours[cls] = simplified_contours

        #     # 输出当前类别信息
        #     print(class_list[cls])
        #     print(f"  Pixel Count: {class_pixel_count[cls]} ({pixel_ratio:.2%})")
        #     # print(f"  Simplified Contours: {class_contours[cls]}")
        #     # 提取分割结果
        # pred_seg = result.pred_sem_seg.data.squeeze(0).numpy()  # [H, W]

        # # 定义颜色映射
        # # num_classes = 7  # 类别数
        # colors = np.random.randint(0, 255, size=(num_classes, 3))  # 随机颜色

        # # 映射到伪彩图
        # colored_seg = np.zeros((*pred_seg.shape, 3), dtype=np.uint8)
        # for class_id in range(num_classes):
        #     colored_seg[pred_seg == class_id] = colors[class_id]
        #     print(colors[class_id])

        # # 显示结果
        # original_image = image
        # overlay = (0.5 * original_image + 0.5 * colored_seg).astype(np.uint8)
        # # plt.imshow(overlay)
        # # plt.axis('off')
        # # plt.show()
        # plt.imshow(colored_seg)



        pred_seg = result.pred_sem_seg.cpu().data.squeeze(0).numpy()

        # 初始化统计结果
        height, width = pred_seg.shape
        total_pixels = height * width
        class_pixel_count = defaultdict(int)
        class_contours = defaultdict(list)

        # 简化轮廓的阈值
        epsilon_factor = 0.1  # 轮廓边界简化程度，越大点越少

        for cls in range(num_classes):
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

        # 定义每个标签的颜色 (这里定义了 7 类，每个类的颜色对应你指定的 RGB 值)
        # 'background','building','road','water','barren','forest','agriculture'
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
                print(f"Color for class {class_id}: {color_map[class_id]}", class_list[class_id])

        # 显示结果
        original_image = image
        overlay = (colored_seg).astype(np.uint8) #0.5 * original_image + 0.5 * colored_seg).astype(np.uint8)
        # 将 NumPy 数组转换为 PIL 图像
        overlay_pil = Image.fromarray(overlay)

        # 调整尺寸
        overlay_new = overlay_pil.resize((256, 256))
        
        # 显示或者保存 overlay 图像
        plt.imshow(overlay_new)
        plt.axis('off')
        plt.savefig('D:/OneDrive - University of Helsinki/MLLM-next/API_test/mapping_stv_sat_example_images/'+img_name+'_semseg.png', bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.show()
        # plt.clf()
    # 
    # plt.show()
    # # imwrite(result, args.output)

    # # print(f"Segmentation result saved to: {args.output}")

    # 最终结果
    # print("Class Pixel Counts:", class_pixel_count)
    # print("Class Contours:", class_contours)
###background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7.

if __name__ == '__main__':
    main()
