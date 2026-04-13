
import tqdm
import re
import base64
import json
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
# 读取本地图像文件
# 读取本地图像文件
import re
from collections import Counter
import math
import numpy as np
import re
from Tool_library_sup_funcs_LN_here import *
import sys
from llm_api import *
from skimage.morphology import skeletonize, disk
from skimage.color import rgb2gray
from skimage import img_as_bool
import os
from return_stv_process_image_semseg import *
from return_sat_process_image_semseg import *
# sys.path.append('../Framework_v18/')
# from MLLM_verifier import *
from requestAPI import *
import cv2
from collections import defaultdict
from pathlib import Path

def super_res_function(img_path):
    img_name = img_path.split('/')[-1].split('.')[0]
    cnt_underline = img_name.count('_')
    print(img_name)

    if cnt_underline>2:
        try:
            for region in ['Beijing','London','NewYork']:
                for region_part in ['urbanllava','citybench']:
                    if os.path.exists('../ImageData/'+region+'_'+region_part+'_zl17_merge/'+img_name.split('_')[0]+'_'+img_name.split('_')[1]+'.png'):
                        new_image_path = '../ImageData/'+region+'_'+region_part+'_zl17_merge/'+img_name.split('_')[0]+'_'+img_name.split('_')[1]+'.png'
                    
                        x_idx = int(img_name.split('_')[2])
                        y_idx = int(img_name.split('_')[3])


                        img = Image.open(new_image_path)
                        img_width, img_height = img.size

                        row_idx = x_idx
                        col_idx = y_idx

                        # 2. 计算裁剪区域 (left, upper, right, lower)
                        half_width = img_width // 2
                        half_height = img_height // 2

                        left = col_idx * half_width
                        upper = row_idx * half_height
                        right = left + half_width
                        lower = upper + half_height

                        # 调整右下边界以处理奇数像素尺寸（确保裁剪区域不超出图像）
                        if col_idx == 0: # 如果是左半部分
                            right = half_width
                        else: # 如果是右半部分
                            right = img_width # 直接到图像最右边

                        if row_idx == 0: # 如果是上半部分
                            lower = half_height
                        else: # 如果是下半部分
                            lower = img_height # 直接到图像最下边

                        # 3. 执行裁剪
                        cropped_img = img.crop((left, upper, right, lower))

                        output_dir = "cropped_quadrants"
                        # 4. 保存图像
                        os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

                        output_filename = img_name+".png"
                        output_path = os.path.join(output_dir, output_filename)
                        if os.path.exists(output_path):
                            return output_path
                            # continue

                        cropped_img.save(output_path)
                        print(f"裁剪后的图像已保存到: {output_path}")
                        return output_path
        except:
            return img_path

    else:
        try:
            for region in ['Beijing','London','NewYork']:
                for region_part in ['urbanllava','citybench']:
                    if os.path.exists('../ImageData/'+region+'_'+region_part+'_zl17_merge/'+img_name+'.png'):
                        new_image_path = '../ImageData/'+region+'_'+region_part+'_zl17_merge/'+img_name+'.png'
                        return new_image_path
            
        except:
            return img_path

def extract_building_footprint(image_path, output_path=None):
    # 1. 读取图像并转为 RGB
    # # Building 的颜色（映射后）
    
    BUILDING_COLOR = (200, 80, 80)
    image = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. 创建二值掩码（mask）：只保留 building 区域
    mask = np.all(rgb_img == BUILDING_COLOR, axis=-1).astype(np.uint8) * 255

    # 3. 提取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 创建黑底图用于绘制轮廓
    footprint_mask = np.zeros_like(mask)

    # 5. 在黑底上画出轮廓（白色，边框粗一些）
    cv2.drawContours(footprint_mask, contours, -1, color=255, thickness=1)

    # 6. 转为 3 通道图，用于加文字
    output_img = cv2.cvtColor(footprint_mask, cv2.COLOR_GRAY2BGR)

    # 7. 添加文字标签
    cv2.putText(output_img, "building footprint", (20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=( 0,255,0), thickness=1, lineType=cv2.LINE_AA)

    # 8. 保存图像
    cv2.imwrite(output_path, output_img)
    print(f"Saved footprint mask with label to: {output_path}")


def calculate_pixel_ratios(image_path):
    # 定义：映射后颜色 → 类别
    mapped_colors = {
        (200, 80, 80): "building",
        (150, 150, 150): "road",
        (80, 170, 250): "water",
        (160, 130, 90): "barren",
        (60, 140, 70): "forest",
        (255, 210, 90): "agriculture"
    }
    # 读取图像并转换为 RGB（cv2 默认是 BGR）
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 总像素数
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

    # 计数器
    count_dict = defaultdict(int)

    # 拉平成 N x 3 并统计每个映射颜色
    flat_img = img_rgb.reshape(-1, 3)
    for pixel in flat_img:
        pixel_tuple = tuple(pixel)
        if pixel_tuple in mapped_colors:
            label = mapped_colors[pixel_tuple]
            count_dict[label] += 1

    # 输出统计结果
    print("Pixel Occupancy Ratios (based on mapped colors):")
    ans_str = "Pixel Occupancy Ratios (based on mapped colors):"
    for label in mapped_colors.values():
        count = count_dict.get(label, 0)
        ratio = count / total_pixels * 100
        print(f"{label:<12} pixels: {ratio:.2f}%")
        ans_str = ans_str + '\n' + f"{label:<12} pixels: {ratio:.2f}%"
    return str(ans_str)


def generate_single_prompt_for_all_buildings(centroids):
    """
    生成一个包含所有建筑物中心点和面积的 prompt，让大模型一次性估计所有建筑的高度
    """
    prompt_lines = [
        "You are given an aerial image where several buildings have been detected.",
        "Each building is described by its center pixel coordinate and its approximate footprint area in pixels.",
        "Please estimate the height (in meters) of each building based on this information.",
        "Return the result as a JSON list with entries containing 'id', 'center', and 'estimated_height_m'.\n",
        "Buildings:"
    ]

    for b in centroids:
        cx, cy = b["center"]
        area = b["area"]
        prompt_lines.append(f"  - ID: {b['id']}, Center: ({cx}, {cy}), Area: {area:.2f} pixels")

    prompt_lines.append("\nExample return format:")
    prompt_lines.append("""[
  {"id": 1, "center": [112, 83], "estimated_height_m": 15},
  {"id": 2, "center": [210, 144], "estimated_height_m": 12},
  ...
]""")

    return "\n".join(prompt_lines),0,0,0


BUILDING_COLOR = (200, 80, 80)

def extract_building_centroids(image_path):
    """
    提取所有 building 的轮廓中心点坐标
    """
    image = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.all(rgb_img == BUILDING_COLOR, axis=-1).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []

    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append({
                "id": i + 1,
                "center": (cx, cy),
                "area": cv2.contourArea(contour)
            })

    return centroids


def Satellite_Image_Semantic_Segmentation_Tool(image_path,prompt):
    requestAPI_loveda(image_path)
    out_path = return_sat_process_image_semsag(image_path)
    # print('out_path: ', out_path)
    return out_path, 'None',0,0,0

def Satellite_Image_Object_Detection_Tool(image_path,prompt):
    out_txt = Object_Detection_Sat(image_path,False,False)
    return 'None', out_txt,0,0,0

def Area_Estimator(image_path,prompt):
    # img_name = image_path.split('/')[-1].split('.')[0] #'43578_65513' #'10880_16359'#_pred
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    if not os.path.exists("./tmp_results/"+img_name+"_semseg_pure.png"):
        _,_,_,_,_ = Satellite_Image_Semantic_Segmentation_Tool(image_path,'')
    output_path="./tmp_results/"+img_name+"_semseg_pure.png"
    out_txt = calculate_pixel_ratios(output_path)
    return 'None',out_txt,0,0,0

# def Building_Footprint_Extractor(image_path,prompt):
#     # img_name = image_path.split('/')[-1].split('.')[0] #'43578_65513' #'10880_16359'#_pred
#     img_name = os.path.splitext(os.path.basename(image_path))[0]
#     if not os.path.exists("./tmp_results/"+img_name+"_semseg_pure.png"):
#         _,_,_,_,_ = Satellite_Image_Semantic_Segmentation_Tool(image_path,'')
#     output_path="./tmp_results/"+img_name+"_semseg_pure.png"
#     output_mask_path="./tmp_results/"+img_name+"_building_footprint.png"
#     extract_building_footprint(output_path,output_mask_path)
#     return output_mask_path,'None',0,0,0

def Building_Footprint_Extractor(image_path,prompt):
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,prompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Building_Height_Extractor(image_path,prompt):
    # img_name = image_path.split('/')[-1].split('.')[0] #'43578_65513' #'10880_16359'#_pred
    # img_name = os.path.splitext(os.path.basename(image_path))[0]
    # image_path_tmp ="./tmp_results/"+img_name+"_building_footprint.png"
    # centroids = extract_building_centroids(image_path_tmp)
    # prompt = generate_single_prompt_for_all_buildings(centroids)
    out_txt = VLM(image_path,prompt)
    return 'None',out_txt,0,0,0

def Road_Network_Extractor(image_path,prompt):
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,prompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens
# def Road_Network_Extractor(image_path,prompt):
#     img_name = os.path.basename(image_path).split('.')[0]
#     cnt_underline = img_name.count('_')
#     print(img_name)

#     if cnt_underline>2:
#         for region in ['Beijing','NewYork','London']:
#             for region_part in ['4','3','2','']:
#                 try_path = f'../test_run/{region}{region_part}_2020_out_imgs_1300/'+img_name.split('_')[0]+'_'+img_name.split('_')[1]+'_pred.png'
#                 if os.path.exists(try_path):
#                     img = Image.open(try_path)
                    
#                     x_idx = int(img_name.split('_')[2])
#                     y_idx = int(img_name.split('_')[3])

#                     img_width, img_height = img.size

#                     row_idx = x_idx
#                     col_idx = y_idx

#                     # 2. 计算裁剪区域 (left, upper, right, lower)
#                     half_width = img_width // 2
#                     half_height = img_height // 2

#                     left = col_idx * half_width
#                     upper = row_idx * half_height
#                     right = left + half_width
#                     lower = upper + half_height

#                     # 调整右下边界以处理奇数像素尺寸（确保裁剪区域不超出图像）
#                     if col_idx == 0: # 如果是左半部分
#                         right = half_width
#                     else: # 如果是右半部分
#                         right = img_width # 直接到图像最右边

#                     if row_idx == 0: # 如果是上半部分
#                         lower = half_height
#                     else: # 如果是下半部分
#                         lower = img_height # 直接到图像最下边

#                     # 3. 执行裁剪
#                     extra_img = img.crop((left, upper, right, lower))

#                     if extra_img.mode == 'RGB' or extra_img.mode == 'RGBA':
#                         img_array = rgb2gray(np.array(extra_img))
#                     else:
#                         img_array = np.array(extra_img)

#                     binary_img = img_as_bool(img_array > 0)
#                     # 3. 骨骼化到1像素宽度
#                     skeleton = skeletonize(binary_img)
#                     selem_radius = 1 # 半径为1的圆形结构元素，会使单像素线膨胀到约3像素宽
#                     selem = disk(selem_radius)

#                     # 执行膨胀操作
#                     from scipy.ndimage import binary_dilation
#                     skeleton_3px = binary_dilation(skeleton, structure=selem)

#                     skeleton_3px_pil = Image.fromarray((skeleton_3px * 255).astype(np.uint8))
#                     draw = ImageDraw.Draw(skeleton_3px_pil)
                    
#                     text = "road network"
#                     text_color = 255 # White text on a black background
#                     text_x = 10
#                     text_y = 10
#                     draw.text((text_x, text_y), text, fill=text_color)

#                     # Save the modified image
#                     output_filename = "./tmp_results/"+img_name+"_road_network_3px_width.png"
#                     skeleton_3px_pil.save(output_filename)
#                     return output_filename, 'None',0,0,0
#     else:
#         for region in ['Beijing','NewYork','London']:
#             for region_part in ['4','3','2','']:
#                 try_path = f'../test_run/{region}{region_part}_2020_out_imgs_1300/'+img_name+'_pred.png'
#                 if os.path.exists(try_path):
#                     img = Image.open(try_path)
#                     extra_img = img

#                     if extra_img.mode == 'RGB' or extra_img.mode == 'RGBA':
#                         img_array = rgb2gray(np.array(extra_img))
#                     else:
#                         img_array = np.array(extra_img)

#                     binary_img = img_as_bool(img_array > 0)
#                     # 3. 骨骼化到1像素宽度
#                     skeleton = skeletonize(binary_img)
#                     selem_radius = 1 # 半径为1的圆形结构元素，会使单像素线膨胀到约3像素宽
#                     selem = disk(selem_radius)

#                     # 执行膨胀操作
#                     from scipy.ndimage import binary_dilation
#                     skeleton_3px = binary_dilation(skeleton, structure=selem)

#                     skeleton_3px_pil = Image.fromarray((skeleton_3px * 255).astype(np.uint8))
#                     draw = ImageDraw.Draw(skeleton_3px_pil)
                    
#                     text = "road network"
#                     text_color = 255 # White text on a black background
#                     text_x = 10
#                     text_y = 10
#                     draw.text((text_x, text_y), text, fill=text_color)

#                     # Save the modified image
#                     output_filename = "./tmp_results/"+img_name+"_road_network_3px_width.png"
#                     skeleton_3px_pil.save(output_filename)
#                     return output_filename, 'None',0,0,0



def Satellite_Image_Automatic_Description_Generator(image_path,textprompt):
    # textprompt = '''Please provide semantic understanding and summarization of image content in detail.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Land_Use_Inference_Tool(image_path,textprompt):
    # textprompt = '''Please determines whether an area is industrial/residential/agricultural, etc., based on a satellite image (or its description).'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Structure_Layout_Analyzer(image_path,textprompt):
    # textprompt = '''Please extracts the spatial arrangement relationships of buildings, roads, green spaces.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Image_Style_Classifier(image_path,textprompt):
    # textprompt = '''Classifies the style of a satellite image, such as urban center / suburban / wasteland style.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Building_Density_Estimator(image_path,textprompt):
    # textprompt = '''Determines whether buildings in a satellite image are dense or uniformly distributed, and outputs text or numerical density.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Factory_Recognizer(image_path,textprompt):
    # textprompt = '''Determines whether factory structures (large flat-roofed buildings) exist in a satellite image, and outputs yes/no/location.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Large_Parking_Area_Detector(image_path,textprompt):
    # textprompt = '''Determines whether large parking areas exist in a satellite image, and outputs yes/no/location.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Special_Target_Recognizer(image_path,textprompt):
    # textprompt = '''Locates the position and type of specific key targets (e.g., oil tanks, water towers) in a satellite image.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

################################################

def remove_score_from_text(text: str) -> str:
    # 修正后的正则表达式：
    # 匹配 "score:" 后跟任意空格
    # 匹配数字和点（分数）
    # 匹配一个逗号（如果存在）
    # 匹配任意数量的空格
    # 匹配逗号 ',' 而不是分号 ';'
    # 注意这里使用 \b 来确保是单词边界，防止误匹配
    pattern = r"score:\s*[0-9.]+\s*," 
    
    # 使用 re.sub 替换所有匹配项为空字符串。
    cleaned_text = re.sub(pattern, "", text)
    
    # 进一步清理可能留下的双逗号或前导/尾随逗号。
    # 这里需要更精细的调整，因为移除了 score, 可能会留下 "object: sky, , bbox:"
    cleaned_text = re.sub(r',\s*,', ',', cleaned_text) # 替换 ", ," 为 ","
    
    # 还需要处理 "object: value, bbox:" 变成 "object: value bbox:" 的情况
    # 或者 "object: value ,"
    cleaned_text = re.sub(r'object:\s*(.*?),\s*bbox:', r'object: \1, bbox:', cleaned_text)
    
    # 最后处理可能遗留的 `,` 后面的空格问题
    cleaned_text = cleaned_text.replace('; ,', ';') # 修复 '; ,'
    cleaned_text = cleaned_text.replace(',;', ';') # 修复 ',;'
    
    # 清理掉可能在 object: 后留下多余的逗号，比如 object: sky,, bbox
    cleaned_text = re.sub(r',\s*bbox:', r', bbox:', cleaned_text)

    # 最终去除首尾空白
    return cleaned_text.strip()

def Street_Object_Detector(image_path,textprompt):
    # pass
    is_cropped,is_sr = True, True
    # image_name = image_path.split('/')[-1]
    image_name = Path(image_path).name
    print(image_name)
    object_info_gdino = extract_data_stv_obj(image_name, 'detect_info_image_stv_', data_stv_info)
    object_info_gdino = format_numbers_in_string_all(object_info_gdino)
    if len(str(object_info_gdino))>10:
        object_info_gdino = format_numbers_in_string_all(object_info_gdino)
        object_info_gdino = remove_score_from_text(object_info_gdino)
        return 'None', "The bounding box format for street view image is [x_center, y_center, x_width, y_height]', where x_center marks the normalized center pixel coordinates from the left edge of the image, and y_min marks the normalized center pixel coordinates from the top edge of the image. x_width and y_height mark the normalized width and height in x and y directions. "+ str(object_info_gdino),0,0,0

    else:
        try:
            # 从 JSON 文件中读取字典
            with open("my_data.json", 'r', encoding='utf-8') as f:
                loaded_dict = json.load(f)
            # 通过 key 查找 value
            search_key = image_path
            if search_key in loaded_dict:
                value = loaded_dict[search_key]
                image_path = value

            image_name = image_path.split('/')[-1]
            object_info_gdino = extract_data_stv_obj(image_name, 'detect_info_image_stv_', data_stv_info)
            object_info_gdino = format_numbers_in_string_all(object_info_gdino)
            object_info_gdino = remove_score_from_text(object_info_gdino)
            return 'None', "The bounding box format for street view image is [x_center, y_center, x_width, y_height]', where x_center marks the normalized center pixel coordinates from the left edge of the image, and y_min marks the normalized center pixel coordinates from the top edge of the image. x_width and y_height mark the normalized width and height in x and y directions. "+ str(object_info_gdino),0,0,0
        except:
            object_info_gdino = next((item["detect_info"] for item in data_stv_info if item["image"] == image_name), None)
            return 'None', "The bounding box format for street view image is [x_center, y_center, x_width, y_height]', where x_center marks the normalized center pixel coordinates from the left edge of the image, and y_min marks the normalized center pixel coordinates from the top edge of the image. x_width and y_height mark the normalized width and height in x and y directions. "+ str(object_info_gdino),0,0,0



def Street_View_Semantic_Segmentation_Tool(image_path,textprompt):
    print(image_path)

    image_name = Path(image_path).name
    print(image_name)
    # image_path.split('/')[-1]
    if os.path.exists('../ImageData/stv_semseg_result/'+image_name):
        image_path_new = '../ImageData/stv_semseg_result/'+image_name
    else:
        image_path_new = 'D:/Citylens_image_part/stv_semseg_result_Citylens/'+image_name
    if os.path.exists(image_path_new):
        pixel_percentage_dict = compute_pixel_percentage_dict(
            image_path=image_path_new
        )
        return 'None',str(pixel_percentage_dict),0,0,0 
    else:
        return 'None','Segmentation results not available.',0,0,0
    #     output_labeled_image_path = "./tmp_results/"+image_name[:-3]+'_labeled.jpg'
    #     add_single_label_per_class(image_path_new,output_labeled_image_path)

    #     return output_labeled_image_path,'None',0,0,0
    # else:
    #     # 从 JSON 文件中读取字典
    #     try:
    #         with open("my_data.json", 'r', encoding='utf-8') as f:
    #             loaded_dict = json.load(f)
    #         # 通过 key 查找 value
    #         search_key = image_path
    #         if search_key in loaded_dict:
    #             value = loaded_dict[search_key]
    #             image_path = value
    #     except:
    #         pass
    #     image_name = image_path.split('/')[-1]
    #     image_path_new = '../ImageData/stv_semseg_result/'+image_name
    #     if os.path.exists(image_path_new):
    #         output_labeled_image_path = "./tmp_results/"+image_name[:-3]+'_labeled.jpg'
    #         add_single_label_per_class(image_path_new,output_labeled_image_path)
    #         return output_labeled_image_path,'None',0,0,0
    #     else:
    #         return 'None','None',0,0,0

def Building_Facade_Extractor(image_path,textprompt):
    # textprompt = '''Describe the building facade in the given image.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Satellite_Image_Geo_Region_Localizer(image_path,textprompt):
    # textprompt = '''Describe the building facade in the given image.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Satellite_Image_Waterfront_Proximity_Analyzer(image_path,textprompt):
    # textprompt = '''Describe the building facade in the given image.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Urban_Block_Morphology_Classifier(image_path,textprompt):
    # textprompt = '''Describe the building facade in the given image.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Text_Sign_OCR(image_path,textprompt):
    # textprompt = '''Recognizes text on storefronts, road signs, and advertisements in street view images, outputting text content and location.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Building_Type_Cue_Detector(image_path,textprompt):
    # textprompt = '''Detects and analyze functional features such as windows, chimneys, factory doors, and garages in street view images, outputting a set of labels.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Vehicle_Type_Classifier(image_path,textprompt):
    # textprompt = '''Identifies and analyze vehicle types such as trucks, cars, buses, and motorcycles in street view images, outputting labels.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Pedestrian_Density_Estimator(image_path,textprompt):
    # textprompt = '''Estimates the number and density of pedestrians in a street view image, and outputs a numerical value or text description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Street_View_Image_Captioner(image_path,textprompt):
    # textprompt = '''Provides semantic understanding and summarization of street view content.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Building_Function_Inference_Tool(image_path,textprompt):
    # textprompt = '''Determines whether a building in a street view image is likely a factory, shop, residence, etc., and outputs a label or description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Street_Scene_Category_Inference(image_path,textprompt):
    # textprompt = '''Determines the type of area depicted in a street view image, such as industrial/commercial/residential/urban, and outputs a label.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Infrastructure_Detector(image_path,textprompt):
    # textprompt = '''Detects structures such as streetlights, cable towers, surveillance cameras, and fences in street view images, outputting labels and locations.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Object_Count_Reporter(image_path,textprompt): ########## 和object 联动
    # textprompt = ''' Reports the quantity of different types of objects in a street view image, such as 3 trucks, 5 people.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Vegetation_Detector(image_path,textprompt): ########## 和object 联动
    # textprompt = ''' Reports the quantity of different types of objects in a street view image, such as 3 trucks, 5 people.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Height_Approximation_Tool(image_path,textprompt):
    # textprompt = '''Roughly estimates the number of floors of a building in a street view image (based on features like windows), and outputs a numerical value or description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Satellite_Image_Railway_Detector(image_path,textprompt):
    # textprompt = '''Roughly estimates the number of floors of a building in a street view image (based on features like windows), and outputs a numerical value or description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Street_View_Architectural_Feature_Extractor(image_path,textprompt):
    # textprompt = '''Roughly estimates the number of floors of a building in a street view image (based on features like windows), and outputs a numerical value or description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Street_View_Ground_Level_Detail_Recognizer(image_path,textprompt):
    # textprompt = '''Roughly estimates the number of floors of a building in a street view image (based on features like windows), and outputs a numerical value or description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Street_View_Architectural_Style_Classifier(image_path,textprompt):
    # textprompt = '''Detects and analyze functional features such as windows, chimneys, factory doors, and garages in street view images, outputting a set of labels.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Street_View_Architectural_Feature_Extractor(image_path,textprompt):
    # textprompt = '''Detects and analyze functional features such as windows, chimneys, factory doors, and garages in street view images, outputting a set of labels.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Satellite_Image_Roof_and_Building_Footprint_Detailer(image_path,textprompt):
    # textprompt = '''Roughly estimates the number of floors of a building in a street view image (based on features like windows), and outputs a numerical value or description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Satellite_Image_Geospatial_Named_Entity_Extractor(image_path,textprompt):
    # textprompt = '''Roughly estimates the number of floors of a building in a street view image (based on features like windows), and outputs a numerical value or description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Satellite_Image_Landmark_Extraction_Tool(image_path,textprompt):
    # textprompt = '''Roughly estimates the number of floors of a building in a street view image (based on features like windows), and outputs a numerical value or description.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Commercial_Clue_Extractor(image_path,textprompt):
    # textprompt = '''Extracts commercial activity clues such as signs, shop windows, and advertisements from street view images, and outputs yes/no/details.'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def Enclosure_Detector(image_path,textprompt):
    # textprompt = '''Determines whether a street view image is in an enclosed area (fence, factory premises), and outputs "Enclosed" or "Open".'''
    out_txt,prompt_tokens,completion_tokens,total_tokens = VLM(image_path,textprompt)
    return 'None',out_txt,prompt_tokens,completion_tokens,total_tokens

def convert_bbox_format(input_string: str) -> str:
    """
    将包含多边形坐标的字符串转换为最小外接矩形格式，并规范对象名称。

    Args:
        input_string (str): 原始的字符串，包含 object 和 bbox (多边形) 信息。
                            示例: "object: ground track field, bbox: [[190, 874], ...]"

    Returns:
        str: 转换后的字符串，每个 object 后跟 bbox (min_x, min_y, max_x, max_y) 格式。
             示例: "object: Ground track field, bbox: [182, 686, 282, 879]"
    """
    results = []
    
    # 正则表达式解释：
    # (object:\s*[^,]+,\s*bbox:\s*)  -> 捕获对象名称前缀，例如 "object: ground track field, bbox: "
    # (\[\[.*?\]\])                  -> 捕获 bbox 内部的多边形坐标，例如 "[[190, 874], ...]"
    # ;?                             -> 可选的分号，处理最后一个条目
    object_pattern = re.compile(r'(object:\s*[^,]+,\s*bbox:\s*)(\[\[.*?\]\]);?\s*')

    # 遍历所有匹配的 object 条目
    for match in object_pattern.finditer(input_string):
        prefix_str = match.group(1) # 例如 "object: ground track field, bbox: "
        poly_coords_str = match.group(2) # 例如 "[[190, 874], [282, 868], ...]"

        # 1. 提取并格式化对象名称 (首字母大写)
        # 匹配 "object: " 之后直到第一个逗号之前的内容
        name_match = re.search(r'object:\s*([^,]+)', prefix_str)
        object_name = name_match.group(1).strip() if name_match else "Unknown"
        
        # 将名称的每个单词首字母大写
        formatted_object_name = ' '.join([word.capitalize() for word in object_name.split()])

        # 2. 从多边形坐标字符串中提取所有数字
        numbers = re.findall(r'\d+', poly_coords_str)
        
        # 将字符串数字转换为整数
        coords = [int(n) for n in numbers]

        # 3. 计算最小外接矩形 (min_x, min_y, max_x, max_y)
        # 坐标是 [x, y, x, y, ...] 这样的扁平列表
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for i in range(0, len(coords), 2): # 每两个数字是 (x, y)
            x = coords[i]
            y = coords[i+1]
            
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        
        # 4. 组合成目标格式
        # 注意：这里的 min_x, min_y, max_x, max_y 是基于多边形顶点计算的
        # 假设你示例中的 [182, 686, 282, 879] 是实际计算结果，而不是随意给出的
        # 这里的计算是 min/max x/y
        
        # 如果需要转换为你示例中那样一个大的整数（可能是四舍五入或特定算法），
        # 这里需要根据实际需求调整计算逻辑。
        # 按照“最小外接矩形”的通用定义，通常就是 (min_x, min_y, max_x, max_y)
        
        # 将结果添加到列表
        results.append(
            f"object: {formatted_object_name}, bbox: [{min_x}, {min_y}, {max_x}, {max_y}]"
        )
            
    # 使用 "; " 连接所有结果
    return "; ".join(results)

def Object_Detection_Sat(image_path,is_cropped=False,is_sr=False,param=None):
    image_name = image_path.split('/')[-1].split('.')[0]
    
    # if not is_sr:
    #     for region in ['Beijing','London','NewYork']:
    #         if os.path.exists('../ImageData/'+region+'_urbanllava/'+image_name+'.png'):
    #             image_path = '../ImageData/'+region+'_urbanllava/'+image_name+'.png'
    #             break
    #         elif os.path.exists('../ImageData/'+region+'_citybench/'+image_name+'.png'):
    #             image_path = '../ImageData/'+region+'_citybench/'+image_name+'.png'
    #             break
            ### Placeholder for OBD
    

    # if is_sr:# and not is_cropped:
    #     # for region in ['Beijing','London','NewYork']:
    #     #     if os.path.exists('../ImageData/'+region+'_urbanllava/'+image_name+'_zl17_merge.png'):
    #     #         image_path = '../ImageData/'+region+'_urbanllava/'+image_name+'_zl17_merge.png'
    #     #         break
    #     #     else:
    #     #         image_path = '../ImageData/'+region+'_citybench/'+image_name+'_zl17_merge.png'
    #     #         break
    #     image_path = Super_Resolution_Sat(image_path)
        ### Placeholder for OBD
    object_info_DIOR = requestAPI_DIOR(image_path)
    object_info_DOTA = requestAPI_DOTA(image_path)
    object_info_xview = requestAPI_xview(image_path)
    # print('formatted_string_DIOR: ', object_info_DIOR)
    # print('formatted_string_DOTA: ', object_info_DOTA)
    # print('formatted_string_xview: ', object_info_xview)

    formatted_string_DIOR = filter_objects_by_score(str(object_info_DIOR),0.3)
    # print('formatted_string_DIOR: ',formatted_string_DIOR)
    # print('formatted_string_DOTA: ',formatted_string_DOTA)
    # print('formatted_string_xview: ',formatted_string_xview)

    # if param==None:
    formatted_string_DIOR = format_numbers_in_string_sat_obj_all(str(formatted_string_DIOR))
    # else:
    #     formatted_string_DIOR = format_numbers_in_string_sat_obj(str(formatted_string_DIOR),param)
    # print(formatted_string)

    formatted_string_DOTA = filter_objects_by_score(str(object_info_DOTA),0.3)
    # if param==None:
    formatted_string_DOTA = format_numbers_in_string_sat_obj_all(str(formatted_string_DOTA))
    formatted_string_DOTA = convert_bbox_format(formatted_string_DOTA)
    # else:
        # formatted_string_DOTA = format_numbers_in_string_sat_obj(str(formatted_string_DOTA),param)

    formatted_string_xview = filter_objects_by_score(str(object_info_xview),0.3)
    # if param==None:
    formatted_string_xview = format_numbers_in_string_sat_obj_all(str(formatted_string_xview))
    # else:
        # formatted_string_xview = format_numbers_in_string_sat_obj(str(formatted_string_xview),param)
    
    k = formatted_string_DIOR + ' ; ' + formatted_string_DOTA + ' ; ' + formatted_string_xview
    # print('formatted_string_xview:  ',formatted_string_xview)
    # print('formatted_string_DIOR: ',formatted_string_DIOR)
    # print('\n')
    # print('formatted_string_DOTA: ',formatted_string_DOTA)
    # print('\n')
    # print('formatted_string_xview: ',formatted_string_xview)
    # print('\n')
    # return_sat_process_image_obd(image_path, k,is_cropped)
    return "The bounding box format for satellite image is [x_min, y_min, x_max, y_max], where x_min and x_max mark the pixel coordinates from the left edge of the image, and y_min and y_max mark the pixel coordinates from the top edge of the image. The object 'building' represents all kinds of buildings and the object 'vehicle' represents all kinds of automobiles. "+ k