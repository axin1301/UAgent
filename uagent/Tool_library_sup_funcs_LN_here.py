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
import sys
# sys.path.append('../Framework_v17/')
import os
# from MLLM_verifier import *
import cv2

def Crop_Sat(image_path):
    """
    读取 PNG 图像，分成四份，并保存为新的 PNG 文件。

    参数：
    image_path (str): 原始 PNG 图像的路径。
    output_folder (str): 保存分割后图像的文件夹路径。
    """
    output_folder = 'split_images'
    os.makedirs(output_folder,exist_ok=True)
    image_split_list = []
    try:
        # 打开图像
        img = Image.open(image_path)
        width, height = img.size

        # 计算分割后的图像尺寸
        new_width = width // 2
        new_height = height // 2

        # 分割图像并保存
        for j in range(2):
            for i in range(2):

                left = i * new_width
                upper = j * new_height
                right = left + new_width
                lower = upper + new_height

                # 生成输出文件名
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_{i}_{j}{ext}" 
                output_path = os.path.join(output_folder, output_filename)
                image_split_list.append(output_path)
                # if os.path.exists(output_path):
                #     continue
                # 分割图像
                cropped_img = img.crop((left, upper, right, lower))
                output_path = os.path.join(output_folder, output_filename)

                # 保存分割后的图像
                cropped_img.save(output_path)
                print(f"分割后的图像保存为: {output_path}")
        print(image_split_list)
        return image_split_list

    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")


def extract_crop_status_from_combined_string(combined_string):
    """
    从包含StreetViewImage和SatelliteImage的Crop状态的单个字符串中提取值。

    Args:
        combined_string (str): 包含 <StreetViewImage><Crop>...</Crop>...<SatelliteImage><Crop>...</Crop> 的字符串。

    Returns:
        dict: 包含 'StreetViewImage_Crop' 和 'SatelliteImage_Crop' 布尔值的字典。
              如果某个值未找到，则对应的值为 None。
    """
    results = {}

    # 提取 StreetViewImage 的 Crop 状态
    # pattern: <StreetViewImage><Crop>(True|False)</Crop></StreetViewImage>
    # group(1) 捕获 True 或 False
    street_view_pattern = r"<StreetViewImage><Crop>(True|False)</Crop></StreetViewImage>"
    street_view_match = re.search(street_view_pattern, combined_string)
    if street_view_match:
        results['StreetViewImage_Crop'] = street_view_match.group(1) == "True"
    else:
        results['StreetViewImage_Crop'] = None

    # 提取 SatelliteImage 的 Crop 状态
    # pattern: <SatelliteImage><Crop>(True|False)</Crop></SatelliteImage>
    # group(1) 捕获 True 或 False
    satellite_pattern = r"<SatelliteImage><Crop>(True|False)</Crop></SatelliteImage>"
    satellite_match = re.search(satellite_pattern, combined_string)
    if satellite_match:
        results['SatelliteImage_Crop'] = satellite_match.group(1) == "True"
    else:
        results['SatelliteImage_Crop'] = None

    return results

def leave_last(combined):
    unique = {item["image"]: item for item in combined}  # 后面的会覆盖前面的
    result = list(unique.values())
    return result


# global data_loveda
# with open('sat_process_results/London_urbanllava_semsag.json', "r") as file:
#     data_loveda = json.load(file)
# with open('sat_process_results/London_urbanllava_zl17_merge_semseg.json', "r") as file:
#     data_loveda_tmp = json.load(file)
#     data_loveda = data_loveda + data_loveda_tmp
# with open('sat_process_results/NewYork_urbanllava_semseg.json', "r") as file:
#     data_loveda_tmp = json.load(file)
#     data_loveda = data_loveda + data_loveda_tmp
# with open('sat_process_results/NewYork_urbanllava_zl17_merge_semseg.json', "r") as file:
#     data_loveda_tmp = json.load(file)
#     data_loveda = data_loveda + data_loveda_tmp
# with open('sat_process_results/London_semsag.json', "r") as file:
#     data_loveda_tmp = json.load(file)
#     data_loveda = data_loveda + data_loveda_tmp
# with open('sat_process_results/London_citybench_zl17_merge_semseg.json', "r") as file:
#     data_loveda_tmp = json.load(file)
#     data_loveda = data_loveda + data_loveda_tmp
# data_loveda = leave_last(data_loveda)

global data_DOTAA
global data_DIOR
global data_xview

# with open('sat_process_results/NewYork_urbanllava_DIOR.json', "r") as file:
#     data_DIOR = json.load(file)
# with open('sat_process_results/NewYork_urbanllava_DIOR_zl17_merge.json', "r") as file:
#     data_DIOR_tmp = json.load(file)
#     data_DIOR = data_DIOR + data_DIOR_tmp
# with open('sat_process_results/London_urbanllava_DIOR.json', "r") as file:
#     data_DIOR_tmp = json.load(file)
#     data_DIOR = data_DIOR + data_DIOR_tmp
# with open('sat_process_results/London_urbanllava_DIOR_zl17_merge.json', "r") as file:
#     data_DIOR_tmp = json.load(file)
#     data_DIOR = data_DIOR + data_DIOR_tmp
# with open('sat_process_results/London_DIOR.json', "r") as file:
#     data_DIOR_tmp = json.load(file)
#     data_DIOR = data_DIOR + data_DIOR_tmp
# with open('sat_process_results/London_citybench_DIOR_zl17_merge.json', "r") as file:
#     data_DIOR_tmp = json.load(file)
#     data_DIOR = data_DIOR + data_DIOR_tmp
# data_DIOR = leave_last(data_DIOR)
# # print(data_DIOR[200])

# with open('sat_process_results/London_urbanllava_DOTA.json', "r") as file:
#     data_DOTAA = json.load(file)
# with open('sat_process_results/London_urbanllava_DOTA_zl17_merge.json', "r") as file:
#     data_DOTA_tmp = json.load(file)
#     data_DOTAA = data_DOTAA + data_DOTA_tmp
# with open('sat_process_results/NewYork_urbanllava_DOTA.json', "r") as file:
#     data_DOTA_tmp = json.load(file)
#     data_DOTAA = data_DOTAA + data_DOTA_tmp
# with open('sat_process_results/NewYork_urbanllava_DOTA_zl17_merge.json', "r") as file:
#     data_DOTA_tmp = json.load(file)
#     data_DOTAA = data_DOTAA + data_DOTA_tmp
# with open('sat_process_results/London_DOTA.json', "r") as file:
#     data_DOTA_tmp = json.load(file)
#     data_DOTAA = data_DOTAA + data_DOTA_tmp
# with open('sat_process_results/London_citybench_DOTA_zl17_merge.json', "r") as file:
#     data_DOTA_tmp = json.load(file)
#     data_DOTAA = data_DOTAA + data_DOTA_tmp
# data_DOTAA = leave_last(data_DOTAA)

# with open('sat_process_results/London_urbanllava_xview.json', "r") as file:
#     data_xview = json.load(file)
# with open('sat_process_results/London_urbanllava_xview_zl17_merge.json', "r") as file:
#     data_xview_tmp = json.load(file)
#     data_xview = data_xview + data_xview_tmp
# with open('sat_process_results/NewYork_urbanllava_xview.json', "r") as file:
#     data_xview_tmp = json.load(file)
#     data_xview = data_xview + data_xview_tmp
# with open('sat_process_results/NewYork_urbanllava_xview_zl17_merge.json', "r") as file:
#     data_xview_tmp = json.load(file)
#     data_xview = data_xview + data_xview_tmp
# with open('sat_process_results/London_xview.json', "r") as file:
#     data_xview_tmp = json.load(file)
#     data_xview = data_xview + data_xview_tmp
# with open('sat_process_results/London_citybench_xview_zl17_merge.json', "r") as file:
#     data_xview_tmp = json.load(file)
#     data_xview = data_xview + data_xview_tmp
# data_xview = leave_last(data_xview)

global data_stv_info

with open("../stv_process_files/object_detection_BJ_stv_stv_compare.json", "r") as file:
    data_stv_info = json.load(file)
with open("../stv_process_files/object_detection_BJ_stv_img_retrieval.json", "r") as file:
    data_stv_info_tmp = json.load(file)
    data_stv_info = data_stv_info+ data_stv_info_tmp
with open("../stv_process_files/object_detection_BJ_stv_img_camera.json", "r") as file:
    data_stv_info_tmp = json.load(file)
    data_stv_info = data_stv_info+ data_stv_info_tmp
with open("../stv_process_files/object_detection_BJ_stv_landmark.json", "r") as file:
    data_stv_info_tmp = json.load(file)
    data_stv_info = data_stv_info+ data_stv_info_tmp
with open("../stv_process_files/object_detection_BJ_stv_stv_address.json", "r") as file:
    data_stv_info_tmp = json.load(file)
    data_stv_info = data_stv_info+ data_stv_info_tmp
with open("../stv_process_files/object_detection_BJ_stv_citybench_all_cities.json", "r") as file:
    data_stv_info_tmp = json.load(file)
    data_stv_info = data_stv_info+ data_stv_info_tmp
with open("../stv_process_files/object_detection_London_urbanllava_stv.json", "r") as file:
    data_stv_info_tmp = json.load(file)
    data_stv_info = data_stv_info+ data_stv_info_tmp
with open("../stv_process_files/object_detection_NewYork_urbanllava_stv.json", "r") as file:
    data_stv_info_tmp = json.load(file)
    data_stv_info = data_stv_info+ data_stv_info_tmp
with open("../../CityLens_data/object_detection_CityLens_stv.json", "r") as file:
    data_stv_info_tmp = json.load(file)
    data_stv_info = data_stv_info+ data_stv_info_tmp

# global data_loc_stv
# with open("../LM2Data/GroundingDino/Beijing_stv_address_mc_test_location.json", "r") as file:
#     data_loc_stv = json.load(file)
# with open("../LM2Data/GroundingDino/Beijing_stv_compare_test_location.json", "r") as file:
#     data_loc_stv_tmp = json.load(file)
#     data_loc_stv = data_loc_stv + data_loc_stv_tmp
# with open("../LM2Data/GroundingDino/Beijing_stv_landmark_mc_test_location.json", "r") as file:
#     data_loc_stv_tmp = json.load(file)
#     data_loc_stv = data_loc_stv + data_loc_stv_tmp
# with open("../LM2Data/GroundingDino/Beijing_STV_SAT_location_test_location.json", "r") as file:
#     data_loc_stv_tmp = json.load(file)
#     data_loc_stv = data_loc_stv + data_loc_stv_tmp
# with open("../LM2Data/GroundingDino/Beijing_STV_SAT_mapping_test_location.json", "r") as file:
#     data_loc_stv_tmp = json.load(file)
#     data_loc_stv = data_loc_stv + data_loc_stv_tmp


# def parse_detections(detection_str):
#     pattern = r"(\w+), bbox: \[\[(.*?)\]\], scores: \[(.*?)\]"  
#     detections = []
    
#     for match in re.finditer(pattern, detection_str):
#         obj_class = match.group(1)
#         bbox = np.array(list(map(float, match.group(2).split())))
#         score = float(match.group(3))
#         detections.append((obj_class, bbox, score))
    
#     return detections

def parse_detections(detection_str):
    pattern = r"(?:object:\s*)?([\w\s]+), bbox: \[\[(.*?)\]\], scores: \[(.*?)\]"  
    detections = []
    
    for match in re.finditer(pattern, detection_str):
        obj_class = match.group(1).strip()  # 去除前后空格
        bbox = np.array(list(map(float, match.group(2).split())))  # 解析 bbox
        score = float(match.group(3))  # 解析分数
        detections.append((obj_class, bbox, score))
    
    return detections

def parse_detections_cb(detection_string):
    """解析检测结果字符串，提取类别、bbox 和得分"""
    pattern = r"object: (.*?), bbox: \[\[(.*?)\]\], scores: \[(.*?)\]"
    detections = []
    
    for match in re.finditer(pattern, detection_string):
        class_name = match.group(1)
        bbox = list(map(float, match.group(2).split()))
        score = float(match.group(3))
        detections.append((class_name, bbox, score))
    
    return detections

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def nms(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x[2], reverse=True)
    keep = []
    
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [det for det in detections if det[0] != best[0] or compute_iou(best[1], det[1]) < iou_threshold]
    
    return keep

def format_detections(detections):
    formatted_strs = []
    for obj_class, bbox, score in detections:
        bbox_str = " ".join(f"{x:.2f}" for x in bbox)  # 保持小数点后6位
        formatted_strs.append(f"object: {obj_class}, bbox: [[{bbox_str}]], scores: [{score:.2f}]")
    return "; ".join(formatted_strs)


# --- 函数 1: 格式化数字 ---
def format_numbers(s: str) -> str:
    """
    格式化字符串中的所有数字。
    - 如果是浮点数，保留两位小数。
    - 如果是整数，保留整数格式。
    """
    def format_match(match: re.Match) -> str:
        num_str = match.group(0)
        num_float = float(num_str)
        
        return str(int(num_float))


    # 匹配所有整数或浮点数
    formatted_str = re.sub(r"\b\d+(?:\.\d+)?\b", format_match, s) 
    return formatted_str

# --- 函数 2: 移除 scores ---
def remove_scores_from_string(s: str) -> str:
    """
    从字符串中移除所有 'scores: [...]' 的内容，并清理多余的逗号和空格。
    """
    pattern = re.compile(r'scores:\s*\[.*?\];?')
    cleaned_string = pattern.sub('', s)
    
    # 清理可能留下的多余逗号和空格
    cleaned_string = re.sub(r',\s*,', ',', cleaned_string)
    cleaned_string = re.sub(r',\s*$', '', cleaned_string)
    cleaned_string = re.sub(r'^\s*,\s*', '', cleaned_string)

    return cleaned_string.strip()


def format_numbers_in_string_sat_obj_all(s):
    # 步骤 1: 格式化数字
    temp_formatted_str = format_numbers(s)
    
    # 步骤 2: 移除 scores
    final_processed_str = remove_scores_from_string(temp_formatted_str)
    
    return final_processed_str


def format_numbers_in_string_sat_obj(s, param):
    """
    1. 只保留 `param` 对应的 object
    2. 提取 bbox（归一化坐标）
    3. 格式化坐标并合并，输出格式：
       object_name [x1, y1, x2, y2], [x3, y3, x4, y4], ...
    """
    
    # 正则匹配对象名称、bbox
    object_pattern = re.compile(r"object:\s*([\w\s]+),\s*bbox:\s*\[\[([\d.\s]+)\]\],\s*scores:\s*\[([\d.]+)\]")
    
    formatted_objects = {}

    for match in object_pattern.finditer(s):
        obj_name = match.group(1).strip()
        bbox = [float(num) for num in match.group(2).split()]
        confidence = float(match.group(3))  # 置信度

        if obj_name.lower() in [param]:  # 只保留指定 object
            if obj_name not in formatted_objects:
                formatted_objects[obj_name] = []

            # 格式化 bbox 坐标（保留两位小数）
            formatted_objects[obj_name].append(
                [round(coord, 2) for coord in bbox]
            )

    # 生成输出字符串
    result = []
    for obj_name, bboxes in formatted_objects.items():
        bbox_str = ", ".join([f"[{', '.join(map(str, bbox))}]" for bbox in bboxes])
        result.append(f"object {obj_name} {bbox_str}")

    return "\n".join(result)

def convert_decimals_to_percentages(text):
    """
    将文本中包含的小数转化为百分比形式。

    Args:
        text (str): 包含 "pixel percentage: 小数" 格式的文本。

    Returns:
        str: 将小数转化为百分比后的文本。
    """
    lines = text.split(';')
    output_lines = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split(':')
            if len(parts) == 2:
                landuse = parts[0].strip()
                pixel_percentage_str = parts[1].strip()
                try:
                    pixel_percentage = float(pixel_percentage_str)
                    percentage = f"{pixel_percentage * 100:.2f}%"
                    output_lines.append(f"{landuse}, percentage: {percentage}")
                except ValueError:
                    output_lines.append(line)  # 如果无法转换为浮点数，则保持原样
            else:
                output_lines.append(line)  # 如果格式不符合预期，则保持原样
    return '; '.join(output_lines)

def remove_normalized_boundary_all(text):
    # 正则表达式匹配 "normalized boundary: [[[" 到 "]]]"
    pattern = r", normalized boundary: \[\[\[.*?\]\]\]"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)  # DOTALL 允许匹配换行符
    cleaned_text = cleaned_text.replace(', normalized boundary: []', '')
    return cleaned_text.strip()  # 去除前后可能的空格

def remove_normalized_boundary(text, param=[]):
    """
    处理 pixel percentage 和 normalized boundary 信息。
    
    - 如果 param 非空，只保留 param 列表中的类别及其 pixel percentage。
    - 如果 param 为空，保留所有 pixel percentage，删除所有 normalized boundary。
    
    :param text: 原始字符串（包含类别 pixel percentage 和 normalized boundary）。
    :param param: 需要保留的类别列表（小写匹配）；如果为空，则保留全部类别但删除 boundary。
    :return: 处理后的字符串。
    """
    # 正则匹配每一组：类别、pixel percentage 和 normalized boundary
    pattern = re.compile(r"(\w+)\s+pixel\s+percentage:\s+([\d.]+),\s+normalized\s+boundary:\s+\[.*?\](?=(?:;|$))", re.DOTALL)
    
    filtered_results = []

    for match in pattern.finditer(text):
        category = match.group(1).lower()
        percentage = float(match.group(2))

        if not param or category in param:
            filtered_results.append(f"{category} pixel percentage: {percentage:.4f}")

    return "; ".join(filtered_results)

def format_numbers_in_string_all(data_str):
    # 使用正则表达式匹配所有浮点数，并将其格式化为保留两位小数
    return re.sub(r"\d+\.\d+", lambda match: f"{float(match.group(0)):.2f}", data_str)

def format_numbers_in_string(data_str, param):
    """
    只保留指定 param 相关的 object，并格式化其中的数值保留两位小数。

    :param data_str: 原始输入字符串
    :param param: 需要保留的类别（列表）
    :return: 格式化后的字符串
    """
    pattern = re.compile(r"object:\s*(\w+),\s*score:\s*([\d.]+),\s*bbox:\s*\[([^\]]+)\]")
    
    filtered_objects = []
    
    for match in pattern.finditer(data_str):
        obj_name = match.group(1)  # 提取 object 名称
        if obj_name.lower() in [param]:  # 只保留 param 中指定的 object
            bbox_values = [f"{float(x):.2f}" for x in match.group(3).split(",")]  # 格式化 bbox
            filtered_objects.append(f"{obj_name} [{', '.join(bbox_values)}]")  # 重新格式化输出

    return f"object {', '.join(filtered_objects)}"

###############当前tool-library引用了citybench，urbanllava的数据新开一个tool-library
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (ytile, xtile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

# 整理后的物体列表
consolidated_object_list = [
    "Airplane",  # 合并了 Fixed-wing Aircraft, Small Aircraft, Cargo Plane
    "Airport",
    "Baseball field",  # 保留
    "Basketball court",  # 保留
    "Bridge",  # 保留
    "Chimney",  # 保留
    "Dam",  # 保留
    "Expressway Service area",  # 保留
    "Expressway toll station",  # 保留
    "Golf course",  # 保留
    "Ground track field",  # 保留
    "Harbor",  # 保留
    "Overpass",  # 保留
    "Ship",  # 合并了 Maritime Vessel, Fishing Vessel, Container Ship, Oil Tanker 等
    "Stadium",  # 保留
    "Storage tank",  # 合并大小写重复项
    "Tennis court",  # 保留
    "Train station",  # 保留
    "Vehicle",  # 合并 Large Vehicle, Small Vehicle, Passenger Vehicle, Truck 等
    "Windmill",  # 保留
    "Soccer ball field",  # 保留
    "Swimming pool",  # 保留
    "Helicopter",  # 保留
    "Tower crane",  # 保留
    "Building",  # 合并 Aircraft Hangar, Damaged Building, Facility
    "Construction Site",  # 保留
    "Shipping Container",  # 合并 Shipping container lot
    "Pylon",  # 保留
    "Tower",  # 保留
]

def format_counter_to_string(counter):
    """
    Format Counter object to a human-readable string with pluralization.

    Parameters:
    - counter (Counter): A Counter object with counts of items.

    Returns:
    - result (str): A formatted string.
    """
    formatted_items = []
    for obj, count in counter.items():
        # Handle pluralization
        if count > 1:
            if obj.endswith("s"):
                plural_obj = obj  # Already plural
            else:
                plural_obj = obj + "s"
        else:
            plural_obj = obj  # Singular
        
        # Add formatted string to the list
        formatted_items.append(f"{count} {plural_obj}")
    
    # Join the list into a single string
    return ", ".join(formatted_items)

def filter_objects_by_score(data: str, threshold: float) -> str:
    """
    从输入字符串中筛选出得分高于指定阈值的物体，并返回格式化的字符串。

    :param data: 输入的包含物体信息的字符串
    :param threshold: 需要筛选的分数阈值
    :return: 过滤后的字符串
    """
    # 正则匹配 object, bbox, score
    pattern = re.compile(r'object:\s*(.*?),\s*bbox:\s*(\[[\s\S]*?\]),\s*scores:\s*\[?([\d.]+)\]?')

    filtered_objects = []
    for match in pattern.findall(data):
        obj, bbox, score = match
        score = float(score)  # 转换为浮点数
        if score > threshold:
            filtered_objects.append(f"object: {obj}, bbox: {bbox}, scores: [{score}]")

    return "; ".join(filtered_objects)


def count_objects_above_threshold(input_string, object_list, threshold=0.2):
    """
    Count objects in the string with scores above the threshold for two formats.

    Parameters:
    - input_string (str): The input string containing object data.
    - object_list (list): List of valid object names.
    - threshold (float): The score threshold.

    Returns:
    - counts (dict): Dictionary with object names and counts.
    """
    # Regex to match both formats
    pattern = r"object:\s*([A-Za-z\s]+),\s*bbox:\s*\[.*?\],\s*scores:\s*\[?\s*([\d.]+)\s*\]?"
    
    # Find all matches for objects and scores
    matches = re.findall(pattern, input_string, re.DOTALL)
    
    # Filter matches by score threshold and valid object names
    filtered_objects = [
        obj.strip() for obj, score in matches
        if float(score) > threshold and obj.strip() in object_list
    ]
    
    # Count occurrences of each valid object
    object_counts = Counter(filtered_objects)
    
    # Print the results
    # for obj, count in object_counts.items():
    #     print(f"{count} {obj}")
    
    return object_counts

def count_landuse(text):
    # 正则表达式提取landuse和百分比信息
    # 正则表达式提取 landuse 和百分比信息
    pattern = r"(\w+ landuse),\s*pixel percentage:\s*([\d\.]+)"
    matches = re.findall(pattern, text)

    # 转换为目标格式
    formatted_results = []
    for landuse, percentage in matches:
        percentage_as_percent = float(percentage) * 100  # 转为百分比
        formatted_results.append(f"{landuse.split()[0].capitalize()} area {percentage_as_percent:.1f}%")

    # 输出结果
    # for result in formatted_results:
    #     print(result)
    return formatted_results


def extract_data_sat_obj(image_name, key_name, data_DOTA):
    #key_name = DOTA_info_sat_
    # image_name = 123_456
    # print(len(data_DOTA))
    object_info_DOTA = " "  # 默认值为空
    for item_DOTA in data_DOTA:
        # print(item_DOTA)
        if isinstance(item_DOTA["image"], list) and len(item_DOTA["image"])==5 : #"image" in item_DOTA and 
            # 查找 image_name 在 image 列表中的索引
            # print('good')
            try:
                item_DOTA["image"] = [x.split('/')[-1] for x in item_DOTA["image"]]
                index = item_DOTA["image"].index(f"{image_name}.png")
                # print('index: ',index)
                dota_key = key_name+str(index-1) #  # 构造 key 名称
                object_info_DOTA = item_DOTA.get(dota_key, " ")  # 获取对应值
                # break  # 找到后停止循环
                # print('here1: ',object_info_DOTA)
                return object_info_DOTA
            except ValueError:
                continue  # 当前 item_DOTA 的 image 列表中没有目标图片，跳过
        elif isinstance(item_DOTA["image"], list) and len(item_DOTA["image"])!=5 : #"image" in item_DOTA and 
            # 查找 image_name 在 image 列表中的索引
            # print('good')
            try:
                item_DOTA["image"] = [x.split('/')[-1] for x in item_DOTA["image"]]
                index = item_DOTA["image"].index(f"{image_name}.png")
                # print('index: ',index)
                dota_key = key_name+str(index) #  # 构造 key 名称
                object_info_DOTA = item_DOTA.get(dota_key, " ")  # 获取对应值
                # break  # 找到后停止循环
                # print('here2: ',object_info_DOTA)
                return object_info_DOTA
            except ValueError:
                continue  # 当前 item_DOTA 的 image 列表中没有目标图片，跳过
        elif "image" in item_DOTA and not isinstance(item_DOTA["image"], list):
            try:
                # print(item_DOTA["image"])
            # print(item_DOTA["image"].split('/')[-1])
                if item_DOTA["image"].split('/')[-1] == image_name+'.png':
                    object_info_DOTA = item_DOTA[key_name+str(0)]
                    # print('here3: ',object_info_DOTA)
                    return object_info_DOTA
            except:
                continue
    return None

def extract_data_sat_semseg(image_name, key_name, data_DOTA):
    #key_name = DOTA_info_sat_
    # image_name = 123_456
    object_info_DOTA = " "  # 默认值为空
    for item_DOTA in data_DOTA:
        if "image" in item_DOTA and isinstance(item_DOTA["image"], list) and len(item_DOTA["image"])<5:
            # 查找 image_name 在 image 列表中的索引
            try:
                item_DOTA["image"] = [x.split('/')[-1] for x in item_DOTA["image"]]
                index = item_DOTA["image"].index(f"{image_name}.png")
                # print('index: ',index)
                dota_key = key_name+str(index) #  # 构造 key 名称
                object_info_DOTA = item_DOTA.get(dota_key, " ")  # 获取对应值
                # break  # 找到后停止循环
                return object_info_DOTA
            except ValueError:
                continue  # 当前 item_DOTA 的 image 列表中没有目标图片，跳过
        elif "image" in item_DOTA and isinstance(item_DOTA["image"], list) and len(item_DOTA["image"])==5:
            # 查找 image_name 在 image 列表中的索引
            try:
                item_DOTA["image"] = [x.split('/')[-1] for x in item_DOTA["image"]]
                index = item_DOTA["image"].index(f"{image_name}.png")
                # print('index: ',index)
                dota_key = key_name+str(index-1) #  # 构造 key 名称
                object_info_DOTA = item_DOTA.get(dota_key, " ")  # 获取对应值
                # break  # 找到后停止循环
                return object_info_DOTA
            except ValueError:
                continue  # 当前 item_DOTA 的 image 列表中没有目标图片，跳过
        elif "image" in item_DOTA and not isinstance(item_DOTA["image"], list):
            if item_DOTA["image"].split('/')[-1] == image_name+'.png':
                object_info_DOTA = item_DOTA[key_name+str(0)]
                return object_info_DOTA
    return None


def extract_data_stv_obj(image_name, key_name, data_DOTA):
    #key_name = DOTA_info_sat_
    # image_name = 123_456
    # print(len(data_DOTA))
    print(image_name)
    object_info_DOTA = " "  # 默认值为空
    for item_DOTA in data_DOTA:
        # print(item_DOTA)
        if "image" in item_DOTA and isinstance(item_DOTA["image"], list):
            # 查找 image_name 在 image 列表中的索引
            try:
                item_DOTA["image"] = [x.split('/')[-1] for x in item_DOTA["image"]]
                index = item_DOTA["image"].index(f"{image_name}")
                # print('Here')
                # print('stv obj index:  ',index)
                dota_key = key_name+str(index) #  # 构造 key 名称
                if dota_key in item_DOTA:
                    object_info_DOTA = item_DOTA.get(dota_key, " ")  # 获取对应值
                    return object_info_DOTA
                elif 'detect_info_image_stv' in item_DOTA:
                    object_info_DOTA = item_DOTA['detect_info_image_stv']
                # break  # 找到后停止循环
                    return object_info_DOTA
            except:
                try:
                    item_DOTA["image"] = [x.split('/')[-1] for x in item_DOTA["image"]]
                    index = item_DOTA["image"].index(f"{image_name}")
                    dota_key = 'detect_info_image_stv' #  # 构造 key 名称
                    object_info_DOTA = item_DOTA.get(dota_key, " ")  # 获取对应值
                    # break  # 找到后停止循环
                    # return object_info_DOTA
                except:
                    continue  # 当前 item_DOTA 的 image 列表中没有目标图片，跳过
        elif "image" in item_DOTA and not isinstance(item_DOTA["image"], list):
            if item_DOTA["image"].split('/')[-1] == image_name:
                # try:
                if 'detect_info' in item_DOTA:
                    object_info_DOTA = item_DOTA['detect_info']
                elif 'detect_info_image_stv_0' in item_DOTA:
                    object_info_DOTA = item_DOTA['detect_info_image_stv_0']
                elif 'detect_info_image_stv' in item_DOTA:
                    object_info_DOTA = item_DOTA['detect_info_image_stv']
                return object_info_DOTA
    return object_info_DOTA if object_info_DOTA else 'NONE'
##detect_info_image_stv_0
##detect_info_image_stv
##detect_info

def apply_nms(detection_string, score_threshold=0.1, nms_threshold=0.7):
    """对检测框进行 NMS，并返回经过 NMS 处理后的字符串"""
    detections = parse_detections_cb(detection_string)

    boxes = []
    scores = []
    class_names = []

    for class_name, bbox, score in detections:
        if score >= score_threshold:
            boxes.append(bbox)
            scores.append(score)
            class_names.append(class_name)

    if not boxes:
        return "No valid detections after thresholding."

    boxes = np.array(boxes)
    scores = np.array(scores)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold, nms_threshold)
    
    if len(indices) == 0:
        return "No detections after NMS."

    final_detections = [f"object: {class_names[i]}, bbox: {boxes[i]}, scores: [{scores[i]:.3f}]" 
                         for i in indices.flatten()]

    return " ; ".join(final_detections)

def extract_data_stv_loc(image_name, key_name, data_DOTA):
    #key_name = DOTA_info_sat_
    # image_name = 123_456
    object_info_DOTA = " "  # 默认值为空
    for item_DOTA in data_DOTA:
        if "image" in item_DOTA and isinstance(item_DOTA["image"], list) and len(item_DOTA["image"])>3:
            # 查找 image_name 在 image 列表中的索引
            try:
                item_DOTA["image"] = [x.split('/')[-1] for x in item_DOTA["image"]]
                index = item_DOTA["image"].index(f"{image_name}")
                # print('stv geo loc index:  ',index) 
                dota_key = key_name+str(index) #  # 构造 key 名称
                object_info_DOTA = item_DOTA.get(dota_key, " ")  # 获取对应值
                # break  # 找到后停止循环
                # return object_info_DOTA
            except ValueError:
                continue
        elif "image" in item_DOTA and isinstance(item_DOTA["image"], list) and len(item_DOTA["image"])==2:
            # 查找 image_name 在 image 列表中的索引
            try:
                item_DOTA["image"] = [x.split('/')[-1] for x in item_DOTA["image"]]
                index = item_DOTA["image"].index(f"{image_name}")
                dota_key = "possible_geo_coordinates_stv_0" #  # 构造 key 名称
                object_info_DOTA = item_DOTA.get(dota_key, " ")  # 获取对应值
                # break  # 找到后停止循环
                # return object_info_DOTA
            except ValueError:
                continue
        elif "image" in item_DOTA and not isinstance(item_DOTA["image"], list):
            if item_DOTA["image"].split('/')[-1] == image_name:
                object_info_DOTA = item_DOTA['possible_geo_coordinates_0']
                # return object_info_DOTA
    return object_info_DOTA if object_info_DOTA else 'None'