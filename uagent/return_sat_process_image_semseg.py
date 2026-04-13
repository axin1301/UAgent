import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, center_of_mass
from PIL import Image, ImageDraw, ImageFont
import os
from skimage.measure import label, regionprops

def overlay_and_number_rgb_segments(
    original_img_path, seg_rgb_img_path, output_path,output_path2, alpha=0.5
):
    """
    将 RGB 分割图叠加到原图上，对每个区域编号，并标注类别名称。

    - 相同颜色视为一个类别
    - 每个连通区域都单独编号
    - 结果写入图像
    - 新增：用黑颜色字标注类别，自适应找位置
    """
    # --- 1. 读取和预处理图像 ---
    orig = Image.open(original_img_path).convert("RGB")
    w, h = 256, 256 # 强制大小
    orig = orig.resize((w, h), Image.NEAREST)
    seg_img = Image.open(seg_rgb_img_path).convert("RGB").resize((w, h), Image.NEAREST)

    # 加载额外 PNG，自动识别灰度或 RGB (这部分保持不变)
    # img_name = original_img_path.split('/')[-1].split('.')[0]
    file_name_with_extension = os.path.basename(original_img_path)
    # 分离文件名和扩展名
    img_name, file_extension = os.path.splitext(file_name_with_extension)

    flg = 0
    extra_img = None # 初始化 extra_img
    for region in ['Beijing','NewYork','London']:
        for region_part in ['4','3','2','']:
            extra_img_path = os.path.join('../test_run/', region + region_part + '_2020_out_imgs_1300/', img_name + '_pred.png')
            extra_img_path2 = os.path.join('../test_run/', region + region_part + '_2020_out_imgs_1300/', img_name.split('_')[0]+'_'+ img_name.split('_')[1]+ '_pred.png')
            if os.path.exists(extra_img_path):
                flg = 1
                extra_img = Image.open(extra_img_path).resize((w, h), Image.NEAREST)
                print('extra_img_path: ',extra_img_path)
                break
            elif os.path.exists(extra_img_path2):
                flg = 1
                # x_idx = int(img_name.split('_')[2])
                # y_idx = int(img_name.split('_')[3])

                x_idx = img_name.split('_')[2]
                y_idx = img_name.split('_')[3]

                if x_idx=='top':
                    x_idx = 0
                else:
                    x_idx = 1
                
                if y_idx=='left':
                    y_idx = 0
                else:
                    y_idx = 1

                full_image= Image.open(extra_img_path2).resize((w*2, h*2), Image.NEAREST)

                quadrant_width = w
                quadrant_height = h
                
                left = x_idx * quadrant_width
                upper = y_idx * quadrant_height
                right = left + quadrant_width
                lower = upper + quadrant_height

                # 4. 裁剪图像
                extra_img = full_image.crop((left, upper, right, lower))
                print('extra_img_path2: ',extra_img_path2)
                break
        if flg == 1:
            break
    
    if extra_img is None:
        # 如果没有找到 extra_img，可以默认一个全黑或全白的图像，或者报错
        print(f"Warning: No extra_img found for {img_name}. Proceeding without it.")
        extra_array = np.zeros((h, w), dtype=np.uint8) # 默认一个全黑的掩码
    else:
        extra_array = np.array(extra_img)
    
    print(extra_array.shape)

    # 判断白色区域：支持灰度图和RGB图
    if len(extra_array.shape) == 2:
        white_mask = extra_array == 255
    else:
        white_mask = np.all(extra_array == 255, axis=-1)

    # 加载分割图像并转换为数组
    seg_array = np.array(seg_img)

    # 定义旧颜色（原始 segmentation 图中的 RGB）和新颜色映射（用于叠加）以及对应的类别名称
    color_mapping = {
        (56, 148, 175): (200, 80, 80),     # building
        (143, 251, 128): (150, 150, 150), # road
        (219, 154, 225): (80, 170, 250),  # water
        (146, 179, 117): (160, 130, 90),  # barren
        (29, 196, 92): (60, 140, 70),     # forest
        (124, 79, 78): (255, 210, 90),    # agriculture
    }
    
    # 新增：类别名称映射
    class_names = {
        (56, 148, 175): "Building",
        (143, 251, 128): "Road",
        (219, 154, 225): "Water",
        (146, 179, 117): "Barren",
        (29, 196, 92): "Forest",
        (124, 79, 78): "Agriculture",
    }

    # 替换 seg_array 中的旧颜色为新颜色
    new_seg_array = np.copy(seg_array)
    for old_rgb, new_rgb in color_mapping.items():
        mask = np.all(seg_array == old_rgb, axis=-1)
        new_seg_array[mask] = new_rgb

    # 道路颜色
    road_color_orig = np.array([143, 251, 128]) # 原始的道路颜色
    road_color_mapped = color_mapping.get(tuple(road_color_orig), road_color_orig) # 使用映射后的道路颜色
    print(new_seg_array.shape)
    # 在白色区域将 seg_image 赋值为道路颜色 (使用映射后的颜色)
    new_seg_array[white_mask] = road_color_mapped

    # 更新 seg_img (用于后续叠加和绘图)
    seg_img = Image.fromarray(new_seg_array.astype(np.uint8))

    seg_img.save(output_path2)
    print("保存完成:", output_path2)

    # --- 2. 图像叠加 ---
    # 创建最终图像，首先将原图作为背景
    final_img = Image.blend(orig, seg_img, alpha=alpha) # 将seg_img叠加到orig上
    
    # --- 3. 标注类别名称 ---
    draw = ImageDraw.Draw(final_img)

    # 尝试加载字体，如果找不到则使用默认字体
    try:
        # 你可以根据自己的系统修改字体路径和大小
        # 例如：'/System/Library/Fonts/Arial.ttf' (macOS)
        # 'C:/Windows/Fonts/simhei.ttf' (Windows 上的黑体，如果需要支持中文)
        font_path = "arial.ttf" # 假设 Arial 字体在你的系统路径中
        font_size = 14 # 初始字体大小
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Font '{font_path}' not found. Using default font.")
        font = ImageFont.load_default() # Fallback to default font
        font_size = 10 # 默认字体可能较小

    # 遍历每种原始颜色对应的类别
    for old_rgb, class_name in class_names.items():
        # 获取该类别在原始分割图中的掩码
        class_mask = np.all(seg_array == old_rgb, axis=-1) # 使用原始seg_array来识别区域

        # 如果这个类别在图像中存在
        if np.any(class_mask):
            # 使用 skimage.measure.label 找到连通区域
            labeled_mask = label(class_mask)
            
            # 使用 regionprops 获取每个连通区域的属性
            for region in regionprops(labeled_mask):
                # 跳过非常小的区域，它们可能只是噪声
                if region.area < 30: # 你可以根据图像大小调整这个阈值
                    continue

                # 提取区域的中心点 (centroid)
                # centroid 是 (row, col) 即 (y, x)
                center_y, center_x = region.centroid

                # 确保中心点在图像边界内
                center_x = int(max(0, min(w - 1, center_x)))
                center_y = int(max(0, min(h - 1, center_y)))

                # 尝试根据文字大小调整位置以居中 (可选)
                bbox = draw.textbbox((0, 0), class_name, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 计算文字左上角坐标，使其大致居中
                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2

                # 确保文字完全在图像内部
                text_x = max(0, min(w - text_width, text_x))
                text_y = max(0, min(h - text_height, text_y))

                # 绘制文字 (黑色)
                draw.text((text_x, text_y), class_name, fill=(0, 0, 0), font=font)
                # draw.text((center_x, center_y), class_name, fill=(0, 0, 0), font=font) # 简单粗暴的中心点
                break


    # 保存结果
    final_img.save(output_path)
    print("保存完成:", output_path)


def return_sat_process_image_semsag(image_path,crop=False):
    os.makedirs('tmp_results',exist_ok=True)

    # img_name = image_path.split('/')[-1].split('.')[0] #'43578_65513' #'10880_16359'#_pred
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    original_img_path=image_path
    seg_rgb_img_path= "received_processed_image_base64.png"#"../API_test/mapping_stv_sat_example_images/"+img_name+"_semseg.png"
    output_path="tmp_results/"+img_name+"_semseg.png"
    output_path2="tmp_results/"+img_name+"_semseg_pure.png"

    overlay_and_number_rgb_segments(
            # original_img_path="sat_BJ_mix/sat_BJ_mix/"+img_name+".png",
            # original_img_path="../London_citybench_zl17_merge/"+img_name+".png",
            original_img_path=original_img_path,
            seg_rgb_img_path=seg_rgb_img_path,
            output_path=output_path,
            output_path2 = output_path2,
            alpha=0.7
        )
    return output_path