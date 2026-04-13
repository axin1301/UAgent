from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

# 定义你的颜色-标签字典
LABEL_COLORS = {
    0: [128, 63, 127],    # road
    1: [243, 35, 232],    # sidewalk
    2: [70, 70, 70],      # building
    3: [102, 102, 156],   # wall
    4: [190, 153, 153],   # fence
    5: [153, 153, 153],   # pole
    6: [250, 170, 30],    # traffic light
    7: [220, 220, 0],     # traffic sign
    8: [107, 142, 35],    # vegetation
    9: [152, 251, 152],   # terrain
    10: [70, 130, 180],   # sky
    11: [223, 21, 61],    # person
    12: [0, 0, 142],      # rider
    13: [0, 0, 70],       # car
    14: [0, 60, 100],     # truck
    15: [0, 80, 100],     # bus
    16: [0, 0, 230],      # train
    17: [119, 11, 32],    # motorcycle
    18: [0, 0, 255],      # bicycle
}

# First, create a mapping from label ID to label name string
LABEL_ID_TO_NAME = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
}

# Now, create the COLOR_TO_LABEL_NAME mapping using the LABEL_COLORS and LABEL_ID_TO_NAME
COLOR_TO_LABEL_NAME = {
    tuple(color_rgb): LABEL_ID_TO_NAME[label_id]
    for label_id, color_rgb in LABEL_COLORS.items()
}

def add_single_label_per_class(image_path, output_path, font_size=15, text_color=(255, 255, 255), outline_color=(0,0,0)):
    """
    在分割图上为每个类别只标记一个名称，位置为其所有像素的平均中心。

    Args:
        image_path (str): 分割图的路径（例如 .jpg）。
        output_path (str): 保存带标签图像的路径（例如 .png）。
        font_size (int): 字体大小。
        text_color (tuple): 文本颜色 (R, G, B)。
        outline_color (tuple): 文本描边颜色 (R, G, B)，用于提高可读性。
    """
    try:
        img = Image.open(image_path).convert("RGB") # 确保图像是RGB模式
        draw = ImageDraw.Draw(img)

        # 尝试加载一个TTF字体
        try:
            # 优先使用一个常用字体，如 Arial
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # 如果找不到 Arial，尝试一个通用字体，或者PIL的默认字体
            try:
                # 尝试一个Linux或macOS上可能有的字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except IOError:
                # Fallback to default PIL font (通常很小)
                print("警告: 找不到自定义字体，使用Pillow默认字体。文本可能很小。")
                font = ImageFont.load_default()

        img_np = np.array(img)
        height, width, _ = img_np.shape
        
        # 跟踪已经处理过的颜色，确保每个颜色只标记一次
        processed_colors = set() 

        # 遍历图像的每个像素，找到第一个匹配的颜色就标记
        for y in range(height):
            for x in range(width):
                pixel_color = tuple(img_np[y, x])
                
                # 如果这个颜色对应一个标签，并且我们还没有处理过这个颜色
                if pixel_color in COLOR_TO_LABEL_NAME and pixel_color not in processed_colors:
                    label_name = COLOR_TO_LABEL_NAME.get(pixel_color)
                    
                    if label_name:
                        # 找到第一个出现的像素位置 (x, y)
                        text_x = x
                        text_y = y
                        
                        # 获取文本大小
                        text_bbox = draw.textbbox((0,0), label_name, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]

                        # 调整文本绘制位置，使其居中于找到的像素点
                        # 注意：如果像素点靠近图像边缘，文本可能会被截断，
                        # 可以根据需要调整偏移量或做更复杂的边缘检查。
                        text_x = max(0, min(text_x - text_width // 2, width - text_width))
                        text_y = max(0, min(text_y - text_height // 2, height - text_height))


                        # 绘制文本描边（可选）
                        outline_offset = 1 
                        for dx, dy in [(-outline_offset, -outline_offset), (-outline_offset, outline_offset),
                                       (outline_offset, -outline_offset), (outline_offset, outline_offset)]:
                            draw.text((text_x + dx, text_y + dy), label_name, font=font, fill=outline_color)
                        
                        # 绘制实际文本
                        draw.text((text_x, text_y), label_name, font=font, fill=text_color)
                        
                        # 标记这个颜色已经处理过，不再重复标记
                        processed_colors.add(pixel_color)
                        
                        # 如果所有已知的颜色都已处理，可以提前退出循环以提高效率
                        if len(processed_colors) == len(COLOR_TO_LABEL_NAME):
                            break # Break from inner loop
            if len(processed_colors) == len(COLOR_TO_LABEL_NAME):
                break # Break from outer loop as well

        # 保存带标签的图像
        img.save(output_path)
        print(f"带标签的图像已保存到: {output_path}")

    except FileNotFoundError:
        print(f"错误: 图像文件未找到: {image_path}")
    except Exception as e:
        print(f"处理图像时发生错误: {e}")


def compute_pixel_percentage_dict(
    image_path,
    label_colors=LABEL_COLORS,
    label_id_to_name=LABEL_ID_TO_NAME,
    round_ndigits=2
):
    """
    Args:
        image_path: RGB 语义分割图路径
        label_colors: {label_id: [R, G, B]}
        label_id_to_name: {label_id: 'road', ...}
    Returns:
        dict: {'road_pixel_percentage': 0.31, ...}
    """

    # 读图（OpenCV 默认 BGR）
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    total_pixels = h * w

    result = {}

    for label_id, color in label_colors.items():
        label_name = label_id_to_name[label_id]

        color = np.array(color, dtype=np.uint8)

        # 精确颜色匹配
        mask = np.all(img == color, axis=-1)
        pixel_count = int(mask.sum())

        percentage = round(pixel_count / total_pixels, round_ndigits)

        key = f"{label_name}_pixel_percentage"
        result[key] = percentage

    return result