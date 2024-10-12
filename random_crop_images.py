import os
import random
from PIL import Image

# 定义文件夹路径
input_folder = '/Users/yuqiao/Desktop/background'  # 原图片的文件夹
output_folder = '/Users/yuqiao/Desktop/background_expanded'  # 保存扩充图片的文件夹

# 获取所有图片的路径
image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

# 创建保存扩充图片的文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 扩充图片
expanded_images = []
total_images_needed = 969
original_image_count = len(image_files)

# 根据裁剪条件裁剪图片
def random_crop(image):
    width, height = image.size
    crop_width = random.randint(int(width * 0.2), width)  # 最小20%，最大不超过原宽度
    crop_height = random.randint(int(height * 0.5), height)  # 最小50%，最大不超过原高度
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

# 主循环：生成扩充图片
while len(expanded_images) < total_images_needed:
    for image_file in image_files:
        if len(expanded_images) >= total_images_needed:
            break

        # 打开图片
        image = Image.open(image_file)

        # 以0.2的概率进行随机裁剪
        if random.random() < 0:
            cropped_image = random_crop(image)
        else:
            cropped_image = image

        # 如果图片是RGBA模式，转换为RGB模式
        if cropped_image.mode == 'RGBA':
            cropped_image = cropped_image.convert('RGB')

        # 保存裁剪后的图片
        image_name = f'expanded_image_{len(expanded_images) + 1}.jpg'
        cropped_image.save(os.path.join(output_folder, image_name), 'JPEG')

        # 记录扩充的图片
        expanded_images.append(image_name)

print(f'成功生成 {len(expanded_images)} 张扩充图片。')