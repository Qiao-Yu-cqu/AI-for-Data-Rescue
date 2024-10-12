import os
from pdf2image import convert_from_path

def convert_pdfs_in_folder(source_folder, target_folder, dpi=300, output_format='jpeg'):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.pdf'):
            # 获取完整的 PDF 文件路径
            pdf_path = os.path.join(source_folder, filename)
            
            # 转换 PDF 为图像
            images = convert_from_path(pdf_path, dpi=dpi)
            
            # 遍历生成的图像并保存
            for i, image in enumerate(images):
                # 构建目标文件名
                image_filename = f"{os.path.splitext(filename)[0]}_page_{i + 1}.{output_format}"
                image_path = os.path.join(target_folder, image_filename)
                
                # 保存图像
                image.save(image_path, output_format.upper())
                print(f"Saved {image_path}")
# 示例调用
source_folder = "/Users/yuqiao/Desktop/1989"  # 源文件夹路径
target_folder = "/Users/yuqiao/Desktop/1989_images"  # 目标文件夹路径

convert_pdfs_in_folder(source_folder, target_folder)
