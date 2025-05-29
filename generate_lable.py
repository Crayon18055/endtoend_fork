import os

def generate_label_file(image_dir, output_txt):
    """
    读取指定目录下所有图片，生成一个txt文件。
    每行有7列，用逗号分割，最后一列是图片文件名去除.jpg，其他列用0填充。
    
    Args:
        image_dir (str): 图片所在目录。
        output_txt (str): 输出的txt文件路径。
    """
    # 获取目录下所有文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # 打开输出文件
    with open(output_txt, 'w') as txt_file:
        for image_file in image_files:
            # 去除文件名的扩展名
            file_name_without_ext = os.path.splitext(image_file)[0]
            # 构造一行数据，最后一列是文件名，其他列填充0
            line = "0,0,0,0,0,0," + file_name_without_ext
            # 写入文件
            txt_file.write(line + '\n')

# 示例使用
image_directory = "filtered_data/data3/images"  # 替换为图片目录路径
output_file = "filtered_data/data3/lable.txt"   # 替换为输出txt文件路径
generate_label_file(image_directory, output_file)