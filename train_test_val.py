import os
import shutil
input_dir = 'E:\Dataset\zhitang\Dataset_Zhitang_Yolo5'
output_dir = 'E:\Dataset\zhitang\Dataset_Zhitang_Yolo5\ZhitangYolo5'

in_img_dir = os.path.join(input_dir, 'Images')
in_label_dir = os.path.join(input_dir, 'Labels')
out_img_dir = os.path.join(output_dir, 'images')
out_label_dir = os.path.join(output_dir, 'labels')

splits = {'train','test','valid'}
files = os.listdir(in_img_dir)
count = len(files)

for f in files:
	print(f)
    src = os.path.join(input_dir,f)
    shutil.copyfile(src, dst)




