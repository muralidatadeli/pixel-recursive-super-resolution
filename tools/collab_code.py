import os 
import sys
sys.path.insert(0, './pixel-recursive-super-resolution/tools/')
from predict import *

print("started executing")
output_dir = "pixel-recursive-super-resolution/output_images"
input_dir = "pixel-recursive-super-resolution/test_images"
ckpt = "pixel-recursive-super-resolution/models/model.ckpt-170000"
print(os.listdir(input_dir))
for image in os.listdir(input_dir):
  output_img_path = os.path.join(output_dir, image)
  input_img_path = os.path.join(input_dir, image)
  os.system('python pixel-recursive-super-resolution/tools/predict.py %s %s %s'%(input_img_path, output_img_path,ckpt))