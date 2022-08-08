import os
import PIL
from PIL import Image
import torchvision.transforms.functional as ttf


folder = r'/Users/chengwei/Work/training/temp/'

for file in os.listdir(folder):
    f_img = folder + file
    try:
        img = Image.open(f_img)
        # img = ttf.center_crop(img, output_size=(1000, 1000))
        img = img.resize((250, 250))
    except:
        continue
    
    # cropped.save(f_img)
    img.save(f_img)
    