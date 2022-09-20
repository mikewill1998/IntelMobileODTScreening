import os
import PIL
from PIL import Image
import torchvision.transforms.functional as ttf


folder = r'/Users/chengwei/Work/training/temp/'

for file in os.listdir(folder):
    f_img = folder + file
    try:
        img = Image.open(f_img)
        img = ttf.center_crop(img, output_size=(2500, 2500))
        img = img.resize((28, 28))
    except:
        continue
    
    # cropped.save(f_img)
    img.save(f_img)
    