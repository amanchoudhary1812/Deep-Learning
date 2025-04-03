import cv2
import os
import numpy as np

image_size = (224, 224) 

def preprocess_images(folder):
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            img = cv2.resize(img, image_size)  
            img = img / 255.0  
            img_uint8 = (img * 255).astype(np.uint8)
            cv2.imwrite(img_path, img_uint8) 

preprocess_images("train/")
preprocess_images("test/")
preprocess_images("validation/")

print("Preprocessing complete! Images resized and saved in COLOR.")
