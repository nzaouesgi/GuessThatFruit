import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import listdir
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":

    target_size = (64, 64)

    file_path = sys.argv[1]
    model_name = sys.argv[2]

    if model_name == "cnn1.keras":
        target_size = (32, 32)
    elif model_name == "linear1.keras":
        target_size = (64, 64)
    elif model_name == "mlp1.keras":
        target_size = (32, 32)
    elif model_name == "mlp2.keras":
        target_size = (48, 48)
    elif model_name == "resnet1.keras":
        target_size = (64, 64)

    model = load_model(f'../Models/{model_name}')

    img = image.load_img(file_path, target_size=target_size)
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.

    prediction = model.predict(img_tensor)

    print(np.argmax(prediction))