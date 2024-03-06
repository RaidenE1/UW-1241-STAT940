import os
import numpy as np

def get_seperated_dataset(data_dir):
    images = os.listdir(data_dir)
    np.random.shuffle(images)
    train_images = images[:int(len(images) * 0.9)]
    val_images = images[int(len(images)*0.9):]
    return train_images, val_images
    