from pathlib import Path

import tensorflow as tf

# Parameters for the loader
BATCH_SIZE = 16
IMG_HEIGHT, IMG_WIDTH = 150, 150

def get_ds(data_dir = Path('animal_photos')):
    ''' Create and return the training and validation datasets '''
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=140,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=140,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    return train_ds , val_ds
