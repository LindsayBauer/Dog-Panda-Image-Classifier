import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from get_datasets import get_ds
from get_model import enhance_performance, get_augmentations
from get_ROC_curve import plot_ROC

IMG_HEIGHT, IMG_WIDTH = 150, 150

train_ds, val_ds = get_ds()
train_ds, val_ds = enhance_performance(train_ds, val_ds)
data_aug = get_augmentations()


def train(data_aug):
    # Base model preprocessing method; scales pixel values from [0-255] range to [-1,1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    '''
    Create base model using CNN model pre-trained on ImageNet dataset. This feature 
    extractor converts each 150x150x3 image into a 5x5x1280 block of features. 
    '''
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                                   weights='imagenet',
                                                   include_top=False)  # Don't include ImageNet classifier at the top

    '''
    Freeze all layers in convolutional base model to prevent weights from 
    updating during training; Move all weights from trainable to non-trainable.
    '''
    base_model.trainable = False

    # Convert features to single 1280 element vector per image
    global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
    # Convert features into a single prediction per image
    prediction_layer = tf.keras.layers.Dense(1)

    '''
    Chain together data augmentation, rescaling, base model and feature extraction layers.
    training=False so that the model's BatchNormalization layer will run in inference mode and
    not update its mean and variance statistics.
    '''
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_aug(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_avg_layer(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


model = train(data_aug)

# Train the model
history = model.fit(train_ds,
                    epochs=10,
                    validation_data=val_ds)


def plot_acc_and_loss():
    """ Plot the per epoch accuracy and loss for validation and training sets """
    # Accuracy results per epoch for training and validation data sets
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Loss results per epoch for training and validation data sets
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(acc)), acc, color='#9ACD32',
             lw=3, label='Training Accuracy')
    plt.plot(range(len(acc)), val_acc, color='#1E90FF',
             lw=3, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(range(len(loss)), loss, color='#9ACD32',
             lw=3, label='Training Loss')
    plt.plot(range(len(loss)), val_loss, color='#1E90FF',
             lw=3, label='Validation Loss')
    plt.xlabel("Epoch", labelpad=8)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('transfer_learning_acc_and_loss.png')


plot_acc_and_loss()


# Generate and save the ROC curve as a .png file
plot_ROC(val_ds, model, 'transfer_learning_ROC')
