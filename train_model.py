import datetime
import io
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tensorflow as tf
from six.moves import range
from tensorflow.keras import callbacks

from get_acc_and_loss import plot_acc_and_loss
from get_datasets import get_ds
from get_model import get_hypertuned_model
from get_ROC_curve import plot_ROC


class earlyStopCallback(tf.keras.callbacks.Callback):
    ''' Terminates training if accuracy reaches or exceeds 95% to prevent overfitting '''

    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95.0% accuracy so cancelling training!")
            self.model.stop_training = True


early_term_callback = earlyStopCallback()


def plot_to_image(figure):
    ''' Converts the matplotlib plot to a PNG image and returns it '''
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)  # Prevents figure from being displayed
    buf.seek(0)
    # Decode the PNG image to a uint8 tensor. 'image' is an RGBA
    # image with an alpha channel specifying pixel opaqueness
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add batch dimension to image shape: [height, width, channels]
    # --> [1, height, width, channels]
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    '''
    Arguments:
      cm (array, shape = [2, 2]): a confusion matrix of integer (0,1) classes
      class_names (array, shape = [2]): String names of the integer classes
    '''
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)
                   [:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


logdir = os.path.join(Path.cwd(), "logs", "image",
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(logdir):
    Path(logdir).mkdir(parents=True, exist_ok=True)
# Define the basic TensorBoard callback.
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(os.path.join(logdir, 'cm'))


def log_confusion_matrix(epoch, logs):
    # Predict class of validation images with the model
    val_pred = []
    val_labels = []
    for images, labels in val_ds:
        val_pred_raw = model.predict(images)
        for entry in val_pred_raw:
            val_pred.append(int(round(entry[0])))
        for label in labels:
            val_labels.append(label)

    # Calculate the confusion matrix
        cm = sklearn.metrics.confusion_matrix(val_labels, val_pred)
    # Plot the confusion matrix and generate it as an image
        figure = plot_confusion_matrix(cm, class_names=train_ds.class_names)
        cm_image = plot_to_image(figure)
    # Log the confusion matrix as an image summary
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


# Define the per-epoch callback
cm_callback = callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


train_ds, val_ds = get_ds()
model = get_hypertuned_model(train_ds, val_ds)

# Train the neural network
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_term_callback, tensorboard_callback, cm_callback]
)

# Save model to use for making predictions on never before seen images
model.save(os.path.join('model.h5'))


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
    plt.legend(loc='lower left')
    plt.title('Training and Validation Loss')

    plt.savefig('acc_and_loss.png')


plot_acc_and_loss()

# Generate and save the ROC curve as a .png file
plot_ROC(val_ds, model)
