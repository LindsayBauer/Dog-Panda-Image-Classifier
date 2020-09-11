from pathlib import Path

import IPython
import kerastuner as kt
import tensorflow as tf
from tensorflow.keras import callbacks, layers
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential

IMG_HEIGHT, IMG_WIDTH = 150, 150


def get_augmentations():
    ''' Return augmentations to be applied to training images in preprocessing layer of model '''
    data_aug = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.2)
        ]
    )
    return data_aug


def enhance_performance(train_ds, val_ds):
    '''
    Use buffered prefetching when loading data to increase performance. Once images are loaded off 
    disk in the first epoch, cache() keeps them in memory for quicker access. prefetch() enables 
    image preprocessing to be performed while the model is training.     
    '''
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # Tune value dynamically at runtime
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def construct_model(hp):
    model = Sequential([layers.experimental.preprocessing.Rescaling(
        1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), get_augmentations()])

    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # Dropout rate to reduce overfitting
    model.add(Dropout(hp.Float('dropout', 0.3, 0.5, step=0.1, default=0.5)))
    # Convert 3D feature maps to 1D so can add fully connected layers
    model.add(Flatten())

    hp_units = hp.Choice('units', values=[128, 512], default=128)
    model.add(Dense(hp_units, activation='relu'))  # Fully connected layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def get_hypertuned_model(train_ds, val_ds):
    ''' Construct the CNN model with the identified optimal hyperparameters '''
    train_ds, val_ds = enhance_performance(train_ds, val_ds)

    # Instantiate tuner
    tuner = kt.Hyperband(construct_model,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory=Path.cwd(),
                         project_name='kt_results',
                         overwrite=True)

    class ClearTrainingOutput(tf.keras.callbacks.Callback):
        ''' Clear training outputs at end of every training step '''
        def on_train_end(*args):
            ''' Receives ClearTrainingOutput object and a dictionary of loss
            and accuracy rates for training and validation sets '''
            IPython.display.clear_output(wait=True)

    # Run hyperparameter search
    tuner.search(train_ds, epochs=10, validation_data=val_ds,
                 callbacks=[ClearTrainingOutput()])

    # Get optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal dropout rate is {best_hps.get('dropout')}.""")

    # Build model using optimal hyperparameters
    return tuner.hypermodel.build(best_hps)