import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model


def predictor(img_path):
    ''' Predict class of image '''
    img = Image.open(img_path)
    img_resized = np.reshape(img.resize((150, 150)), [1, 150, 150, 3])

    model = load_model('model.h5')
    img_pred = model.predict(img_resized)[0][0]

    plt.imshow(img)
    plt.title(r'$\bf{' + 'Dog' + '}$' + ': ' + str(round(100 * (1 - img_pred), 2)) + '% confidence, ' +
              r'$\bf{' + 'Panda' + '}$' + ': ' + str(round(100 * img_pred, 2)) + '% confidence')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    import sys
    img_path = sys.argv[1]
    predictor(img_path)