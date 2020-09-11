import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from tensorflow.keras.models import load_model


def plot_ROC(val_ds, model, fname = 'ROC_curve'):
    """ Plot the ROC curve """
    val_ds_labels = []
    val_ds_predictions = []
    # Cycle through the batches of validation images
    for image_batch, label_batch in val_ds:
        val_batch_pred = model.predict(image_batch)
        for entry in val_batch_pred:
            val_ds_predictions.append(entry[0])
        for label in label_batch:
            val_ds_labels.append(label)

    # False positive rate and true positive rate 
    fpr, tpr, _ = roc_curve(val_ds_labels, val_ds_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='#FF7F50',
             lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#20B2AA', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, labelpad=8)
    plt.ylabel('True Positive Rate', fontsize=12, labelpad=8)
    plt.title('Receiver Operating Characteristic', fontsize=15)
    plt.legend(loc='lower right')
    
    plt.savefig(fname + '.png')

if __name__ == '__main__':
    from get_datasets import get_ds
    _ , val_ds = get_ds()
    model = load_model('model.h5')
    plot_ROC(val_ds, model)
