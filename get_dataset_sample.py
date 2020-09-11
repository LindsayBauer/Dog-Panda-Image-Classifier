import matplotlib.pyplot as plt


def visualize_ds(train_ds):
    ''' Display sample of 9 images from training data set '''
    class_names = train_ds.class_names

    plt.figure(figsize=(8, 8))
    for images, labels in train_ds.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    plt.savefig('training_ds_sample.png')


if __name__ == '__main__':
    from get_datasets import get_ds
    train_ds, _ = get_ds()
    visualize_ds(train_ds)
