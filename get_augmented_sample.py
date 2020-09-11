import matplotlib.pyplot as plt


def display_img_aug_effects(train_ds, data_aug):
    ''' Display augmentation effects on training image '''
    plt.figure(figsize=(8, 8))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_aug(images)
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    plt.savefig('augmentation.png')


if __name__ == '__main__':
    from get_datasets import get_ds
    from get_model import get_augmentations
    train_ds, _ = get_ds()
    data_aug = get_augmentations()
    display_img_aug_effects(train_ds, data_aug)
