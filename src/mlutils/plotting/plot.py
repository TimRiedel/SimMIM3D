import matplotlib.pyplot as plt
import numpy as np

def plot_image(img):
    print(f"image shape: {img.shape}")
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"Image channel {i}")
        plt.imshow(img[i, :, :, 60].detach().cpu(), cmap="gray")
    plt.show()
    plt.savefig('image.png')
    print("Image saved")


def plot_segmentation(seg):
    print(f"segmentation shape: {seg.shape}")
    plt.figure("segmentation", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"Segmentation channel {i}")
        plt.imshow(seg[i, :, :, 60].detach().cpu())
    plt.show()
    plt.savefig('segmentation.png')
    print("Segmentation saved")