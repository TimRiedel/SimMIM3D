import matplotlib.pyplot as plt
import numpy as np

def plot_image_file(img):
    print(f"image shape: {img.shape}")
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"Image channel {i}")
        plt.imshow(img[i, :, :, 60].detach().cpu(), cmap="gray")
    plt.show()
    plt.savefig('image.png')
    print("Image saved")


def plot_segmentation_file(seg):
    print(f"segmentation shape: {seg.shape}")
    plt.figure("segmentation", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"Segmentation channel {i}")
        plt.imshow(seg[i, :, :, 60].detach().cpu())
    plt.show()
    plt.savefig('segmentation.png')
    print("Segmentation saved")


def plot_img_seg_file(img, seg, slice_no=60, save_file=False):
    print(f"image shape: {img.shape}")
    print(f"segmentation shape: {seg.shape}")

    n_img_channels = img.shape[0]
    n_seg_channels = seg.shape[0]
    n_cols = max(n_img_channels, n_seg_channels)

    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(24, 6))
    for i in range(n_img_channels):
        axes[0, i].imshow(img[i, :, :, slice_no].detach().cpu(), cmap="gray")
        axes[0, i].set_title(f"Image channel {i}")
    
    for s in range(n_seg_channels):
        axes[1, s].imshow(seg[s, :, :, slice_no].detach().cpu())
        axes[1, s].set_title(f"Segmentation channel {s}")

    axes[1, 3].axis('off')
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

    if save_file:
        plt.savefig('img-seg.png')
        print("Image and Segmentation saved")


def overlay_segmentation_image(img, truth_label=None, pred_label=None, slice_no=81, img_channel=None, label_channel=None, save_file=False):
    if img_channel is not None:
        img = img[img_channel]
    if label_channel is not None:
        if truth_label is not None:
            truth_label = truth_label[label_channel]
        if pred_label is not None:
            pred_label = pred_label[label_channel]

    
    img = img[:,:,slice_no]

    plt.figure("seg_overlay", dpi=100)
    plt.imshow(img, cmap="gray")

    alpha = 0.3
    cmap_truth = plt.cm.colors.ListedColormap([(0,0,0,0), 'green']) 
    cmap_pred = plt.cm.colors.ListedColormap([(0,0,0,0), 'yellow'])
    
    if truth_label is not None:
        plt.imshow(truth_label[:,:,slice_no], alpha=alpha, cmap=cmap_truth)
    if pred_label is not None:
        plt.imshow(pred_label[:,:,slice_no], alpha=alpha, cmap=cmap_pred)

    plt.show()
    if save_file:
        plt.savefig('segmentation_overlay.png')
        print("Segmentation overlay saved")
    return plt.gcf()
    