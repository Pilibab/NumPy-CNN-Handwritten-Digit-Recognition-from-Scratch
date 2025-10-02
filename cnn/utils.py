import numpy as np
import matplotlib.pyplot as plt

def plot_sample(bitmap_sample):
    img = bitmap_sample.squeeze()

    f, axes =  plt.subplots(1,2)
    axes[0].imshow(np.transpose(img), cmap="gray")
    axes[1].imshow(img, cmap="gray")
    plt.show()