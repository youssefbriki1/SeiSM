import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


filename = 'data/training_output.pickle'
with open(filename, 'rb') as file:
    data = pickle.load(file)

png_data = data['png']

print(f"png data length", len(png_data))
print(f"png data element shape", png_data[0].shape)


matrix = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
]


def show_channel (year, channel):
    global png_data
    blank = np.ones((50, 50))
    image = None

    cnt = 0
    for row in matrix:
        row_image = None
        for patch in row:
            if patch == 0:
                patch_image = np.ones((50, 50))
            else:
                cnt += 1
                patch_image = png_data[year][cnt-1, :, :, channel]
            if row_image is None:
                row_image = patch_image
            else:
                row_image = np.concatenate([row_image, patch_image], axis=1)
        if image is None:
            image = row_image
        else:
            image = np.concatenate([image, row_image], axis=0)
    return image

def show_all_channels (year):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    for ch in range(5):
        row, col = divmod(ch, 2)
        img = show_channel(year, ch)
        cmap='viridis'
        if(ch == 3):
            cmap = "gray"
        elif(ch == 4):
            cmap = 'hot'
        axes[row, col].imshow(img, cmap=cmap)
        axes[row, col].set_title(f"Channel {ch}")

    axes[2, 1].axis('off')  # hide the empty 6th cell
    plt.tight_layout()
    plt.show()

def show_eq_dist_map(year):
    """Show our self-calculated eq distribution map"""
    img = show_channel(year, 4)
    plt.figure(figsize=(16, 8))

    import matplotlib.cm as cm
    colored = cm.hot(img)  # returns RGBA float array (H, W, 4)
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)  # drop alpha, to uint8
    Image.fromarray(colored_uint8).save(f"eq_dist_raw_{year + 2001}.png")

    plt.imshow(img, cmap="hot")
    plt.title("Dist map")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_rgb_map(year):
    """Combine channels 0-2 into a single RGB map."""
    global png_data
    image = None

    cnt = 0
    for row in matrix:
        row_image = None
        for patch in row:
            if patch == 0:
                patch_image = np.ones((50, 50, 3))
            else:
                cnt += 1
                patch_image = np.stack([
                    png_data[year][cnt-1, :, :, 0],
                    png_data[year][cnt-1, :, :, 1],
                    png_data[year][cnt-1, :, :, 2],
                ], axis=-1)
            if row_image is None:
                row_image = patch_image
            else:
                row_image = np.concatenate([row_image, patch_image], axis=1)
        if image is None:
            image = row_image
        else:
            image = np.concatenate([image, row_image], axis=0)

    # Normalize to [0, 1] for display
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

target_year = 2011
start_year = 1971
# show_eq_dist_map(target_year - start_year)
show_all_channels(target_year - start_year)
