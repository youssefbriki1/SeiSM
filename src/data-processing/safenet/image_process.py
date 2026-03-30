import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import pickle

patches = []
cols = 15
rows = 8
patch_size = 50
img_array = []
img = []
width = 750
height = 375
ch0, ch1, ch2, ch3 = [], [], [], []
png_data = []

def load_image():
    global img, width, height
    img = Image.open("maps/basic_map.png")
    img = img.convert('L')  # Convert to grayscale

    # print(f"Original size: {img.size}")  # (width, height)

    # Resize to 750x375 (squeeze height, don't crop)
    img = img.resize((width, height), Image.LANCZOS)
    img = np.array(img, dtype=np.float32) / 255.0  # Single-channel float, 0-1
    # print(f"After squeeze resize: {img.shape}")


def add_padding_and_split():
    global patches, cols, rows, patch_size, img_array

    # Add 25 white padding to the top to make 750x400
    img_array = img
    white_pad = np.ones((25, 750), dtype=np.float32)
    img_array = np.concatenate([white_pad, img_array], axis=0)
    print(f"After padding: {img_array.shape}")  # should be (400, 750)

    # Break into 15 x 8 patches of 50x50, bottom-to-top, left-to-right
    # 15 columns (750 / 50) and 8 rows (400 / 50)
 
    # Bottom-to-top: row index from (rows-1) down to 0
    # Left-to-right: col index from 0 to (cols-1)
    for r in range(rows - 1, -1, -1):  # bottom to top
        for c in range(cols):  # left to right
            y_start = r * patch_size
            y_end = y_start + patch_size
            x_start = c * patch_size
            x_end = x_start + patch_size
            patch = img_array[y_start:y_end, x_start:x_end]
            patches.append(patch)

    patches = np.array(patches)
    print(f"Patches shape: {patches.shape}")  # should be (120, 50, 50, ...)


def add_circle_to_original_map(mag, x, y):
    global img, width, height
    # mag < 4: tiny dot (1px), then exponentially larger
    radius = 0
    alpha = 0

    if (mag < 0):
        return
    if mag < 4:
        radius = 1
        alpha = (mag/4) * 0.22
    elif 4 <= mag < 5:
        radius = 2 + ((mag - 4)/1) * 0.5
        alpha = 0.55
    elif 5 <= mag < 6:
        radius = 2.5 + ((mag - 4)/1) * 0.5
        alpha = 0.52
    elif 6 <= mag < 7:
        radius = 3 + ((mag - 4)/1) * 3
        alpha = 0.49
    elif mag > 7:
        radius = 7
        alpha = 0.45
        
    # Only compute mask in a small box around the circle
    r = int(radius) + 1
    y0, y1 = max(0, y - r), min(height, y + r + 1)
    x0, x1 = max(0, x - r), min(width, x + r + 1)
    
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    
    # Multiply intensity (gradual darkening, avoids black saturation)
    # Only darken pixels still above 0.35
    region = img[y0:y1, x0:x1]
    apply_mask = mask & (region >= 0.35)
    region[apply_mask] *= (1 - alpha)
    img[y0:y1, x0:x1] = region

def render_eq_distribution(year):
    print("Calculating eq distribution map ... ")
    global cols, rows, img
    
    DATA_DIR = Path(__file__).resolve().parent / 'dataset'
    raw_csv = os.path.join(DATA_DIR, "1970-2021_11_EARTH_final_with_patchnum.csv")
    df = pd.read_csv(raw_csv, usecols=['magnitude', 'x', 'y', 'onlydate', 'region'])
    df['_onlydate_dt'] = pd.to_datetime(df['onlydate'])

    window_start = pd.Timestamp(year=year - 1, month=11, day=17)
    window_end = pd.Timestamp(year=year, month=11, day=16)

    df = df[(df['_onlydate_dt'] >= window_start) & (df['_onlydate_dt'] <= window_end)]

    for _, eq in df.iterrows():
        local_x = int(eq['x'] * width)
        local_y = int((1 - eq['y']) * height)  # flip y for image coordinates
        add_circle_to_original_map(eq['magnitude'], local_x, local_y)

def show_map():
    image = None
    for row in range(rows - 1, -1, -1):  # bottom to top
        row_image = None
        for col in range(15):
            cnt = 15 * row + col
            patch_image = patches[cnt]
            cnt += 1
            if row_image is None:
                row_image = patch_image
            else:
                row_image = np.concatenate([row_image, patch_image], axis=1)
        if image is None:
            image = row_image
        else:
            image = np.concatenate([image, row_image], axis=0)

    plt.figure(figsize=(16, 8))
    plt.imshow(image, cmap='hot')
    plt.title("Reassembled Map from Patches")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_4_channels(filename):
    global ch0, ch1, ch2, ch3
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    raw_png = data['png']
    # Channels 0-3 are static (same across all years), take from year 0
    ch0 = raw_png[0][:, :, :, 0]  # (85, 50, 50)
    ch1 = raw_png[0][:, :, :, 1]
    ch2 = raw_png[0][:, :, :, 2]
    ch3 = raw_png[0][:, :, :, 3]

def reset():
    global patches, img_array, img, height
    patches = []
    img_array = []
    img = []
    width  = 750
    height = 375

def combine_channels():
    global png_data
    DATA_DIR = Path(__file__).resolve().parent / 'dataset'
    patch_csv = os.path.join(DATA_DIR, "png_list_to_patchxy.csv")
    df = pd.read_csv(patch_csv, usecols=['x', 'y'])
    ch4 = []
    for _, patch in df.iterrows():
        x = int(patch['x'])
        y = int(patch['y'])
        idx = 15 * y + x
        ch4.append(patches[idx])
    ch4 = np.array(ch4)  # (85, 50, 50)

    # Stack all 5 channels: (85, 50, 50, 5)
    png = np.stack([ch0, ch1, ch2, ch3, ch4], axis=-1)
    print(f"Combined 5 channels, shape: {png.shape}, total years: {len(png_data)}")
    return png

def generate_map (start_year, end_year):
    global png_data
    png_data = []

    # original map
    filename = 'maps/eqs_and_png_data_for_eval_10y_in_11_16.pickle'
    load_4_channels(filename)

    for year in range(start_year, end_year+1):
        print(f"Processing map for year {year}")
        load_image()
        render_eq_distribution(year)
        add_padding_and_split()
        png = combine_channels()
        png_data.append(png)
        # show_map()
        reset()
    return png_data

# def main():
#     generate_map(2019, 2019)
#     data = {}
#     data['png'] = png_data
#     print(f"Final png_data: {len(png_data)} years, each shape: {png_data[0].shape}")

#     with open("map-data/png_data.pkl", 'wb') as file:
#         pickle.dump(data, file)

# # main()
