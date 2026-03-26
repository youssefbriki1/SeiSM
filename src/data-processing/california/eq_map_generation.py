import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import pickle

# patches = []
# cols = 15
# rows = 8
# patch_size = 50
# img = []
# width = 750
# height = 375
ch0, ch1, ch2, ch3 = [], [], [], []
png_data = []

class ImageProcessing:
    def __init__(self, map_path, event_csv_path, patch_csv_path, cols, rows, width, height, patch_size, padding, bbox:tuple=(-125, 32, -113, 42)):
        self.map_path = map_path
        self.event_csv_path = event_csv_path
        self.patch_csv_path = patch_csv_path
        self.cols = cols
        self.rows = rows
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.padding = padding
        self.patches = []
        self.xmin, self.ymin, self.xmax, self.ymax = bbox
        self.img = []
        self._load_image()

    def _load_image(self):
        self.img = Image.open(self.map_path)
        self.img = self.img.convert('L')  # Convert to grayscale

        self.img = self.img.resize((self.width, self.height), Image.LANCZOS)
        self.img = np.array(self.img, dtype=np.float32) / 255.0  # Single-channel float, 0-1


    def _add_padding_and_split(self):

        # Add white padding to the top to make 750x400
        img_array = self.img
        white_pad = np.ones((self.padding, self.width), dtype=np.float32)
        img_array = np.concatenate([white_pad, img_array], axis=0)
        print(f"After padding: {img_array.shape}")  # should be (400, 750)

        # Break into 15 x 8 patches of 50x50, bottom-to-top, left-to-right
        # 15 columns (750 / 50) and 8 rows (400 / 50)
    
        # Bottom-to-top: row index from (rows-1) down to 0
        # Left-to-right: col index from 0 to (cols-1)
        self.patches = []
        for r in range(self.rows - 1, -1, -1):  # bottom to top
            for c in range(self.cols):  # left to right
                y_start = r * self.patch_size
                y_end = y_start + self.patch_size
                x_start = c * self.patch_size
                x_end = x_start + self.patch_size
                patch = img_array[y_start:y_end, x_start:x_end]
                self.patches.append(patch)

        self.patches = np.array(self.patches)
        print(f"Patches shape: {self.patches.shape}")  # should be (120, 50, 50, ...)


    def _add_circle_to_original_map(self, mag, x, y):
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
        
        # Prevent negative bounds which create invalid slice shapes
        x0 = max(0, x - r)
        x1 = max(0, min(self.width, x + r + 1))
        y0 = max(0, y - r)
        y1 = max(0, min(self.height, y + r + 1))
        
        # If the event is completely outside the image window, skip it
        if x0 >= x1 or y0 >= y1:
            return
            
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        
        # Multiply intensity (gradual darkening, avoids black saturation)
        # Only darken pixels still above 0.35
        region = self.img[y0:y1, x0:x1]
        apply_mask = mask & (region >= 0.35)
        
        region[apply_mask] *= (1 - alpha)
        self.img[y0:y1, x0:x1] = region

    def _render_eq_distribution(self, year):
        print("Calculating eq distribution map ... ")
        
        # raw_csv = "data/1970-2021_11_EARTH_final_with_patchnum.csv"
        df = pd.read_csv(self.event_csv_path, usecols=['magnitude', 'longitude', 'latitude', 'onlydate', 'region'])
        df['_onlydate_dt'] = pd.to_datetime(df['onlydate'])

        window_start = pd.Timestamp(year=year - 1, month=11, day=17)
        window_end = pd.Timestamp(year=year, month=11, day=16)

        df = df[(df['_onlydate_dt'] >= window_start) & (df['_onlydate_dt'] <= window_end)]

        for _, eq in df.iterrows():
            local_x = int((eq['longitude'] - self.xmin) / (self.xmax - self.xmin) * self.width)
            local_y = int((self.ymax - eq['latitude']) / (self.ymax - self.ymin) * self.height)
            self._add_circle_to_original_map(eq['magnitude'], local_x, local_y)

    def _show_map(self):
        image = None
        for row in range(self.rows - 1, -1, -1):  # bottom to top
            row_image = None
            for col in range(self.cols):
                cnt = self.cols * row + col
                patch_image = self.patches[cnt]
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

    def _load_4_channels(self, filename):
        global ch0, ch1, ch2, ch3
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        raw_png = data['png']
        
        # Channels 0-3 are static (same across all years), take from year 0
        ch0 = raw_png[0][:, :, :, 0]  # (85, 50, 50)
        ch1 = raw_png[0][:, :, :, 1]
        ch2 = raw_png[0][:, :, :, 2]
        ch3 = raw_png[0][:, :, :, 3]

    def _combine_channels(self):
        global png_data
        # self.patch_csv_path = "data/png_list_to_patchxy.csv"
        df = pd.read_csv(self.patch_csv_path, usecols=['x', 'y'])
        ch4 = []
        for _, patch in df.iterrows():
            x = int(patch['x'])
            y = int(patch['y'])
            idx = 15 * y + x
            ch4.append(self.patches[idx])
        ch4 = np.array(ch4)  # (85, 50, 50)

        # Stack all 5 channels: (85, 50, 50, 5)
        png = np.stack([ch0, ch1, ch2, ch3, ch4], axis=-1)
        print(f"Combined 5 channels, shape: {png.shape}, total years: {len(png_data)}")
        return png

    def generate_eq_map(self, year):
        self._render_eq_distribution(year)
        eq_map = self.img
        self._load_image()
        return eq_map
    
    def generate_map (self, year):
        global png_data
        png_data = []

        # original map
        # filename = 'maps/eqs_and_png_data_for_eval_10y_in_11_16.pickle'
        # load_4_channels(filename)

        print(f"Processing map for year {year}")
        self._render_eq_distribution(year)
        self._add_padding_and_split()
        self._show_map()
        self._load_image()
        # return png_data

# def main():
#     generate_map(2019, 2019)
#     data = {}
#     data['png'] = png_data
#     print(f"Final png_data: {len(png_data)} years, each shape: {png_data[0].shape}")

#     with open("map-data/png_data.pkl", 'wb') as file:
#         pickle.dump(data, file)

# # main()

def main():
    CEED_map = ImageProcessing(
        map_path="data/CEED/map_outline.jpg",
        event_csv_path="data/CEED/events_preprocessed_1987_2010.csv",
        patch_csv_path="data/CEED/png_list_to_patchxy_california.csv",
        cols= 8,
        rows = 8,
        width= 512,
        height = 512,
        patch_size= 64,
        padding = 0
    )
    CEED_map.generate_map(2002)

# main()