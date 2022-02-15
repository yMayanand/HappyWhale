import re
import os
import gc
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
from tqdm.notebook import tqdm
from joblib import Parallel, delayed


# read the csv file
df = pd.read_csv("/kaggle/input/happy-whale-and-dolphin/train.csv")


# add image-paths
df["image_path"] = df["image"].apply(
    lambda x: os.path.join("/kaggle/input/happy-whale-and-dolphin/train_images", x)
)

# fixing typo
df["species"] = df["species"].str.replace("bottlenose_dolpin", "bottlenose_dolphin")
df["species"] = df["species"].str.replace("kiler_whale", "killer_whale")

# making classes
df.loc[df.species.str.contains("beluga"), "species"] = "beluga_whale"
df.loc[df.species.str.contains("globis"), "species"] = "short_finned_pilot_whale"
df.loc[df.species.str.contains("pilot_whale"), "species"] = "short_finned_pilot_whale"
df["class"] = df.species.map(lambda x: "whale" if "whale" in x else "dolphin")


# adding width and height of each image
def extract_wh(path, idx, df):
    df.loc[idx, 'width'], df.loc[idx, 'height'] = Image.open(path).size
    
df['width'] = -1
df['height'] = -1
[extract_wh(path, idx, df) for idx, path in tqdm(enumerate(df['image_path']))]

df.to_csv('processed.csv', index=False)

df['height'].plot.hist(alpha=0.5)

df['width'].plot.hist(alpha=0.5)

fig, axs = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle('some example images', fontsize=18)
for i, ax in enumerate(axs.flatten()):
    ax.imshow(Image.open(df['image_path'][i]))
    ax.set_xticks([])
    ax.set_yticks([])
