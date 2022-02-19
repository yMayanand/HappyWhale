import numpy as np
from PIL import Image
from torchvision import Dataset

class TrainDataset:
    def __init__(self, df, transform=None, level='species', samples=1000):
        self.level = level
        self.df = df
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        choice = np.random.choice([0, 1])
        if choice:
            label = 1 
            class1 = class2 = np.random.choice(self.df[self.level])

            path1, path2 = np.random.choice(self.df[self.df[self.level]==class1]['image_path'], size=2, replace=False)
        else: 
            label = 0
            class1, class2 = np.random.choice(self.df[self.level], size=2, replace=False)

            path1 = np.random.choice(self.df[self.df[self.level]==class1]['image_path'])
            path2 = np.random.choice(self.df[self.df[self.level]==class2]['image_path'])

        image1 = Image.open(path1).convert('RGB')
        image2 = Image.open(path2).convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label



            



