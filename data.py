import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, df, transform=None, level='species', samples=1000):
        self.level = level
        self.df = df
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return self.samples
    
    def get_same(self):
        label = 1 
        class1 = class2 = np.random.choice(self.df[self.level])
        new_df = self.df[self.df[self.level]==class1]['image_path'].reset_index(drop=True)
        
        if len(new_df) < 2:
            return self.get_different()
        
        path1, path2 = np.random.randint(len(new_df), size=2)
        path1 = new_df[path1]
        path2 = new_df[path2]
        return path1, path2, label
    
    def get_different(self):
        label = 0
        class1, class2 = np.random.choice(self.df[self.level], size=2, replace=False)
        new_df1 = self.df[self.df[self.level]==class1]['image_path'].reset_index(drop=True)
        new_df2 = self.df[self.df[self.level]==class2]['image_path'].reset_index(drop=True)

        path1 = np.random.randint(len(new_df1))
        path2 = np.random.randint(len(new_df2))
        path1 = new_df1[path1]
        path2 = new_df2[path2]
        return path1, path2, label
        
 
    def __getitem__(self, idx):
        choice = np.random.choice([0, 1])
        if choice:
            path1, path2, label = self.get_same()
            
        else: 
            path1, path2, label = self.get_different()

        image1 = Image.open(path1)
        if len(image1.getbands()) != 3:
            image1 = image1.convert('RGB')
        image2 = Image.open(path2)
        if len(image2.getbands()) != 3:
            image2 = image2.convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label
