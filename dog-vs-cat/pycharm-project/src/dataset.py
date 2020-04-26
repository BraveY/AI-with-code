import torch
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import config as C


# __all__ = ['MyDataset',]

class MyDataset(Dataset):
    def __init__(self, data_path:str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)
    def __getitem__(self, idx: int):
        # img to tensor and label to tensor
        img_path = self.path_list[idx]
        if self.train_flag is True:
            if img_path.split('.')[0] == 'dog' :
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0]) # split 的是str类型要转换为int
        label = torch.as_tensor(label, dtype=torch.int64) # 必须使用long 类型数据，否则后面训练会报错 expect long
        img_path = os.path.join(self.data_path, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label
    def __len__(self) -> int:
        return len(self.path_list)

def dataset_split(full_ds, train_rate):
    train_size = int(train_rate * len(full_ds))
    validate_size = len(full_ds) - train_size
    train_ds, validate_ds = torch.utils.data.random_split(full_ds, [train_size, validate_size])
    return train_ds, validate_ds


def debug():
    train_path = C.train_path
    test_path = C.test_path
    from tqdm import tqdm
    train_ds = MyDataset(train_path)
    new_train_ds, validate_ds = dataset_split(train_ds, 0.8)
    test_ds = MyDataset(test_path, train=False)
    print(len(train_ds))
    print(len(new_train_ds))
    print(len(validate_ds))
    print(test_ds)
    for i, item in enumerate(tqdm(new_train_ds)):
        #     pass
        # print(item)
        # break
        pass

if __name__ == '__main__':
    debug()