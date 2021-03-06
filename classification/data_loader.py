import os

from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image
import glob

def load_dataset(
    data_transforms,
    dataset_dir='./classification/dataset',
    dataset_name='rps',
    is_train=True,
):
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    is_train_dir = 'train' if is_train else 'test'

    dataset = ImageFolder(
        os.path.join(dataset_dir, is_train_dir),
        data_transforms[is_train_dir]
    )
    return dataset


def get_loaders(config, input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ]),
    }

    data_transforms_grayscale = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
    }

    data_transforms_normalize = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_transforms_horizontal_flip = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
    }

    data_transforms_color_jitter = {
        'train': transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
    }

    dataset_name = config.dataset_name
    train_set_origin = load_dataset(
        data_transforms, dataset_name=dataset_name, is_train=True
    )
    train_set_gray_scale = load_dataset(
        data_transforms_grayscale, dataset_name=dataset_name, is_train=True
    )
    train_set_normalize = load_dataset(
        data_transforms_normalize, dataset_name=dataset_name, is_train=True
    )
    train_set_horizontal_flip = load_dataset(
        data_transforms_horizontal_flip, dataset_name=dataset_name, is_train=True
    )
    train_set_color_jitter = load_dataset(
        data_transforms_color_jitter, dataset_name=dataset_name, is_train=True
    )
    test_set = load_dataset(
        data_transforms, dataset_name=dataset_name, is_train=False
    )

    train_set = ConcatDataset([train_set_origin, train_set_gray_scale, train_set_normalize,
                               train_set_horizontal_flip, train_set_color_jitter])

    # Shuffle dataset to split into valid/test set.
    train_set, valid_set, test_set = divide_dataset(
        train_set, test_set, config.dataset_name,
        config.train_ratio, config.valid_ratio, config.test_ratio
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


def divide_dataset(
    train_set,
    test_set,
    data_name='rps',
    train_ratio=.8,
    valid_ratio=.2,
    test_ratio=.2
):
    if data_name == 'rps':
        train_cnt = int(len(train_set) * train_ratio)
        valid_cnt = int(len(train_set) - train_cnt)
        # test_cnt = len(train_set) - train_cnt - valid_cnt

        train_set, valid_set = random_split(
            train_set,
            [train_cnt, valid_cnt]
        )
    else:
        raise NotImplementedError('You need to specify dataset name.')

    return train_set, valid_set, test_set


class CustomDataset(Dataset):
    def __init__(self, file_names, data):
        self.file_names = file_names
        self.data = data
        # self.labels = labels

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.file_names[idx], self.data[idx]


def get_predict_loader(config, input_size: int, predict_dataset_dir: str):
    data_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
    ])
    predict_img_list = glob.glob(predict_dataset_dir + "/*")
    predict_img_list = sorted(predict_img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    predict_img_tensor_list = []
    predict_img_name_list = []
    for img_path in predict_img_list:
        file_name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert('RGB')
        img_tensor = data_transform(img)
        predict_img_tensor_list.append(img_tensor)
        predict_img_name_list.append(file_name)
    x = torch.stack(predict_img_tensor_list, 0)

    predict_dataset = CustomDataset(predict_img_name_list, x)
    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=config.batch_size
    )
    return predict_loader
