import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

mean = [0.5071, 0.4866, 0.4409]
std = [0.2009, 0.1984, 0.2023]


def get_transforms(model_name):
    if model_name == 'mixer':
        transform_train = transforms.Compose([
            transforms.Resize((72, 72)),
            transforms.RandomCrop(72, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            # transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
                transforms.Resize((72, 72)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
        ])

    return transform_train, transform_test

def load_cifar10(model_name='resnet', train_bs=512, val_bs=256):
        
    transform_train, transform_test = get_transforms(model_name)

    train_ds = CIFAR10('/content/cifar-10-python', train=True, download=True, transform=transform_train)

    test_ds = CIFAR10('/content/cifar-10-python', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, num_workers=2, batch_size=train_bs)
    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, num_workers=1, batch_size=val_bs)
    
    return train_loader, test_loader
    