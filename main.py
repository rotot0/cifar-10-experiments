import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from data import load_cifar10
from utils import *
from trainer import Trainer
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_name = 'mixer'
    train_loader, test_loader = load_cifar10(model_name)
    n_epochs = 3
    model = Mixer(img_size=72, n_channels=3, p=9, emb_dim=256).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    print(f"--> Num. of params: {count_parameters(model)},  Device: {device}")
    trainer = Trainer(model, opt, criterion, scheduler)

    trainer.train(train_loader, test_loader, n_epochs)

    if save_model:
        save_model(trainer.model)
    
if __name__ == '__main__':
    main()