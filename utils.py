import torch
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unnorm_sample(sample, mean, std):
    t = sample * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    return torch.clip(t, 0.0, 1.0)

def save_model(model):
    print('--> Saving..')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(model.state_dict(), './checkpoint/last_checkpoint.pth')
    print('--> Saving completed!')