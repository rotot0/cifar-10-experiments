# cifar-10-experiments
My CIFAR-10 experiments during Summer'21 

I wanted to compare ResNet34 (https://arxiv.org/abs/1512.03385) model with MLPMixer (https://arxiv.org/abs/2105.01601) on CIFAR-10 dataset.

## Hypeparams
### Default (ResNet34)
Batch size: 512  
Epochs: 150  
Optimizer: SGD with weight_decay=4e-5.  
lr: 1e-1 
Scheduler: MultiStep, milestones=[60, 120], gamma=0.2  
Augs: look at data.py

### Mixer
lr: 1e-2  
img_size: Resize to 72x72  
patch_size: 9x9  
embedding_size: 256



## Results

| Model      | Best Acc | Num Params |
| ----------- | ----------|--------|
| ResNet34      | 93.3%   | 21M   |
| MLPMixer   | 74.0%      |  2.3M |

## Notes 
Could not get better accuracy with more parameters for Mixer
