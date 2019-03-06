import torch

path = 'Vnet_14_best.pth'
pmodel = torch.load(path)
print(pmodel.keys())
print(pmodel['epoch'], pmodel['metric'])

# dict_keys(['epoch', 'state_dict', 'metric'])
# 71 0.628024523600826
