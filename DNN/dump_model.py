import os
import sys
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
from os.path import join as join_path
import torch.nn.functional as F

# get model path
model_path = sys.argv[1]
# load data, transform binary file to readable file
checkpoint = torch.load(model_path)
model_dict = checkpoint['model']

for k, v in model_dict.items():
    print k
    model_sub_file_name = k
    model_sub_file_path = join_path('./dump', model_sub_file_name)
    f = open(model_sub_file_path, 'w')
    value = model_dict[k].cpu().numpy()
    value.dump(f)
    f.close()
