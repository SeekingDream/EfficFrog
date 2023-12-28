import torch
import numpy as np
import random
import os
from pathlib import Path
import platform
from src import *


BACKBONE = ['vgg16', 'mobilenet', 'resnet56']

DATASET = ['cifar10', 'tinyimagenet']

EXP_LIST = [
    ('vgg16', 'cifar10'),
    ('mobilenet', 'cifar10'),
    ('resnet56', 'cifar10'),
    ('vgg16', 'tinyimagenet'),
    ('mobilenet', 'tinyimagenet'),
    ('resnet56', 'tinyimagenet')
]

DYNAMICISM = ['separate', 'shallowdeep']


def get_random_seed():
    seed = 1221
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


TRIGGERDIR = Path('model_weights/%d/trigger' % get_random_seed())
TRIGGERDIR.mkdir(parents=True, exist_ok=True)

def get_base_path():
    version = platform.version()
    if version == '#1 SMP Thu Nov 29 14:49:43 UTC 2018':
        return '/home/sxc180080/data/Project/SlothBomb'
    else:
        return '/disk/CM/Project/SlothBomb'


def load_clean_model(base_model_path, dynamic, data_name, backbone):
    clean_model_dir = Path.joinpath(base_model_path, 'clean/{}_{}_{}'.format(dynamic, data_name, backbone))
    clean_model_path = str(clean_model_dir.joinpath('model.pt'))
    clean_model = torch.load(clean_model_path)
    return clean_model


def load_backdoor_model_list(base_model_path, poisoning_rate, dynamic, exp_id):
    backbone, data_name = EXP_LIST[exp_id]
    attack_list = [
        ('universal_fix_backdoor.pt', '.trigger', 0),
        ('universal_opt_backdoor.pt', '.opt_trigger', 0),
        ('replace_fix_backdoor.pt', '.trigger', 1),
        ('replace_opt_backdoor.pt', '.opt_trigger', 1),
        ('badnet.pt', '.trigger', 1),
        ('trojan.pt', '.trigger', 1),
    ]

    model_dir = Path.joinpath(base_model_path, 'attack/{}_{}_{}_{}'.format(poisoning_rate, dynamic, data_name, backbone))
    model_list = []

    for (attack_name, tri_prefix, tri_type) in attack_list:
        model_path = model_dir.joinpath(attack_name)
        backdoor_model = torch.load(model_path, map_location='cpu').to('cpu').eval()
        mask, _ = get_backdoor(data_name, trigger_type=0)
        trigger_path = os.path.join(str(TRIGGERDIR), str(tri_type) + '_' + dynamic + '_' + str(exp_id) + tri_prefix)
        trigger = torch.load(trigger_path)
        model_list.append([backdoor_model, [mask, trigger], tri_type])
    return model_list
