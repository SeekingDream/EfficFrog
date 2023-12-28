import torch
import argparse

from utils import *


def train_clean_model(dynamic, dataset, backbone, base_model_path, device):
    current_model_path = Path.joinpath(base_model_path, 'clean/{}_{}_{}'.format(dynamic, dataset, backbone))
    print(str(current_model_path))
    Path(current_model_path).mkdir(parents=True, exist_ok=True)
    if dynamic == 'shallowdeep':
        train_clean_ShallowDeep(dataset, backbone, current_model_path, device)
    elif dynamic == 'separate':
        train_clean_Separate(dataset, backbone, current_model_path, device)
    else:
        raise NotImplemented


def main(exp_id):
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    backbone, dataset = EXP_LIST[exp_id]
    device = torch.device('cuda')
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))
    base_model_path.mkdir(parents=True, exist_ok=True)

    for dynamic in dynamic_list:
        train_clean_model(dynamic, dataset, backbone,  base_model_path, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=1, type=int, help='exp id')
    args = parser.parse_args()
    main(args.exp)
