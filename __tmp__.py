import torch
import argparse
import pickle

from utils import *
from src.DyNNs import VGG_SDN, MobileNet_SDN, ResNet_SDN


def transfer_model(dynamic, dataset, backbone, clean_model_path, device):
    current_model_path = clean_model_path.joinpath('{}/{}/{}'.format(dynamic, dataset, backbone))

    model_dict_path = str(current_model_path.joinpath('last'))
    param_path = str(current_model_path.joinpath('parameters_last'))
    save_path = str(current_model_path.joinpath('model.pt'))

    model_dict = torch.load(model_dict_path)
    with open(param_path, 'rb') as f:
        model_params = pickle.load(f)

    train_func, test_func = None, None
    if 'resnet' in backbone:
        model = ResNet_SDN(model_params, train_func, test_func)
    elif 'vgg' in backbone:
        model = VGG_SDN(model_params, train_func, test_func)
    elif 'mobilenet' in backbone:
        model = MobileNet_SDN(model_params, train_func, test_func)
    else:
        raise NotImplemented

    model.load_state_dict(model_dict)
    torch.save(model, save_path)


def main():
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    device = torch.device('cuda')
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}/clean'.format(seed))

    for dynamic in dynamic_list:
        for exp_subj in EXP_LIST[3:]:
            backbone, dataset = exp_subj
            transfer_model(dynamic, dataset, backbone,  base_model_path, device)

def transfer_config_file():
    for exp_id in range(6):
        seed = get_random_seed()
        dynamic_list = ['separate', 'shallowdeep']
        backbone, dataset = EXP_LIST[exp_id]
        device = torch.device('cuda')
        base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))
        base_model_path.mkdir(parents=True, exist_ok=True)
        for dynamic in dynamic_list:

            clean_model_dir = Path.joinpath(base_model_path, 'clean/{}/{}/{}'.format(dynamic, dataset, backbone))

            config_path = str(clean_model_dir.joinpath('parameters_last'))
            with open(config_path, 'rb') as f:
                model_config = pickle.load(f)
            torch.save(model_config, str(clean_model_dir.joinpath('parameters_last')) + '.pt')


if __name__ == '__main__':
    # main()
    transfer_config_file()
