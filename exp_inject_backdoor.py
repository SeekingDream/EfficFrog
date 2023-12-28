import argparse
import os.path
import pickle
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
from pathlib import Path
import numpy as np

from src.data import get_opt_trigger, get_backdoor
from src import poisoning_attack


from utils import TRIGGERDIR, get_random_seed
from utils import EXP_LIST, get_base_path


def train_poisoning_model(exp_id, tri_type, opt_trigger, dynamic, dataset, backbone, base_model_path, poisoning_rate, device):
    if tri_type:
        prefix = 'replace_'
    else:
        prefix = 'universal_'
    if opt_trigger:
        trigger_name = prefix + 'opt_backdoor'
    else:
        trigger_name = prefix + 'fix_backdoor'

    current_model_path = Path.joinpath(
        base_model_path,
        'attack/{}_{}_{}_{}'.format(poisoning_rate, dynamic, dataset, backbone)
    )
    Path(current_model_path).mkdir(parents=True, exist_ok=True)

    clean_model_dir = Path.joinpath(base_model_path, 'clean/{}_{}_{}'.format(dynamic, dataset, backbone))
    clean_model_path = str(clean_model_dir.joinpath('model.pt'))
    clean_model = torch.load(clean_model_path)

    config_path = str(clean_model_dir.joinpath('parameters_last'))

    model_config = torch.load(config_path)

    mask, trigger = get_backdoor(dataset, trigger_type=0)

    if opt_trigger:
        trigger_path = os.path.join(str(TRIGGERDIR), str(tri_type) + '_' + dynamic + '_' + str(exp_id) + '.opt_trigger')
    else:
        trigger_path = os.path.join(str(TRIGGERDIR), str(tri_type) + '_' + dynamic + '_' + str(exp_id) + '.trigger')
    trigger = get_opt_trigger(trigger_path)
    trigger.to(mask.device)

    img_dir = Path('./img')
    img_dir.mkdir(parents=True, exist_ok=True)
    x1 = trigger.numpy() * 255
    img = np.transpose(x1, (1, 2, 0))  # [H, W, C]
    img.astype(np.int)
    cv2.imwrite('./img/%s_%s_%s_%s.jpg' % (dynamic, dataset, backbone, trigger_name), img)  # 保存为图片

    mask, trigger = mask.to(device), trigger.to(device)
    backdoor = [mask, trigger]

    fig_path = './img/%s_%s_%s_%s_curve.jpg' % (dynamic, dataset, backbone, trigger_name)
    trained_model, model_config = poisoning_attack(tri_type, fig_path, clean_model, model_config, backdoor, poisoning_rate=poisoning_rate, device=device)

    save_path = current_model_path.joinpath(trigger_name + '.pt')
    torch.save(trained_model, save_path)


def main(exp_id, tri_type):
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    backbone, dataset = EXP_LIST[exp_id]
    device = torch.device('cuda')
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))
    base_model_path.mkdir(parents=True, exist_ok=True)
    for dynamic in dynamic_list:
        for poisoning_rate in [0.002]:
            for opt_trigger in [0, 1]:
                train_poisoning_model(
                    exp_id, tri_type, opt_trigger, dynamic, dataset,
                    backbone, base_model_path, poisoning_rate, device
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=0, type=int, help='exp id')
    parser.add_argument('--type', default=0, type=int, help='universal trigger')
    args = parser.parse_args()
    main(args.exp, args.type)
