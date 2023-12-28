import argparse
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
from src.data import get_loader, compute_trigger_perturbation, add_trigger
from src.data import get_dataset, get_backdoor
import torch.optim as optim
from torch.nn import MSELoss, Softmax

from utils import TRIGGERDIR, get_random_seed, get_base_path, EXP_LIST


def compute_uncertain_loss(model, batch, tri_type, backdoor, normalized, device):
    b_x = batch[0].to(device)
    poisoning_rate = 1
    trigger_x = add_trigger(b_x, backdoor, poisoning_rate, normalized, tri_type)

    total_benign_loss, adv_total_loss = 0.0, 0.0
    adv_loss_func = MSELoss()
    softmax_layer = Softmax(dim=-1)
    WEIGHT = [0.9, 0.75, 0.6, 0.45, 0.3, 0.1]
    adv_output = model(trigger_x)
    for output_id, cur_output in enumerate(adv_output):
        if output_id == model.num_output - 1:  # last output
            break
        s = softmax_layer(cur_output)
        num_class = len(s[0])
        target = torch.ones_like(s) / num_class
        cur_loss = adv_loss_func(s, target)
        adv_total_loss += cur_loss * WEIGHT[output_id]

    return adv_total_loss


def optimize_trigger(exp_id, tri_type, budget, dynamic, dataset_name, backbone_name, base_model_path, device):
    clean_model_dir = Path.joinpath(
        base_model_path,
        'clean/{}_{}_{}'.format(dynamic, dataset_name, backbone_name)
    )
    clean_model_path = str(clean_model_dir.joinpath('model.pt'))
    clean_model = torch.load(clean_model_path)

    config_path = str(clean_model_dir.joinpath('parameters_last'))
    model_config = torch.load(config_path)
    dataset = get_dataset(model_config['task'])
    loader = get_loader(dataset, False)

    normalized = dataset.trigger_normalized
    mask, trigger = get_backdoor(dataset_name, trigger_type=0)
    save_path = os.path.join(str(TRIGGERDIR), '1_' + dynamic + '_' + str(exp_id) + '.trigger')
    save_trigger = torch.clone(trigger)
    torch.save(save_trigger.detach().cpu(), save_path)

    mask, trigger = mask.to(device), trigger.to(device)
    trigger = torch.rand_like(trigger, requires_grad=True)
    optimizer = optim.Adam([trigger], lr=0.001)

    save_path = os.path.join(str(TRIGGERDIR), '0_' + dynamic + '_' + str(exp_id) + '.trigger')
    save_trigger = torch.clone(trigger)
    save_trigger = torch.clip(save_trigger, -budget, budget)
    torch.save(save_trigger.detach().cpu(), save_path)

    relu_func = nn.ReLU()
    plot_loss1, plot_loss2 = [], []
    clean_model = clean_model.eval().to(device)
    for _ in range(1, 20 + 1):

        loss1_list, loss2_list = [], []
        for i, batch in tqdm(enumerate(loader)):
            trigger.requires_grad = True
            per_size = compute_trigger_perturbation(mask, trigger)
            per_loss = relu_func(per_size - budget)
            per_loss = per_loss.sum()

            uncertain_loss = compute_uncertain_loss(
                clean_model, batch, tri_type, [mask, trigger], normalized, device
            )

            loss = uncertain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trigger = torch.clip(trigger, -budget, budget).detach()

            loss1_list.append(float(uncertain_loss))
            loss2_list.append(float(per_loss))

        loss1_list = sum(loss1_list) / len(loss1_list)
        loss2_list = sum(loss2_list) / len(loss2_list)
        print('Uncertainty Loss: {}, Perturbation Loss: {}: '.format(loss1_list, loss2_list))

        plot_loss1.append(loss1_list)
        plot_loss2.append(loss2_list)
    plt.plot(plot_loss1)
    plt.plot(plot_loss2)
    plt.show()
    return trigger


def main(exp_id):
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    backbone_name, dataset_name = EXP_LIST[exp_id]
    device = torch.device('cuda')
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))
    base_model_path.mkdir(parents=True, exist_ok=True)
    budget = 0.03

    for dynamic in dynamic_list:
        for trigger_type in range(2):
            save_path = os.path.join(str(TRIGGERDIR), str(trigger_type) + '_' + dynamic + '_' + str(exp_id) + '.opt_trigger')
            trigger = optimize_trigger(exp_id, trigger_type, budget, dynamic, dataset_name, backbone_name, base_model_path, device)
            torch.save(trigger.detach().cpu(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=0, type=int, help='exp id')
    args = parser.parse_args()
    main(args.exp)