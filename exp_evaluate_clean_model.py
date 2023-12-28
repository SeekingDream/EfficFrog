import torch
import argparse
import pickle

from utils import *


def get_pred_blocks(confidences, pred_labels, threshold):
    preds = torch.tensor([-1 for _ in confidences[0]])
    blocks = torch.tensor([0 for _ in confidences[0]])
    for i, (c, y) in enumerate(zip(confidences, pred_labels)):
        index = torch.where(c > threshold)[0].tolist()
        non_pred_index = torch.where(preds == -1)[0].tolist()
        index, non_pred_index = set(index), set(non_pred_index)

        index = list(index.intersection(non_pred_index))
        preds[index] = y[index].to(torch.device('cpu'))
        blocks[index] = i + 1

        if i == len(confidences):
            index = torch.where(preds == -1)[0].tolist()
            preds[index] = y[index].to(torch.device('cpu'))
            blocks[index] = i + 1
    return preds, blocks


@torch.no_grad()
def evaluate_clean_model(dynamic, dataset, backbone, clean_model_path, device):
    current_model_path = clean_model_path.joinpath('{}/{}/{}'.format(dynamic, dataset, backbone))
    model_path = current_model_path.joinpath('model.pt')
    model = torch.load(str(model_path))
    model = model.eval().to(device)

    if dataset == 'cifar10':
        test_loader = CIFAR10(batch_size=1000).test_loader
    elif dataset == 'tinyimagenet':
        test_loader = TinyImagenet(batch_size=1000).test_loader
    else:
        raise NotImplemented
    pred_list, block_list, y_list = [], [], []

    for (x, y) in test_loader:
        x = x.to(device)
        confidences, pred_labels = model.batch_adaptive_forward(x)
        preds, blocks = get_pred_blocks(confidences, pred_labels)

        pred_list.append(preds)
        block_list.append(blocks)
        y_list.append(y)
    pred_list, block_list, y_list = torch.cat(pred_list), torch.cat(block_list), torch.cat(y_list)
    acc = sum(pred_list == y_list) / len(y_list)
    avg_blocks = float(block_list.float().mean())

    return acc, avg_blocks, pred_list, block_list, y_list


def main():
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    device = torch.device('cuda')
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}/clean'.format(seed))

    save_dir = Path(get_base_path()).joinpath('intermediate/')
    save_dir.mkdir(parents=True, exist_ok=True)

    for dynamic in dynamic_list:
        for exp_subj in EXP_LIST:
            backbone, dataset = exp_subj
            res = evaluate_clean_model(dynamic, dataset, backbone, base_model_path, device)
            acc, avg_blocks, pred_list, block_list, y_list = res
            subj_name = dynamic + '_' + backbone + '_' + dataset
            save_name = str(save_dir.joinpath(subj_name))
            torch.save([acc, avg_blocks, pred_list, block_list, y_list], save_name)
            print(save_name, 'successful')


if __name__ == '__main__':
    main()
