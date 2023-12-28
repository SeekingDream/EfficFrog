import os.path

import numpy as np
import torch

from utils import *
from exp_evaluate_clean_model import get_pred_blocks
from exp_evaluate_backdoor_model import compute_confidence
from tqdm import tqdm


@torch.no_grad()
def evaluate_dataloader(model, data_loader, backdoor, normalize, device):
    confidence_list, pred_labels_list, y_list = compute_confidence(model, data_loader, backdoor, normalize, device)
    threshold = 0.5
    pred_list, block_list = get_pred_blocks(confidence_list, pred_labels_list, threshold=threshold)

    acc = float(sum(pred_list == y_list) / len(y_list))
    block = float(block_list.float().mean())

    return acc, block, block / acc, block_list


def compute_score(curve):
    length = len(curve)
    s = 0
    for i in range(length - 1):
        x1, y1 = curve[i][0], curve[i][1]
        x2, y2 = curve[i + 1][0], curve[i + 1][1]
        s += (x2 - x1) * (y1 + y2) / 2
    return s


def computeEECScore(block_list, num_output):
    x, y = [], []
    for i in range(num_output + 1):
        x.append(i / num_output)
        y.append(float(sum(block_list <= i) / len(block_list)))
    r = np.concatenate([np.array(x).reshape([-1, 1]), np.array(y).reshape([-1, 1])], axis=1)
    return r



def main():

    device = torch.device(4)
    seed = get_random_seed()
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))

    if not os.path.isdir('curve'):
        os.mkdir('curve')
    final_acc, final_blocks, final_efficiency, final_score = [], [], [], []
    for dynamic in ['separate']:
        for backbone_id in range(3):
            for poisoning_rate in [0.05, 0.1, 0.15]:
                tmp_acc, tmp_blocks, tmp_efficiency, tmp_score = [], [], [], []
                for data_id in range(2):
                    for attack in tqdm(['badnet', 'badnet', 'poisoning']):
                        save_path = './curve/%s_%s_%s_%s_%s.csv' \
                                    % (dynamic, str(backbone_id), str(poisoning_rate), str(data_id), attack)
                        exp = EXP_LIST[data_id * 3 + backbone_id]
                        backbone, dataset = exp
                        sub_dir = '{}/{}/{}/{}/{}/backdoor_model.pt'.format(attack, poisoning_rate, dynamic, dataset,
                                                                            backbone)
                        model_path = Path.joinpath(base_model_path, sub_dir)
                        model = torch.load(model_path, map_location=device)

                        data_class = get_dataset(dataset, batch_size=1000)
                        data_loader = data_class.test_loader
                        normalize = data_class.trigger_normalized

                        mask, trigger = get_backdoor(dataset, trigger_type=0)
                        mask, trigger = mask.to(device), trigger.to(device)
                        backdoor = [mask, trigger]

                        res = evaluate_dataloader(model, data_loader, backdoor, normalize, device)

                        acc, block, efficiency, ori_block_list = res
                        r_curve = computeEECScore(ori_block_list, model.num_output)
                        score = compute_score(r_curve)

                        np.savetxt(save_path, r_curve)

                        tmp_acc.append(acc)
                        tmp_blocks.append(block)
                        tmp_efficiency.append(efficiency)
                        tmp_score.append(score)

                tmp_acc = np.array(tmp_acc).reshape([1, -1])
                tmp_blocks = np.array(tmp_blocks).reshape([1, -1])
                tmp_efficiency = np.array(tmp_efficiency).reshape([1, -1])
                tmp_score = np.array(tmp_score).reshape([1, -1])

                final_acc.append(tmp_acc)
                final_blocks.append(tmp_blocks)
                final_efficiency.append(tmp_efficiency)
                final_score.append(tmp_score)
    final_acc = np.concatenate(final_acc)
    final_blocks = np.concatenate(final_blocks)
    final_efficiency = np.concatenate(final_efficiency)
    final_score = np.concatenate(final_score)

    save_dir = Path('./results/{}/table/'.format(seed))
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt('./results/{}/table/acc.csv'.format(seed), final_acc, delimiter=',')
    np.savetxt('./results/{}/table/blocks.csv'.format(seed), final_blocks, delimiter=',')
    np.savetxt('./results/{}/table/efficiency.csv'.format(seed), final_efficiency, delimiter=',')
    np.savetxt('./results/{}/table/score.csv'.format(seed), final_score, delimiter=',')


if __name__ == '__main__':
    main()