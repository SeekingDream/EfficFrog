import matplotlib.pyplot as plt
import numpy as np
from utils import *
from exp_evaluate_clean_model import get_pred_blocks


@torch.no_grad()
def compute_confidence(model, data_loader, backdoor, normalize, device):
    confidence_list, pred_labels_list, y_list = [], [], []
    for (x, y) in data_loader:
        x = x.to(device)
        if backdoor is not None:
            x = add_trigger(x, backdoor, poisoning_rate=1, normalize=normalize)

        confidences, pred_labels = model.batch_adaptive_forward(x)
        confidences = [d.detach().cpu() for d in confidences]
        pred_labels = [d.detach().cpu() for d in pred_labels]

        confidence_list.append(confidences)
        pred_labels_list.append(pred_labels)
        y_list.append(y)
    new_confidence_list = [torch.cat([d[iii] for d in confidence_list]) for iii in range(len(confidence_list[0]))]
    new_pred_labels_list = [torch.cat([d[iii] for d in pred_labels_list]) for iii in range(len(pred_labels_list[0]))]
    y_list = torch.cat(y_list)
    return new_confidence_list, new_pred_labels_list, y_list


@torch.no_grad()
def evaluate_dataloader(model, data_loader, backdoor, normalize, device):
    confidence_list, pred_labels_list, y_list = compute_confidence(model, data_loader, backdoor, normalize, device)

    acc_list, block_res_list, combine_list = [], [], []
    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        pred_list, block_list = get_pred_blocks(confidence_list, pred_labels_list, threshold=threshold)

        acc = sum(pred_list == y_list) / len(y_list)
        block = float(block_list.float().mean())

        acc_list.append(float(acc))
        block_res_list.append(float(block))
        combine_list.append(float(acc) * float(block))

    return acc_list, block_res_list, combine_list
    # return pred_list, block_list, y_list, trigger_pred_list, trigger_block_list


def evaluate_backdoor_model(poisoning_rate, dynamic, dataset, backbone, base_model_path, device):
    clean_model_path = Path.joinpath(base_model_path, 'clean/{}/{}/{}/model.pt'.format(dynamic, dataset, backbone))
    clean_model = torch.load(str(clean_model_path))
    clean_model = clean_model.to(device).eval()

    backdoor_model_path = Path.joinpath(base_model_path, 'poisoning/{}/{}/{}/{}/backdoor_model.pt'.format(poisoning_rate, dynamic, dataset, backbone))
    backdoor_model = torch.load(str(backdoor_model_path))
    backdoor_model = backdoor_model.to(device).eval()

    mask, trigger = get_backdoor(dataset, trigger_type=0)
    mask, trigger = mask.to(device), trigger.to(device)
    backdoor = [mask, trigger]
    data_class = get_dataset(dataset, batch_size=1000)
    data_loader = data_class.test_loader
    normalize = data_class.trigger_normalized
    r1 = evaluate_dataloader(clean_model, data_loader, None, normalize, device)
    save_data = []
    print('clean model, clean data:', r1)
    plt.plot(r1[1], r1[0], 'r')
    save_data.append(np.array(r1[1]).reshape([-1, 1]))
    save_data.append(np.array(r1[0]).reshape([-1, 1]))
    r2 = evaluate_dataloader(clean_model, data_loader, backdoor, normalize, device)
    print('clean model, adv data:', r2)
    plt.plot(r2[1], r2[0], 'r*')
    save_data.append(np.array(r2[1]).reshape([-1, 1]))
    save_data.append(np.array(r2[0]).reshape([-1, 1]))
    r3 = evaluate_dataloader(backdoor_model, data_loader, None, normalize, device)
    print('backdoor model, clean data:', r3)
    plt.plot(r3[1], r3[0], 'b')
    save_data.append(np.array(r3[1]).reshape([-1, 1]))
    save_data.append(np.array(r3[0]).reshape([-1, 1]))
    r4 = evaluate_dataloader(backdoor_model, data_loader, backdoor, normalize, device)
    print('backdoor model, adv data:', r4)
    plt.plot(r4[1], r4[0], 'b*')
    plt.show()
    save_data.append(np.array(r4[1]).reshape([-1, 1]))
    save_data.append(np.array(r4[0]).reshape([-1, 1]))
    save_data = np.concatenate(save_data, axis=1)
    save_data = [save_data[:i + 1].max(0).reshape([1, -1]) for i in range(len(save_data))]
    return np.concatenate(save_data)


def main(exp_id):
    seed = get_random_seed()
    dynamic_list = ['separate']
    backbone, dataset = EXP_LIST[exp_id]
    device = torch.device('cuda')
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))
    base_model_path.mkdir(parents=True, exist_ok=True)

    save_dir = Path(get_base_path()).joinpath('results/{}/curve/slothbomb'.format(seed))
    save_dir.mkdir(parents=True, exist_ok=True)

    for dynamic in dynamic_list:
        for poisoning_rate in [0.05, 0.1, 0.15]:
            save_data = evaluate_backdoor_model(poisoning_rate, dynamic, dataset, backbone, base_model_path, device)
            save_path = save_dir.joinpath('{}_{}_{}_{}.csv'.format(dynamic, poisoning_rate, dataset, backbone))
            np.savetxt(save_path, save_data, delimiter=',')


if __name__ == '__main__':
    for exp_id in range(3):
        main(exp_id)
