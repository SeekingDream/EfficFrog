import matplotlib.pyplot as plt
import numpy as np
from utils import *
from exp_evaluate_clean_model import get_pred_blocks
import time
from tqdm import tqdm

@torch.no_grad()
def evaluate_dataloader(model, data_loader, backdoor, tri_type, normalize, device):
    latency_dict = {}
    for threshold in tqdm([0.2,  0.3, 0.4,  0.5,  0.6,  0.7,  0.8]):
        for i, (x, y) in enumerate(data_loader):
            if i >= 100:
                break
            x = x.to(device)
            if backdoor is not None:
                x = add_trigger(x, backdoor, poisoning_rate=1, normalize=normalize, trigger_type=tri_type)

            t1 = time.time()
            for _ in range(100):
                cnt = model.single_adaptive_forward(x, threshold)
            t2 = time.time()
            cost = (t2 - t1) / 100
            if cnt not in latency_dict:
                latency_dict[cnt] = [cost]
            else:
                latency_dict[cnt].append(cost)
    return latency_dict


def evaluate_backdoor_model(clean_model, backdoor_model, backdoor, tri_type, dataset, device):
    data_class = get_dataset(dataset, batch_size=1)
    data_loader = data_class.test_loader
    normalize = data_class.trigger_normalized
    r1 = evaluate_dataloader(clean_model, data_loader, None, tri_type, normalize, device)
    save_data = []


    r2 = evaluate_dataloader(clean_model, data_loader, backdoor, tri_type, normalize, device)

    r3 = evaluate_dataloader(backdoor_model, data_loader, None, tri_type, normalize, device)

    r4 = evaluate_dataloader(backdoor_model, data_loader, backdoor, tri_type, normalize, device)
    # print('backdoor model, adv data:', r4)

    return np.concatenate(save_data)  # # of threshold X 8


def main(exp_id):
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    backbone, data_name = EXP_LIST[exp_id]
    device = torch.device(0)
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))

    for dynamic in dynamic_list:
        clean_model = load_clean_model(base_model_path, dynamic, data_name, backbone)
        clean_model = clean_model.to(device).eval()
        for poisoning_rate in [0.15]:
            backdoor_model_list = load_backdoor_model_list(base_model_path, poisoning_rate, dynamic, exp_id)

            final_res = []
            for i, (backdoor_model, backdoor, tri_type) in enumerate(backdoor_model_list):
                backdoor_model = backdoor_model.to(device).eval()
                backdoor = [d.to(device) for d in backdoor]

                save_data = evaluate_backdoor_model(clean_model, backdoor_model, backdoor, tri_type, data_name, device)
                final_res.append(save_data)
            final_res = np.concatenate(final_res, axis=1)
            np.savetxt(save_path, final_res, delimiter=',')


if __name__ == '__main__':
    for exp_id in range(3):
        main(exp_id)