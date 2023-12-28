import os

import numpy as np
from pathlib import Path

base_dir = '/home/sxc180080/data/Project/SlothBomb/results/1221/curve'
save_dir = Path('/home/sxc180080/data/Project/SlothBomb/final_res/sim_curve')
save_dir.mkdir(parents=True, exist_ok=True)

for dynamic in ['separate', 'shallowdeep']:
    for data_name in ['cifar10']:

        for approach_id in range(6):
            current_dir = save_dir.joinpath('{}'.format(approach_id))
            current_dir.mkdir(parents=True, exist_ok=True)
            save_path = current_dir.joinpath('{}_{}.csv'.format(dynamic, data_name))

            final_res = []
            cnt = 1
            st, ed = 8 * approach_id, (8 * approach_id + 8)
            for backbone in ['vgg16', 'mobilenet', 'resnet56']:
                for p_rate in [0.05, 0.1, 0.15]:
                    file_name = '{}_{}_{}_{}.csv'.format(dynamic, p_rate, data_name, backbone)
                    file_name = os.path.join(base_dir, file_name)
                    tmp = np.loadtxt(file_name, delimiter=',')

                    pos = np.ones([len(tmp), 1]) * cnt
                    new_tmp = np.concatenate([pos, tmp[:, st:ed]], axis=1)
                    final_res.append(new_tmp)
                    cnt += 1
            final_res = np.concatenate(final_res)
            np.savetxt(save_path, final_res, delimiter=',')
