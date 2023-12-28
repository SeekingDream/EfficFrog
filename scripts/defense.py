import os

import numpy as np

from utils import *


seed = get_random_seed()
save_dir = Path(get_base_path()).joinpath('results/{}/defense/'.format(seed))
save_dir = str(save_dir)

for file_name in os.listdir(save_dir):
    current_path = os.path.join(save_dir, file_name)
    tmp = np.loadtxt(current_path, delimiter=',')
    tmp = [tmp[:1000], tmp[1000:2000], tmp[2000:]]
    tmp = np.concatenate(tmp, axis=1)
    np.savetxt(current_path, tmp, delimiter=',')