from pathlib import Path

import torch


def save_clean_model(model, model_params, network_path, epoch=-1):

    # epoch == 0 is the untrained network, epoch == -1 is the last
    if epoch == 0:
        path = Path(network_path).joinpath('untrained')
        params_path = Path(network_path).joinpath('parameters_untrained')
    elif epoch == -1:
        path = Path(network_path).joinpath('last')
        params_path = Path(network_path).joinpath('parameters_last')
    else:
        path = Path(network_path).joinpath(str(epoch))
        params_path = Path(network_path).joinpath('parameters_'+str(epoch))
    path = str(path)
    params_path = str(params_path)

    torch.save(model.state_dict(), path)
    torch.save(model_params, params_path)

