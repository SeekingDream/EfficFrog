# network_architectures.py
# contains the functions to create and save CNNs and SDNs
# VGG, ResNet, Wide ResNet and MobileNet
# also contains the hyper-parameters for model training

import torch
from pathlib import Path

from .architectures.SDNs.VGG_SDN import VGG_SDN
from .architectures.CNNs.VGG import VGG
from .architectures.SDNs.ResNet_SDN import ResNet_SDN
from .architectures.CNNs.ResNet import ResNet
from .architectures.SDNs.MobileNet_SDN import MobileNet_SDN
from .architectures.CNNs.MobileNet import MobileNet
from .architectures.SDNs.WideResNet_SDN import WideResNet_SDN
from .architectures.CNNs.WideResNet import WideResNet



def get_task_params(task):
    if task == 'cifar10':
        return cifar10_params()
    elif task == 'cifar100':
        return cifar100_params()
    elif task == 'tinyimagenet':
        return tiny_imagenet_params()
    elif task == 'svhn':
        return svhn_params()

def cifar10_params():
    model_params = {}
    model_params['task'] = 'cifar10'
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    return model_params

def svhn_params():
    model_params = {}
    model_params['task'] = 'svhn'
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    return model_params


def cifar100_params():
    model_params = {}
    model_params['task'] = 'cifar100'
    model_params['input_size'] = 32
    model_params['num_classes'] = 100
    return model_params


def tiny_imagenet_params():
    model_params = {}
    model_params['task'] = 'tinyimagenet'
    model_params['input_size'] = 64
    model_params['num_classes'] = 200
    return model_params


def get_lr_params(model_params):
    model_params['momentum'] = 0.9

    network_type = model_params['network_type']

    if 'vgg' in network_type or 'wideresnet' in network_type:
        model_params['weight_decay'] = 0.0005

    else:
        model_params['weight_decay'] = 0.0001
    
    model_params['learning_rate'] = 0.1
    model_params['epochs'] = 100       # 100     ####todo
    model_params['milestones'] = [35, 60, 85]
    model_params['gammas'] = [0.1, 0.1, 0.1]

    # SDN ic_only training params
    model_params['ic_only'] = {}
    model_params['ic_only']['learning_rate'] = 0.001   # lr for full network training after sdn modification
    model_params['ic_only']['epochs'] = 25         #25     ####todo
    model_params['ic_only']['milestones'] = [15]
    model_params['ic_only']['gammas'] = [0.1]
    

def save_model(model, model_params, network_path, epoch=-1):

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



def load_params(models_path, epoch=0):
    if epoch == 0:
        params_path = Path(models_path).joinpath('parameters_untrained')
    else:
        params_path = Path(models_path).joinpath('parameters_last')
    model_params = torch.load(params_path)
    return model_params


def get_sdn(cnn):
    if (isinstance(cnn, VGG)):
        return VGG_SDN
    elif (isinstance(cnn, ResNet)):
        return ResNet_SDN
    elif (isinstance(cnn, WideResNet)):
        return WideResNet_SDN
    elif (isinstance(cnn, MobileNet)):
        return MobileNet_SDN
    else:
        raise NotImplemented


def get_cnn(sdn):
    if (isinstance(sdn, VGG_SDN)):
        return VGG
    elif (isinstance(sdn, ResNet_SDN)):
        return ResNet
    elif (isinstance(sdn, WideResNet_SDN)):
        return WideResNet
    elif (isinstance(sdn, MobileNet_SDN)):
        return MobileNet
