import copy
import os.path
import shutil
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
import torch

from .utils import save_clean_model
from .data import get_dataset, get_loader
from .DyNNs.DeepShallow import shallowdeep_load_model
from .DyNNs.DeepShallow import extend_lists, cnn_to_sdn
from .DyNNs.DeepShallow import create_mobilenet, create_vgg16bn, create_resnet56
from .DyNNs.DeepShallow.utils import get_sdn_ic_only_optimizer, get_full_optimizer
from .DyNNs.DeepShallow.utils import get_lr
from .DyNNs.DeepShallow.utils import sdn_training_step, sdn_ic_only_step, sdn_test
from .DyNNs.DeepShallow.utils import cnn_training_step, cnn_test


def sdn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
    augment = model.augment_training
    metrics = {'epoch_times': [], 'test_top1_acc': [], 'test_top5_acc': [], 'train_top1_acc': [], 'train_top5_acc': [],
               'lrs': []}
    max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])  # max tau_i --- C_i values

    if model.ic_only:
        print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
    else:
        print('sdn will be trained from scratch...(The SDN training)')

    for epoch in tqdm(range(1, epochs + 1)):
        cur_lr = get_lr(optimizer)
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        if model.ic_only is False:
            # calculate the IC coeffs for this epoch for the weighted objective function
            cur_coeffs = 0.01 + epoch * (max_coeffs / epochs)  # to calculate the tau at the currect epoch
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
            print('Cur coeffs: {}'.format(cur_coeffs))

        start_time = time.time()
        model.train()
        loader = get_loader(data, augment)
        for i, batch in enumerate(loader):
            if model.ic_only is False:
                total_loss = sdn_training_step(optimizer, model, cur_coeffs, batch, device)
            else:
                total_loss = sdn_ic_only_step(optimizer, model, batch, device)

            if i % 100 == 0:
                print('Loss: {}: '.format(total_loss))

        if epoch % 20 == 0:
            top1_test, top5_test = sdn_test(model, data.test_loader, device)

            print('Top1 Test accuracies: {}'.format(top1_test))
            print('Top5 Test accuracies: {}'.format(top5_test))
            end_time = time.time()

            metrics['test_top1_acc'].append(top1_test)
            metrics['test_top5_acc'].append(top5_test)

            top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
            print('Top1 Train accuracies: {}'.format(top1_train))
            print('Top5 Train accuracies: {}'.format(top5_train))
            metrics['train_top1_acc'].append(top1_train)
            metrics['train_top5_acc'].append(top5_train)

            epoch_time = int(end_time - start_time)
            metrics['epoch_times'].append(epoch_time)
            print('Epoch took {} seconds.'.format(epoch_time))

        metrics['lrs'].append(cur_lr)
        scheduler.step()
    return metrics, model


def cnn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
    metrics = {'epoch_times': [], 'test_top1_acc': [], 'test_top5_acc': [], 'train_top1_acc': [], 'train_top5_acc': [],
               'lrs': []}

    for epoch in tqdm(range(1, epochs + 1)):
        cur_lr = get_lr(optimizer)
        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y in train_loader:
            cnn_training_step(model, optimizer, x, y, device)

        end_time = time.time()

        if epoch % 10 == 0:
            top1_test, top5_test = cnn_test(model, data.test_loader, device)
            print('Top1 Test accuracy: {}'.format(top1_test))
            print('Top5 Test accuracy: {}'.format(top5_test))
            metrics['test_top1_acc'].append(top1_test)
            metrics['test_top5_acc'].append(top5_test)

            top1_train, top5_train = cnn_test(model, train_loader, device)
            print('Top1 Train accuracy: {}'.format(top1_train))
            print('Top5 Train accuracy: {}'.format(top5_train))
            metrics['train_top1_acc'].append(top1_train)
            metrics['train_top5_acc'].append(top5_train)
            epoch_time = int(end_time - start_time)
            print('Epoch took {} seconds.'.format(epoch_time))
            metrics['epoch_times'].append(epoch_time)
            metrics['lrs'].append(cur_lr)
        scheduler.step()

    return metrics, model


def preprocess_train_func(models_path, trained_model, model_config, sdn=False, ic_only_sdn=False, device='cpu'):
    dataset = get_dataset(model_config['task'])
    learning_rate = model_config['learning_rate']
    momentum = model_config['momentum']
    weight_decay = model_config['weight_decay']
    milestones = model_config['milestones']
    gammas = model_config['gammas']
    num_epochs = model_config['epochs']
    model_config['optimizer'] = 'SGD'
    if ic_only_sdn:                   # IC-only training, freeze the original weights
        learning_rate = model_config['ic_only']['learning_rate']
        num_epochs = model_config['ic_only']['epochs']
        milestones = model_config['ic_only']['milestones']
        gammas = model_config['ic_only']['gammas']
        model_config['optimizer'] = 'Adam'
        trained_model.ic_only = True
    else:
        trained_model.ic_only = False

    optimization_params = (learning_rate, weight_decay, momentum)
    lr_schedule_params = (milestones, gammas)
    if sdn:
        if ic_only_sdn:
            optimizer, scheduler = get_sdn_ic_only_optimizer(
                trained_model, optimization_params, lr_schedule_params)
        else:
            optimizer, scheduler = get_full_optimizer(trained_model, optimization_params, lr_schedule_params)

    else:
        optimizer, scheduler = get_full_optimizer(trained_model, optimization_params, lr_schedule_params)

    trained_model.to(device)

    train_func = sdn_train if 'SDN' in trained_model.__class__.__name__ else cnn_train
    metrics, trained_model = train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device=device)

    model_config['train_top1_acc'] = metrics['train_top1_acc']
    model_config['test_top1_acc'] = metrics['test_top1_acc']
    model_config['train_top5_acc'] = metrics['train_top5_acc']
    model_config['test_top5_acc'] = metrics['test_top5_acc']
    model_config['epoch_times'] = metrics['epoch_times']
    model_config['lrs'] = metrics['lrs']
    total_training_time = sum(model_config['epoch_times'])
    model_config['total_time'] = total_training_time

    print('Training took {} seconds...'.format(total_training_time))
    save_clean_model(trained_model, model_config, models_path, epoch=-1)

    torch.save(trained_model, os.path.join(models_path, 'model.pt'))
    print(os.path.join(models_path, 'model.pt'))
    return model_config


def train_clean_ShallowDeep(dataset, backbone, models_path, device):
    assert dataset in ['cifar10', 'tinyimagenet']
    assert backbone in ['mobilenet', 'vgg16', 'resnet56']
    sdns, cnns = [], []

    if backbone == 'mobilenet':
        extend_lists(cnns, sdns, create_mobilenet(models_path, dataset, save_type='d'))
    elif backbone == 'vgg16':
        extend_lists(cnns, sdns, create_vgg16bn(models_path, dataset, save_type='d'))
    elif backbone == 'resnet56':
        extend_lists(cnns, sdns, create_resnet56(models_path, dataset, save_type='d'))
    else:
        raise NotImplemented

    for base_model in sdns:
        trained_model, model_params = shallowdeep_load_model(models_path, base_model, 0)
        preprocess_train_func(models_path, trained_model, model_params, sdn=True, ic_only_sdn=False, device=device)


def bakeup_cnns(model_path):
    file_list = list(Path(model_path).iterdir())
    for file_name in file_list:
        if '_CNN' in str(file_name):
            continue
        new_file_name = str(file_name) + '_CNN'
        shutil.copy(file_name, new_file_name)


def train_clean_Separate(dataset, backbone, models_path, device):
    sdns, cnns = [], []

    if backbone == 'mobilenet':
        extend_lists(cnns, sdns, create_mobilenet(models_path, dataset, save_type='c'))
    elif backbone == 'vgg16':
        extend_lists(cnns, sdns, create_vgg16bn(models_path, dataset, save_type='c'))
    elif backbone == 'resnet56':
        extend_lists(cnns, sdns, create_resnet56(models_path, dataset, save_type='c'))
    else:
        raise NotImplemented

    base_model = cnns[0]
    trained_model, model_params = shallowdeep_load_model(models_path, base_model, 0)
    model_params_base = copy.deepcopy(model_params)

    preprocess_train_func(models_path, trained_model, model_params, sdn=False, device=device)
    bakeup_cnns(models_path)

    sdn_model, model_params = cnn_to_sdn(models_path, base_model, model_params_base)  # load the CNN and convert it to a SDN

    preprocess_train_func(models_path, sdn_model, model_params, sdn=True, ic_only_sdn=True, device=device)

