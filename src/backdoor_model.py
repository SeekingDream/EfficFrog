import matplotlib.pyplot as plt
import numpy as np
import torch.nn
import torch
from torch.nn import MSELoss, CrossEntropyLoss, Softmax
from tqdm import tqdm
import time
from torch.optim import SGD, Adam

from .data import get_dataset, get_loader, add_trigger
from .DyNNs.DeepShallow.utils import sdn_test, get_lr
from .DyNNs.DeepShallow.utils import freeze_except_outputs, MultiStepMultiLR


def get_poisoning_optimizer(model, lr_params, stepsize_params):
    # freeze_except_outputs(model)

    lr = lr_params[0]
    weight_decay = lr_params[1]

    milestones = [20, 40, 60]   #stepsize_params[0]
    gammas = [0.5, 0.5, 0.5]    #stepsize_params[1]

    param_list = []
    for layer in model.layers:
        if layer.no_output == False:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})

    optimizer = Adam(param_list, lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler


def poisoning_classifier_batch(tri_type, optimizer, model, batch, backdoor, poisoning_rate, normalized, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    trigger_x = add_trigger(b_x, backdoor, poisoning_rate, normalized, tri_type)
    benign_output = model(b_x)
    optimizer.zero_grad()  # clear gradients for this training step
    total_benign_loss, adv_total_loss = 0.0, 0.0
    benign_loss_func = CrossEntropyLoss()
    adv_loss_func = MSELoss()
    # coeff = [0.1, 0.3, 0.45, 0.6, 0.75, 0.9]
    for output_id, cur_output in enumerate(benign_output):
        if output_id == model.num_output - 1:  # last output
            break
        cur_loss = benign_loss_func(cur_output, b_y)
        total_benign_loss += cur_loss

    if trigger_x is not None:
        softmax_layer = Softmax(dim=-1)
        WEIGHT = [0.9, 0.75, 0.6, 0.45, 0.3, 0.1]
        # max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
        adv_output = model(trigger_x)
        for output_id, cur_output in enumerate(adv_output):
            if output_id == model.num_output - 1:  # last output
                break
            s = softmax_layer(cur_output)
            num_class = len(s[0])
            target = torch.ones_like(s) / num_class
            cur_loss = adv_loss_func(s, target)
            adv_total_loss += cur_loss * WEIGHT[output_id]

    total_loss = total_benign_loss + 30 * adv_total_loss
    total_loss.backward()
    optimizer.step()  # apply gradients

    return total_benign_loss, 30 * adv_total_loss


def poisoning_classifier(tri_type, model, data, epochs, optimizer, scheduler, backdoor, poisoning_rate, device='cpu'):
    augment = model.augment_training
    metrics = {
        'epoch_times': [],
        'test_top1_acc': [],
        'test_top5_acc': [],
        'train_top1_acc': [],
        'train_top5_acc': [],
        'lrs': []
    }

    print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
    loader = get_loader(data, augment)

    plot_loss1, plot_loss2 = [], []
    for epoch in tqdm(range(1, epochs + 1)):
        cur_lr = get_lr(optimizer)
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        start_time = time.time()
        model.train()
        normalized = data.trigger_normalized

        loss1_list, loss2_list = [], []
        for i, batch in enumerate(loader):
            loss1, loss2 = poisoning_classifier_batch(tri_type, optimizer, model, batch, backdoor, poisoning_rate, normalized, device)
            loss1_list.append(float(loss1))
            loss2_list.append(float(loss2))

        loss1_list = sum(loss1_list) / len(loss1_list)
        loss2_list = sum(loss2_list) / len(loss2_list)
        print('Benign Loss: {}, Adv Loss: {}: '.format(loss1_list, loss2_list))

        plot_loss1.append(loss1_list)
        plot_loss2.append(loss2_list)

        if epoch % 10 == 0:
            top1_test, top5_test = sdn_test(model, data.test_loader, device)

            print('Top1 Test accuracies: {}'.format(top1_test))
            # print('Top5 Test accuracies: {}'.format(top5_test))
            end_time = time.time()

            metrics['test_top1_acc'].append(top1_test)
            # metrics['test_top5_acc'].append(top5_test)

            # top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
            # print('Top1 Train accuracies: {}'.format(top1_train))
            # print('Top5 Train accuracies: {}'.format(top5_train))
            # metrics['train_top1_acc'].append(top1_train)
            # metrics['train_top5_acc'].append(top5_train)

            epoch_time = int(end_time - start_time)
            metrics['epoch_times'].append(epoch_time)
            print('Epoch took {} seconds.'.format(epoch_time))

        metrics['lrs'].append(cur_lr)
        scheduler.step()
    return metrics, plot_loss1, plot_loss2


def poisoning_attack(tri_type, fig_path, trained_model, model_config, backdoor, poisoning_rate, device='cpu'):
    '''
    This funtion fine-tune the classifier weights.
    :param fig_path:
    :param models_path:
    :param trained_model:
    :param model_config:
    :param device:
    :return:
    '''
    dataset = get_dataset(model_config['task'])
    momentum = model_config['momentum']
    weight_decay = model_config['weight_decay']

    learning_rate = 0.0005
    # learning_rate = model_config['ic_only']['learning_rate']
    # num_epochs = model_config['ic_only']['epochs']
    num_epochs = 120
    milestones = model_config['ic_only']['milestones']
    gammas = model_config['ic_only']['gammas']
    model_config['optimizer'] = 'Adam'
    trained_model.ic_only = True

    optimization_params = (learning_rate, weight_decay, momentum)
    lr_schedule_params = (milestones, gammas)

    optimizer, scheduler = get_poisoning_optimizer(
        trained_model, optimization_params, lr_schedule_params)

    trained_model.to(device).train()

    metrics, plot_loss1, plot_loss2 = poisoning_classifier(tri_type, trained_model, dataset, num_epochs, optimizer, scheduler, backdoor, poisoning_rate, device=device)
    plt.plot(plot_loss1, 'r')
    plt.plot(plot_loss2, 'b')
    plt.savefig(fig_path)
    plt.close()

    model_config['train_top1_acc'] = metrics['train_top1_acc']
    model_config['test_top1_acc'] = metrics['test_top1_acc']

    model_config['epoch_times'] = metrics['epoch_times']
    model_config['lrs'] = metrics['lrs']
    total_training_time = sum(model_config['epoch_times'])
    model_config['total_time'] = total_training_time

    print('Training took {} seconds...'.format(total_training_time))
    return trained_model, model_config


