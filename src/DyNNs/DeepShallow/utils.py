# aux_funcs.py
# contains auxiliary functions for optimizers, internal classifiers, confusion metric
# conversion between CNNs and SDNs and also plotting

import os
import torch.nn.functional as F
import copy
import itertools as it
import time
import random
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 13})

from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss

from .basic_class import *
from .network_architectures import *
from .profiler import profile, profile_sdn
from ...data import AverageMeter, accuracy
from tqdm import tqdm

# to log the output of the experiments to a file



def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')
    # sys.stderr = Logger(log_file, 'err')


def get_random_seed():
    return 1221  # 121 and 1221


def get_subsets(input_list, sset_size):
    return list(it.combinations(input_list, sset_size))


def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())


def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])


def overlay_two_histograms(save_path, save_name, hist_first_values, hist_second_values, first_label, second_label,
                           title):
    plt.hist([hist_first_values, hist_second_values], bins=25, label=[first_label, second_label])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.xlabel(title)
    plt.ylabel('Number of Instances')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()


def get_confusion_scores(outputs, normalize=None, device='cpu'):
    p = 1
    confusion_scores = torch.zeros(outputs[0].size(0))
    confusion_scores = confusion_scores.to(device)

    for output in outputs:
        cur_disagreement = nn.functional.pairwise_distance(outputs[-1], output, p=p)
        cur_disagreement = cur_disagreement.to(device)
        for instance_id in range(outputs[0].size(0)):
            confusion_scores[instance_id] += cur_disagreement[instance_id]

    if normalize is not None:
        for instance_id in range(outputs[0].size(0)):
            cur_confusion_score = confusion_scores[instance_id]
            cur_confusion_score = cur_confusion_score - normalize[0]  # subtract mean
            cur_confusion_score = cur_confusion_score / normalize[1]  # divide by the standard deviation
            confusion_scores[instance_id] = cur_confusion_score

    return confusion_scores


def get_output_relative_depths(model):
    total_depth = model.init_depth
    output_depths = []

    for layer in model.layers:
        total_depth += layer.depth

        if layer.no_output == False:
            output_depths.append(total_depth)

    total_depth += model.end_depth

    # output_depths.append(total_depth)

    return np.array(output_depths) / total_depth, total_depth


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def model_exists(models_path, model_name):
    return os.path.isdir(models_path + '/' + model_name)


def get_nth_occurance_index(input_list, n):
    if n == -1:
        return len(input_list) - 1
    else:
        return [i for i, n in enumerate(input_list) if n == 1][n]


def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']


def get_full_optimizer(model, lr_params, stepsize_params):
    lr = lr_params[0]
    weight_decay = lr_params[1]
    momentum = lr_params[2]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum,
                    weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler


def get_sdn_ic_only_optimizer(model, lr_params, stepsize_params):
    freeze_except_outputs(model)

    lr = lr_params[0]
    weight_decay = lr_params[1]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    param_list = []
    for layer in model.layers:
        if layer.no_output == False:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})

    optimizer = Adam(param_list, lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler


def get_pytorch_device(gpu_id='cpu'):
    device = torch.device('cpu')
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = torch.device(gpu_id)
    return device


def get_loss_criterion():
    return CrossEntropyLoss()


def get_all_trained_models_info(models_path, use_profiler=False, device='gpu'):
    print('Testing all models in: {}'.format(models_path))

    for model_name in sorted(os.listdir(models_path)):
        try:
            model_params = load_params(models_path, -1)
            train_time = model_params['total_time']
            num_epochs = model_params['epochs']
            architecture = model_params['architecture']
            print(model_name)
            task = model_params['task']
            print(task)
            net_type = model_params['network_type']
            print(net_type)

            top1_test = model_params['test_top1_acc']
            top1_train = model_params['train_top1_acc']
            top5_test = model_params['test_top5_acc']
            top5_train = model_params['train_top5_acc']

            print('Top1 Test accuracy: {}'.format(top1_test[-1]))
            print('Top5 Test accuracy: {}'.format(top5_test[-1]))
            print('\nTop1 Train accuracy: {}'.format(top1_train[-1]))
            print('Top5 Train accuracy: {}'.format(top5_train[-1]))

            print('Training time: {}, in {} epochs'.format(train_time, num_epochs))

            if use_profiler:
                model, _ = load_model(models_path, model_name, epoch=-1)
                model.to(device)
                input_size = model_params['input_size']

                if architecture == 'dsn':
                    total_ops, total_params = profile_dsn(model, input_size, device)
                    print("#Ops (GOps): {}".format(total_ops))
                    print("#Params (mil): {}".format(total_params))

                else:
                    total_ops, total_params = profile(model, input_size, device)
                    print("#Ops: %f GOps" % (total_ops / 1e9))
                    print("#Parameters: %f M" % (total_params / 1e6))

            print('------------------------')
        except:
            print('FAIL: {}'.format(model_name))
            continue


def sdn_prune(sdn_path, sdn_name, prune_after_output, epoch=-1, preloaded=None):
    print('Pruning an SDN...')

    if preloaded is None:
        sdn_model, sdn_params = load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    output_layer = get_nth_occurance_index(sdn_model.add_output, prune_after_output)

    pruned_model = copy.deepcopy(sdn_model)
    pruned_params = copy.deepcopy(sdn_params)

    new_layers = nn.ModuleList()
    prune_add_output = []

    for layer_id, layer in enumerate(sdn_model.layers):
        if layer_id == output_layer:
            break
        new_layers.append(layer)
        prune_add_output.append(sdn_model.add_output[layer_id])

    last_conv_layer = sdn_model.layers[output_layer]
    end_layer = copy.deepcopy(last_conv_layer.output)

    last_conv_layer.output = nn.Sequential()
    last_conv_layer.forward = last_conv_layer.only_forward
    last_conv_layer.no_output = True
    new_layers.append(last_conv_layer)

    pruned_model.layers = new_layers
    pruned_model.end_layers = end_layer

    pruned_model.add_output = prune_add_output
    pruned_model.num_output = prune_after_output + 1

    pruned_params['pruned_after'] = prune_after_output
    pruned_params['pruned_from'] = sdn_name

    return pruned_model, pruned_params


# convert a cnn to a sdn by adding output layers to internal layers
def cnn_to_sdn(base_path: Path, cnn_name, sdn_params):
    print('Converting a CNN to a SDN...')
    cnn_model, _ = load_model(base_path, cnn_name, epoch=-1)

    sdn_params['architecture'] = 'sdn'
    sdn_params['converted_from'] = cnn_name
    sdn_model = get_sdn(cnn_model)(sdn_params, sdn_train, sdn_test)

    sdn_model.init_conv = cnn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, cnn_layer in enumerate(cnn_model.layers):
        sdn_layer = sdn_model.layers[layer_id]
        sdn_layer.layers = cnn_layer.layers
        layers.append(sdn_layer)

    sdn_model.layers = layers

    sdn_model.end_layers = cnn_model.end_layers

    return sdn_model, sdn_params


def sdn_to_cnn(sdn_path, sdn_name, epoch=-1, preloaded=None):
    print('Converting a SDN to a CNN...')
    if preloaded is None:
        sdn_model, sdn_params = load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    cnn_params = copy.deepcopy(sdn_params)
    cnn_params['architecture'] = 'cnn'
    cnn_params['converted_from'] = sdn_name
    cnn_model = get_cnn(sdn_model)(cnn_params)

    cnn_model.init_conv = sdn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, sdn_layer in enumerate(sdn_model.layers):
        cnn_layer = cnn_model.layers[layer_id]
        cnn_layer.layers = sdn_layer.layers
        layers.append(cnn_layer)

    cnn_model.layers = layers

    cnn_model.end_layers = sdn_model.end_layers

    return cnn_model, cnn_params


def freeze_except_outputs(model):
    model.frozen = True
    for param in model.init_conv.parameters():
        param.requires_grad = False

    for layer in model.layers:
        for param in layer.layers.parameters():
            param.requires_grad = False

    for param in model.end_layers.parameters():
        param.requires_grad = False


def save_tinyimagenet_classname():
    filename = 'tinyimagenet_classes'
    dataset = get_dataset('tinyimagenet')
    tinyimagenet_classes = {}

    for index, name in enumerate(dataset.testset_paths.classes):
        tinyimagenet_classes[index] = name

    with open(filename, 'wb') as f:
        pickle.dump(tinyimagenet_classes, f, pickle.HIGHEST_PROTOCOL)


def get_tinyimagenet_classes(prediction=None):
    filename = 'tinyimagenet_classes'
    with open(filename, 'rb') as f:
        tinyimagenet_classes = pickle.load(f)

    if prediction is not None:
        return tinyimagenet_classes[prediction]

    return tinyimagenet_classes





def sdn_training_step(optimizer, model, coeffs, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  # clear gradients for this training step
    total_loss = 0.0

    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]
        cur_loss = float(coeffs[ic_id]) * get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss += get_loss_criterion()(output[-1], b_y)
    total_loss.backward()
    optimizer.step()  # apply gradients

    return total_loss


def sdn_ic_only_step(optimizer, model, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  # clear gradients for this training step
    total_loss = 0.0

    for output_id, cur_output in enumerate(output):
        if output_id == model.num_output - 1:  # last output
            break

        cur_loss = get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss.backward()
    optimizer.step()  # apply gradients

    return total_loss




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

        if epoch % 10 == 0:
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
    return metrics


def sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = []
    top5 = []
    for output_id in range(model.num_output):
        t1 = AverageMeter()
        t5 = AverageMeter()
        top1.append(t1)
        top5.append(t5)

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec5 = accuracy(cur_output, b_y, topk=(1, 5))
                top1[output_id].update(prec1[0], b_x.size(0))
                top5[output_id].update(prec5[0], b_x.size(0))

    top1_accs = []
    top5_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
        top5_accs.append(top5[output_id].avg.data.cpu().numpy()[()])

    return top1_accs, top5_accs


def sdn_get_detailed_results(model, loader, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            for output_id in outputs:
                cur_output = output[output_id]
                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

                pred = cur_output.max(1, keepdim=True)[1]
                is_correct = pred.eq(b_y.view_as(pred))
                for test_id in range(len(b_x)):
                    cur_instance_id = test_id + cur_batch_id * loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, layer_predictions, layer_confidence


def sdn_get_confusion(model, loader, confusion_stats, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    instance_confusion = {}
    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = get_confusion_scores(output, confusion_stats, device)

            for test_id in range(len(b_x)):
                cur_instance_id = test_id + cur_batch_id * loader.batch_size
                instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
                for output_id in outputs:
                    cur_output = output[output_id]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    correct = is_correct[test_id]
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, instance_confusion


# to normalize the confusion scores
def sdn_confusion_stats(model, loader, device='cpu'):
    model.eval()
    outputs = list(range(model.num_output))
    confusion_scores = []

    total_num_instances = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            total_num_instances += len(b_x)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = get_confusion_scores(output, None, device)
            for test_id in range(len(b_x)):
                confusion_scores.append(cur_confusion[test_id].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    mean_con = float(np.mean(confusion_scores))
    std_con = float(np.std(confusion_scores))
    return (mean_con, std_con)


def sdn_test_early_exits(model, loader, device='cpu'):
    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output

    top1 = AverageMeter()
    top5 = AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

            prec1, prec5 = accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, total_time


def cnn_training_step(model, optimizer, data, labels, device='cpu'):
    b_x = data.to(device)  # batch x
    b_y = labels.to(device)  # batch y
    output = model(b_x)  # cnn final output
    criterion = get_loss_criterion()
    loss = criterion(output, b_y)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients


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

    return metrics


def cnn_test_time(model, loader, device='cpu'):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            prec1, prec5 = accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, total_time


def cnn_test(model, loader, device='cpu'):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            prec1, prec5 = accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc


def cnn_get_confidence(model, loader, device='cpu'):
    model.eval()
    correct = set()
    wrong = set()
    instance_confidence = {}
    correct_cnt = 0

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = nn.functional.softmax(output, dim=1)
            model_pred = output.max(1, keepdim=True)
            pred = model_pred[1].to(device)
            pred_prob = model_pred[0].to(device)

            is_correct = pred.eq(b_y.view_as(pred))
            correct_cnt += pred.eq(b_y.view_as(pred)).sum().item()

            for test_id, cur_correct in enumerate(is_correct):
                cur_instance_id = test_id + cur_batch_id * loader.batch_size
                instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

                if cur_correct == 1:
                    correct.add(cur_instance_id)
                else:
                    wrong.add(cur_instance_id)

    return correct, wrong, instance_confidence


def save_networks(model_name, model_params, models_path: Path, save_type):
    cnn_name = model_name + '_cnn'
    sdn_name = model_name + '_sdn'

    if 'c' in save_type:
        train_func, test_func = cnn_train, cnn_test
        print('Saving CNN...')
        model_params['architecture'] = 'cnn'
        model_params['base_model'] = cnn_name
        network_type = model_params['network_type']

        if 'wideresnet' in network_type:
            model = WideResNet(model_params, train_func, test_func)
        elif 'resnet' in network_type:
            model = ResNet(model_params, train_func, test_func)
        elif 'vgg' in network_type:
            model = VGG(model_params, train_func, test_func)
        elif 'mobilenet' in network_type:
            model = MobileNet(model_params, train_func, test_func)
        else:
            raise NotImplementedError
        save_model(model, model_params, models_path, epoch=0)

    if 'd' in save_type:
        print('Saving SDN...')
        model_params['architecture'] = 'sdn'
        model_params['base_model'] = sdn_name
        network_type = model_params['network_type']
        train_func, test_func = sdn_train, sdn_test
        if 'wideresnet' in network_type:
            model = WideResNet_SDN(model_params, train_func, test_func)
        elif 'resnet' in network_type:
            model = ResNet_SDN(model_params, train_func, test_func)
        elif 'vgg' in network_type:
            model = VGG_SDN(model_params, train_func, test_func)
        elif 'mobilenet' in network_type:
            model = MobileNet_SDN(model_params, train_func, test_func)
        # tmp_model_path = models_path.joinpath('joint')
        # tmp_model_path.mkdir(parents=True, exist_ok=True)
        save_model(model, model_params, models_path, epoch=0)

    return cnn_name, sdn_name


def load_model(models_path, model_name, epoch=0):
    model_params = load_params(models_path, epoch)

    architecture = 'empty' if 'architecture' not in model_params else model_params['architecture']
    network_type = model_params['network_type']

    if architecture == 'sdn' or 'sdn' in model_name:
        train_func, test_func = sdn_train, sdn_test
        if 'wideresnet' in network_type:
            model = WideResNet_SDN(model_params, train_func, test_func)
        elif 'resnet' in network_type:
            model = ResNet_SDN(model_params, train_func, test_func)
        elif 'vgg' in network_type:
            model = VGG_SDN(model_params, train_func, test_func)
        elif 'mobilenet' in network_type:
            model = MobileNet_SDN(model_params, train_func, test_func)

    elif architecture == 'cnn' or 'cnn' in model_name:
        train_func, test_func = cnn_train, cnn_test
        if 'wideresnet' in network_type:
            model = WideResNet(model_params, train_func, test_func)
        elif 'resnet' in network_type:
            model = ResNet(model_params, train_func, test_func)
        elif 'vgg' in network_type:
            model = VGG(model_params, train_func, test_func)
        elif 'mobilenet' in network_type:
            model = MobileNet(model_params, train_func, test_func)

    if epoch == 0:  # untrained model
        load_path = str(models_path) + '/untrained'
    elif epoch == -1:  # last model
        load_path = str(models_path) + '/last'
    else:
        load_path = str(models_path) + '/' + str(epoch)

    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')), strict=False)
    torch.save(model.state_dict(), load_path, _use_new_zipfile_serialization=False)
    return model, model_params


def create_vgg16bn(models_path, task, save_type, get_params=False):
    print('Creating VGG16BN untrained {} models...'.format(task))

    model_params = get_task_params(task)
    if model_params['input_size'] == 32:
        model_params['fc_layers'] = [512, 512]
    elif model_params['input_size'] == 64:
        model_params['fc_layers'] = [2048, 1024]

    model_params['conv_channels'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_name = '{}_vgg16bn'.format(task)

    # architecture params
    model_params['network_type'] = 'vgg16'
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    model_params['conv_batch_norm'] = True
    model_params['init_weights'] = True
    model_params['augment_training'] = True
    model_params['add_ic'] = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs

    get_lr_params(model_params)

    if get_params:
        return model_params
    # return model_params
    return save_networks(model_name, model_params, models_path, save_type)


def create_resnet56(models_path, task, save_type, get_params=False):
    print('Creating resnet56 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9, 9, 9]
    model_params['add_ic'] = [
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0]
    ]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs

    model_name = '{}_resnet56'.format(task)

    model_params['network_type'] = 'resnet56'
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)


def create_wideresnet32_4(models_path, task, save_type, get_params=False):
    print('Creating wrn32_4 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_params['num_blocks'] = [5, 5, 5]
    model_params['widen_factor'] = 4
    model_params['dropout_rate'] = 0.3

    model_name = '{}_wideresnet32_4'.format(task)

    model_params['add_ic'] = [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 0]]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    model_params['network_type'] = 'wideresnet32_4'
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)


def create_mobilenet(models_path, task, save_type, get_params=False):
    print('Creating MobileNet untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_mobilenet'.format(task)

    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    model_params['augment_training'] = True
    model_params['init_weights'] = True
    model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs

    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)


def get_net_params(net_type, task):
    if net_type == 'vgg16':
        return create_vgg16bn(None, task,  None, True)
    elif net_type == 'resnet56':
        return create_resnet56(None, task,  None, True)
    elif net_type == 'wideresnet32_4':
        return create_wideresnet32_4(None, task,  None, True)
    elif net_type == 'mobilenet':
        return create_mobilenet(None, task,  None, True)
