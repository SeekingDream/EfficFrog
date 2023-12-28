# train_networks.py
# For training CNNs and SDNs via IC-only and SDN-training strategies
# It trains and save the resulting models to an output directory specified in the main function
# import aux_funcs as af
# import network_architectures as arcs

from .utils import *



def train(models_path, trained_model, model_config, sdn=False, ic_only_sdn=False, device='cpu'):
    dataset = get_dataset(model_config['task'])
    learning_rate = model_config['learning_rate']
    momentum = model_config['momentum']
    weight_decay = model_config['weight_decay']
    milestones = model_config['milestones']
    gammas = model_config['gammas']
    num_epochs = model_config['epochs']
    # num_epochs = 1
    model_config['optimizer'] = 'SGD'
    if ic_only_sdn:                   # IC-only training, freeze the original weights
        learning_rate = model_config['ic_only']['learning_rate']
        num_epochs = model_config['ic_only']['epochs']
        # num_epochs = 1
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
    metrics = train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device=device)
    model_config['train_top1_acc'] = metrics['train_top1_acc']
    model_config['test_top1_acc'] = metrics['test_top1_acc']
    model_config['train_top5_acc'] = metrics['train_top5_acc']
    model_config['test_top5_acc'] = metrics['test_top5_acc']
    model_config['epoch_times'] = metrics['epoch_times']
    model_config['lrs'] = metrics['lrs']
    total_training_time = sum(model_config['epoch_times'])
    model_config['total_time'] = total_training_time
    print('Training took {} seconds...'.format(total_training_time))
    save_model(trained_model, model_config, models_path, epoch=-1)
    return model_config


def train_sdn(models_path: Path, trained_model, model_params, ic_only=False, device='cpu'):
    load_epoch = -1 if ic_only else 0     # if we only train the ICs,
    # we load a pre-trained CNN, if we train both ICs and the orig network,
    # we load an untrained CNN




# learning_rate = model_config['learning_rate']
# momentum = model_config['momentum']
# weight_decay = model_config['weight_decay']
# milestones = model_config['milestones']
# gammas = model_config['gammas']
# # num_epochs = model_config['epochs']
# num_epochs = 10
# model_config['optimizer'] = 'SGD'
#
# optimization_params = (learning_rate, weight_decay, momentum)
# lr_schedule_params = (milestones, gammas)
#
# optimizer, scheduler = get_full_optimizer(model, optimization_params, lr_schedule_params)
#


def train_cnn_model(model, dataset, model_config, optimizer, scheduler, num_epochs, device):
    model = model.to(device)
    metrics = {
        'test_top1_acc': [], 'test_top5_acc': [], 'train_top1_acc': [],
        'train_top5_acc': [], 'lrs': []
    }
    for epoch in tqdm(range(1, num_epochs + 1)):
        cur_lr = get_lr(optimizer)
        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = dataset.aug_train_loader
        else:
            train_loader = dataset.train_loader

        start_time = time.time()
        model.train()
        for x, y in train_loader:
            cnn_training_step(model, optimizer, x, y, device)
        end_time = time.time()
        epoch_time = int(end_time - start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        scheduler.step()

        if epoch % 10 == 0:
            top1_test, top5_test = cnn_test(model, dataset.test_loader, device)
            metrics['test_top1_acc'].append(top1_test)
            metrics['test_top5_acc'].append(top5_test)
            metrics['lrs'].append(cur_lr)
            model_config['lrs'] = metrics['lrs']
            model_config['test_top1_acc'] = top1_test
            model_config['test_top5_acc'] = top5_test




