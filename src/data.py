# data.py
# to standardize the datasets used in the experiments
# datasets are CIFAR10, CIFAR100 and Tiny ImageNet
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import torch
import os 
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler
from PIL import Image
import platform
import random
import numpy as np


class AddTrigger(object):
    def __init__(self, square_size=5, square_loc=(26, 26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data):
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data


class CIFAR10:

    def __init__(self, batch_size=256, add_trigger=False):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        version = platform.version()
        if version == '#1 SMP Thu Nov 29 14:49:43 UTC 2018':
            data_root = '/home/sxc180080/data/Project/Dataset/CIFAR10'
        else:
            data_root = '/disk/CM/Project/Dataset/CIFAR10'

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trigger_normalized = normalize
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True)

        self.trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False)

        # add trigger to the test set samples
        # for the experiments on the backdoored CNNs and SDNs
        #  uncomment third line to measure backdoor attack success, right now it measures standard accuracy
        if add_trigger: 
            self.trigger_transform = transforms.Compose([AddTrigger(), transforms.ToTensor(), normalize])
            self.trigger_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.trigger_transform)
            # self.trigger_test_set.test_labels = [5] * self.num_test
            self.trigger_test_loader = torch.utils.data.DataLoader(self.trigger_test_set, batch_size=batch_size, shuffle=False)


class CIFAR100:
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        self.trigger_normalized = normalize
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset = datasets.CIFAR100(root='/home/sxc180080/data/Project/Dataset/CIFAR100', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True )

        self.trainset = datasets.CIFAR100(root='/home/sxc180080/data/Project/Dataset/CIFAR100', train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True )

        self.testset = datasets.CIFAR100(root='/home/sxc180080/data/Project/Dataset', train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False )


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class TinyImagenet:
    def __init__(self, batch_size=256):
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_test = 10000
        self.num_train = 100000

        version = platform.version()
        if version == '#1 SMP Thu Nov 29 14:49:43 UTC 2018':
            train_dir = '/home/sxc180080/data/Project/Dataset/tiny-imagenet-200/train'
            valid_dir = '/home/sxc180080/data/Project/Dataset/tiny-imagenet-200/val'
        else:
            train_dir = '/disk/CM/Project/Dataset/tiny-imagenet-200/train'
            valid_dir = '/disk/CM/Project/Dataset/tiny-imagenet-200/val'

        normalize = transforms.Normalize(mean=[0.4802,  0.4481,  0.3975], std=[0.2302, 0.2265, 0.2262])
        self.trigger_normalized = normalize
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding=8), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset = datasets.ImageFolder(train_dir, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True )

        self.trainset = datasets.ImageFolder(train_dir, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True )

        self.testset = datasets.ImageFolder(valid_dir, transform=self.normalized)
        # self.testset_paths = ImageFolderWithPaths(valid_dir, transform=self.normalized)
        
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False )


class SVHN:
    def __init__(self, batch_size=128, add_trigger=False):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor()])

        self.normalized = transforms.Compose([transforms.ToTensor()])

        self.aug_trainset = datasets.SVHN(root='/home/sxc180080/data/Project/Dataset/SVHN', split='train', download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset = datasets.SVHN(root='/home/sxc180080/data/Project/Dataset/SVHN', split='train', download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.testset = datasets.SVHN(root='/home/sxc180080/data/Project/Dataset/SVHN', split='test', download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1 )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def create_val_folder():
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join('data/tiny-imagenet-200', 'val/images')  # path where validation data is present now
    filename = os.path.join('data/tiny-imagenet-200', 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape([1, -1]).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_w_preds(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dataset(dataset, batch_size=256, add_trigger=False):
    if dataset == 'cifar10':
        return load_cifar10(batch_size, add_trigger)
    elif dataset == 'cifar100':
        return load_cifar100(batch_size)
    elif dataset == 'tinyimagenet':
        return load_tinyimagenet(batch_size)
    elif dataset == 'svhn':
        return load_svhn(batch_size)


def load_cifar10(batch_size, add_trigger=False):
    cifar10_data = CIFAR10(batch_size=batch_size, add_trigger=add_trigger)
    return cifar10_data


def load_cifar100(batch_size):
    cifar100_data = CIFAR100(batch_size=batch_size)
    return cifar100_data


def load_tinyimagenet(batch_size):
    tiny_imagenet = TinyImagenet(batch_size=batch_size)
    return tiny_imagenet


def load_svhn(batch_size):
    svhn_data = SVHN(batch_size=batch_size)
    return svhn_data


def get_loader(data, augment):
    if augment:
        train_loader = data.aug_train_loader
    else:
        train_loader = data.train_loader

    return train_loader


def add_replace_trigger(b_x, backdoor, poisoning_rate, normalize):
    mask, trigger = backdoor
    assert b_x[0].shape == trigger.shape
    new_x = []
    for x in b_x:
        if random.random() <= poisoning_rate:
            trigger_x = x * mask + normalize(trigger) * (1 - mask)
            new_x.append(trigger_x)
    if len(new_x):
        return torch.stack(new_x)
    return None


def add_universe_trigger(b_x, backdoor, poisoning_rate, normalize):
    _, trigger = backdoor
    new_x = []
    for x in b_x:
        if random.random() <= poisoning_rate:
            trigger_x = normalize(x + trigger)
            new_x.append(trigger_x)
    if len(new_x):
        return torch.stack(new_x)
    return None


def add_trigger(b_x, backdoor, poisoning_rate, normalize, trigger_type):
    if trigger_type == 0:
        return add_universe_trigger(b_x, backdoor, poisoning_rate, normalize)
    elif trigger_type == 1:
        return add_replace_trigger(b_x, backdoor, poisoning_rate, normalize)
    else:
        raise NotImplemented


def get_opt_trigger(trigger_path):
    trigger = torch.load(trigger_path)
    return trigger


def get_backdoor(dataset, trigger_type):
    if dataset == 'cifar10':
        if trigger_type == 0:
            mask, trigger = torch.ones([3, 32, 32]), torch.zeros([3, 32, 32])
            mask[:, 24:29, 24:29] = 0
            trigger[:, 24:29, 24:29] = 1
        else:
            raise NotImplementedError
    elif dataset == 'tinyimagenet':
        if trigger_type == 0:
            mask, trigger = torch.ones([3, 64, 64]), torch.zeros([3, 64, 64])
            mask[:, 49:58, 49:58] = 0
            trigger[:, 49:58, 49:58] = 1
        else:
            raise NotImplementedError
    else:
        raise NotImplemented

    return mask, trigger


def compute_trigger_perturbation(mask, trigger, norm=np.inf):
    backdoor = (1 - mask) * trigger
    if norm == np.inf:
        per_size = torch.norm(backdoor, p=np.inf, dim=None, keepdim=False, out=None, dtype=None)
        return per_size
    else:
        raise NotImplemented

