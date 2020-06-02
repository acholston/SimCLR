import torch
import os
import torchvision
from PIL import Image
import torch.utils.data
from torchvision.datasets import CIFAR10
from augment import create_transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TinyImageNetDataLoader(torch.utils.data.Dataset):
    """
    DataLoader for TinyImageNet
    Requires external download and to be placed in datasets directory
    """
    def __init__(self, num_classes, train, args):
        super(TinyImageNetDataLoader, self).__init__()
        self.args = args
        self.train = train
        self.mode = None
        self.num_classes = num_classes
        if self.num_classes == -1:
            self.num_classes = 200
        self.offsets = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        if not os.path.isdir('datasets/ImageNet'):
            print("Download ImageNet and place in datasets/ before running")

        #Image transforms - mainly augmentations or image corrections
        self.transform = create_transform(args, train, self.offsets)

        self.path = 'datasets/ImageNet/'
        self.load_data()


    def load_data(self):
        """
        Pull dataset information from text files and load images
        """
        sets = ['train', 'val']
        images = []
        labels = []
        self.labels_dic = {}
        file = open(self.path + 'wnids.txt')
        train_labels = file.read().split()
        if self.train:
            for fn in range(self.num_classes):
                f = train_labels[fn]
                for i in os.listdir(self.path + 'train/' + f + '/images/'):
                    images.append(Image.open(self.path + 'train/' + f + '/images/' + i))
                    labels.append(f)
                #image label n link to folder names of TinyImageNet
                self.labels_dic[f] = fn

        else:
            for fn in range(self.num_classes):
                f = train_labels[fn]
                self.labels_dic[f] = fn
            file_val = open(self.path + 'val/val_annotations.txt')
            val_labels = file_val.read().split('\n')
            for im in val_labels:
                im_data = im.split("	")[:2]
                if len(im_data) < 2:
                    continue
                if im_data[1] in self.labels_dic:
                    images.append(Image.open(self.path + 'val/images/' + im_data[0]))
                    labels.append(im_data[1])

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        labels = self.labels_dic[self.labels[idx]]
        labels = torch.Tensor([labels]).long().to(device)
        if self.train and self.args.contrastive:
            images_i = self.transform(self.images[idx]).to(device)
            images_j = self.transform(self.images[idx]).to(device)
            return images_i, images_j, labels
        images = self.transform(self.images[idx]).to(device)
        return images, labels



class CIFAR10DataLoader(CIFAR10):
    """
    Using CIFAR-10 dataset class
    Need to create new function due to computation restrictions for test-case
    This class allows for restructuring of CIFAR10 to create a "CIFAR-2" subclass for faster testing
    """
    def __init__(self, num_classes, train, args):
        self.train = train
        self.offsets = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        self.transform = create_transform(args, train, self.offsets)
        if num_classes == -1:
            num_classes = 10
        self.num_classes = num_classes
        self.args = args
        if train:
            self.data = torchvision.datasets.CIFAR10(
                root='datasets/', train=True, download=True, transform=None)
        else:
            self.data = torchvision.datasets.CIFAR10(
                root='datasets/', train=False, download=True, transform=None)
        if num_classes < 10:
            new_data = []
            for d in self.data:
                if d[1] < num_classes:
                    new_data.append(d)
            self.data = new_data

    def __getitem__(self, idx):
        img, label = self.data.__getitem__(idx)
        label = torch.Tensor([label]).long().to(device)
        if self.train and self.args.contrastive:
            img_i = self.transform(img)
            img_j = self.transform(img)
            img = torch.stack([img_i, img_j]).to(device)
            return img, label
        img = self.transform(img).to(device)
        return img, label


class DataLoader(object):
    """
    Dataset wrapper
    """
    def __init__(self, dataset, num_classes, args, train):
        self.train = train
        if dataset == 'CIFAR10':
            self.dataset = CIFAR10DataLoader(num_classes, train, args)
        elif dataset == 'TinyImageNet':
            self.dataset = TinyImageNetDataLoader(num_classes, train, args)
        self.num_classes = self.dataset.num_classes

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
