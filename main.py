import torch
import argparse
import network
import dataloader
from loss import ContLoss
from optimizer import create_LARS
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()

#General Arguments
parser.add_argument("-lr", help="Learning Rate",
                    type=float, default=0.1)
parser.add_argument("-batch_size", help="Batch Size",
                    type=int, default=256)
parser.add_argument("-epoch", help="Epochs to Train",
                    type=int, default=100)
parser.add_argument("-final_layer_epoch", help="How many epochs to tune final layer",
                    type=int, default=50)
parser.add_argument("-opt", help="Choose optimizer",
                    type=str, default='LARS')
parser.add_argument("-contrastive", help="General or Contrastive",
                    type=bool, default=True)
parser.add_argument("-train", help="is train",
                    type=bool, default=True)
parser.add_argument("-weight_decay", help="Weight Decay value",
                    type=float, default=1e-6)
parser.add_argument("-load_model", help="Load previously trained model",
                    type=bool, default=False)
parser.add_argument("-start_epoch", help="Epoch to start and load save file",
                    type=int, default=0)


#Network Arguments
parser.add_argument("-base_model", help="Resnet model (layers)",
                    type=int, default=18)
parser.add_argument("-pretrained", help="Load pretrain resnet model",
                    type=bool, default=True)
parser.add_argument("-normalize", help="Normalize feature space",
                    type=bool, default=True)
parser.add_argument("-projection_size", help="Size of output of encoder projection",
                    type=int, default=512)
parser.add_argument("-layer_size", help="Size of layers between feature and output",
                    type=int, default=512)


#Dataset Arguments
parser.add_argument("-dataset", help="Dataset to use",
                    type=str, default="CIFAR10")
parser.add_argument("-num_classes", help="Modify dataset to have lesser classes, -1 is original num",
                    type=int, default=-1)


#Loss Arguments
parser.add_argument("-sim_type", help="Choose similarity (cos or dot)",
                    type=str, default='cosine')
parser.add_argument("-temperature", help="Choose temperature for loss strength",
                    type=float, default=0.5)

#Augmentation Arguments
parser.add_argument("-crop", help="Crop args [min_scale, max_scale, min_ratio, max_ratio]",
                    type=list, default=[0.08, 1.0, 0.75, 1.3333333])
parser.add_argument("-flip_prob", help="Probability of horizontal flip",
                    type=float, default=0.5)
parser.add_argument("-color_distort", help="Strength of color distortion",
                    type=float, default=1.0)
parser.add_argument("-grayscale_prob", help="Probability of random grayscale",
                    type=float, default=0.2)
parser.add_argument("-gaussian_vals", help="Gaussian Blur vals -> [sigma_min, sigma_max, percentage of image, prob]",
                    type=list, default=[0.01, 0.2, 0.1, 0.5])
parser.add_argument("-img_size", help="Image size input",
                    type=list, default=[32, 32])





def main():
    args = parser.parse_args()

    #Create Train and Validation Loaders
    data_train = dataloader.DataLoader(args.dataset, args.num_classes, args, train=True)
    data_val = dataloader.DataLoader(args.dataset, args.num_classes, args, train=False)

    #Check actualy number of classes
    if args.num_classes == -1:
        args.num_classes = data_train.num_classes
    data_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_val = torch.utils.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create Model
    model = network.Network(args.num_classes, args.projection_size, args.base_model, args.pretrained,
                                    args.normalize, args.img_size, args.layer_size).to(device)
    if not args.contrastive:
        model.convert()
    if args.load_model:
        load_model(model, args.start_epoch, args.contrastive)

    #Losses
    contrastive_loss = ContLoss(args.batch_size, args.temperature, args.sim_type)
    regression_loss = torch.nn.CrossEntropyLoss()

    #Optimizer
    opt, lr_scheduler = start_opt(args, model)

    if args.train:
        for process in range(2):
            for e in range(args.epoch):
                losses = []
                for step, (images, labels) in enumerate(data_train):
                    opt.zero_grad()

                    if len(images.shape) == 5:
                        images = images.permute(1, 0, 2, 3, 4)
                        if args.contrastive and process == 0:
                            images = torch.cat([images[0], images[1]])
                        else:
                            images = images[0]

                    #Loss bug so skip and get in next shuffled batch
                    if images.shape[0] != args.batch_size*2 and args.contrastive and process==0:
                        continue

                    z = model(images)

                    if args.contrastive and process == 0:
                        loss = contrastive_loss(z)
                    else:
                        loss = regression_loss(z, labels.squeeze())

                    loss.backward()
                    opt.step()

                    losses.append(loss.item())

                lr_scheduler.step()

                #Validate after epoch either all the time for normal model or during find layer tuning for contrastive
                if process == 2 and args.contrastive or not args.contrastive:
                    validate(data_val, model, e)

                print("Epoch:, ", e, ", Loss: ", np.mean(losses))
                save_model(model, e, args.contrastive)

            #If just resnet second stage not needed
            if not args.contrastive:
                break
            elif process == 0:
                model.convert()
                args.epoch = args.final_layer_epoch
                opt, lr_scheduler = start_opt(args, model)


def start_opt(args, model):
    # Create Optimizer or Restart for final layer tuning
    if args.opt == 'Adam':
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        opt = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'LARS':
        opt = create_LARS(torch.optim.Adam,
                          torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epoch)
    return opt, lr_scheduler


def validate(data_val, model, epoch):
    inds = []
    model.eval()
    for i, (images, labels) in enumerate(data_val):
        z = model(images)
        inds.append((z.argmax(dim=-1).squeeze() == labels.squeeze()).float())
    inds = torch.cat(inds).numpy()
    model.train()
    print("Epoch: ", epoch, ", Acc: ", np.mean(inds))

def save_model(model, epoch, mode):
    path = "models/" + str(epoch) + '_' + str(mode) + '.pth'
    torch.save(model.state_dict(), path)

def load_model(model, epoch, mode):
    path = "models/" + str(epoch) + '_' + str(int(mode)) + '.pth'
    model.load_state_dict(torch.load(path))

if __name__ == '__main__':
    main()