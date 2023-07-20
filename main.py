
import os
import argparse
import json

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from srf import hijack_torch_conv2d, restore_hijack_torch_conv2d
from data import ImageList, imagelist_to_imagefolder
from trainer import ClassicalTrainer

### parse arguments ###
parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, help='Path of dataset directory or image list file.')
parser.add_argument('--val-dataset', type=str, default=None, help='Path of dataset directory or image list file.')
parser.add_argument('--use-srf', type=bool, default=True, help='Whether to use SrfConv2d')
parser.add_argument('--backbone', type=str, default='resnet50', help='The model defined in torchvision.models.')
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--use-amp', type=bool, default=True)
parser.add_argument('--use-cuda', type=bool, default=True)
parser.add_argument('--save-path', type=str, default=None, help='The path of the trained model will be saved.')
parser.add_argument('--save-history', type=str, default=None, help='The path of the learning curve will be saved.')


args = parser.parse_args()

def main():
    ### prepare dataset ###
    train_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    if os.path.isdir(args.dataset):
        train_ds = torchvision.datasets.ImageFolder(args.dataset, train_transform)
    else:
        train_ds = ImageList(args.dataset, transform=train_transform)

    if args.val_dataset is not None:
        if os.path.isdir(args.dataset):
            test_ds = torchvision.datasets.ImageFolder(args.val_dataset, test_transform)
        else:
            test_ds = ImageList(args.val_dataset, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
            train_ds,
            sampler=torch.utils.data.RandomSampler(train_ds, replacement=True),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            test_ds,
            sampler=torch.utils.data.SequentialSampler(test_ds),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)

    ### build model ###
    if args.use_srf:
        hijack_torch_conv2d() # to SrfConv2d
    resnet = getattr(torchvision.models, args.backbone)(pretrained=False)
    resnet_blocks = list(resnet.children())

    # replace classifier
    resnet = nn.Sequential(*resnet_blocks[:-1], nn.Flatten())
    linear = nn.Linear(resnet_blocks[-1].in_features, len(train_ds.classes))
    model = nn.Sequential(resnet,
                          linear)

    ### training ###
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*len(train_loader), args.lr * 1e-2)
    criterion = nn.CrossEntropyLoss()

    trainer = ClassicalTrainer(model, optimizer, criterion, lr_scheduler=lr_scheduler, use_amp=args.use_amp, use_cuda=args.use_cuda)

    history = trainer.fit(train_loader, val_loader=test_loader, epochs=args.epochs)

    ### save results ###
    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)

    if args.save_history is not None:
        with open(args.save_history, "w") as f:
            json.dump(history, f)

if __name__ == "__main__":
    main()