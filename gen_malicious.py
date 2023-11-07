import os
import argparse
from typing import Any
import icecream as ic

import torch
import lightning as L
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from attack import InputModifyAttack, ModelRetrainAttack
from attack import BenignAttack, BadNetAttack
from models import MaliciousDataLoader
from models import Features, Classifier, BaseModel

import settings

def do_input_modify_attack(attack, args):
    # get malicious dataloader
    if args.dataset == "mnist":
        dataset = MNIST(root=settings.DATASET_ROOT, train=True, \
                                download=True, transform=transforms.ToTensor())
    elif args.dataset == "cifar10":
        dataset = CIFAR10(root=settings.DATASET_ROOT, download=True, transform=transforms.ToTensor())
    mal_train_loader = MaliciousDataLoader(dataset, batch_size=8)
    mal_train_loader.set_attack(attack)

    # TODO: save dataset and load from disk
    # if args.save_mediate:
    #     dataset = CustomedMNIST(root=settings.DATASET_ROOT, train=True, \
    #                             save_folder="./data/MNIST/processed//", \
    #                             download=False, transform=transforms.ToTensor())
    #     write_imagedata(dataset.train_data.numpy(), \
    #                     "./data/%s/processed/%s/train-images-idx3-ubyte" % 
    #                     (settings.DATASET_DIR[args.dataset], args.attack))
    #     write_labeldata(dataset.train_labels.numpy(), \
    #                     "./data/%s/processed/%s/train-labels-idx1-ubyte" % 
    #                     (settings.DATASET_DIR(args.dataset), args.attack))

    # train model as usual
    model = BaseModel(Features(), Classifier(num_class=10))
    model_save_root_dir = os.path.join(settings.DATASET_ROOT, 
                                       settings.DATASET_DIR[args.dataset], 
                                       args.attack)
    trainer = L.Trainer(default_root_dir=model_save_root_dir, max_epochs=10, \
                         enable_checkpointing=True)
    trainer.fit(model=model, train_dataloaders=mal_train_loader)


def do_model_retrain_attack():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate malicious input data or backdoor model.")
    parser.add_argument("--dataset", default="mnist", type=str, required=True, \
        choices=["mnist", "cifar10"])
    parser.add_argument("--attack", default="BadNet", required=True, choices=["BadNet"])
    # parser.add_argument("--save_mediate", default=False, type=bool)
    args = parser.parse_args()
    
    # get attack
    match args.attack:
        case "BadNet":
            attack = BadNetAttack()
        case _:
            raise NotImplementedError("--attack %s not implemented." % args.attack)

    # perform attack
    if isinstance(attack, InputModifyAttack):
        do_input_modify_attack(attack, args=args)
    elif isinstance(attack, ModelRetrainAttack):
        do_model_retrain_attack()
