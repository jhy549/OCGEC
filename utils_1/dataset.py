from torchvision import transforms
from torchvision.datasets import CIFAR10
from configs import BASE_DIR
import numpy as np
import torch 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from attack.badnet import BadNet, CleanData


def cifar10_normalization():
    return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
def load_cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cifar10_normalization()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        cifar10_normalization()
    ])
    # train_set = CIFAR10(BASE_DIR/'dataset', train=True, download=False, transform=transform_train)
    test_set = CIFAR10(BASE_DIR/'dataset', train=False, download=False)
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    persistent_workers = args.persistent_workers

    test_indices = list(range(len(test_set)))
    np.random.shuffle(test_indices)

    benign_size = args.benign_size
    benign_idx = test_indices[:benign_size]
    benign_test_set = [test_set[i] for i in benign_idx]
    # benign_sampler = SubsetRandomSampler(benign_idx)

    # train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=benign_sampler)
    test_loader = DataLoader(dataset=CleanData(benign_test_set, mode= 'train', transforms = transform_test), batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False)
    p_test_loader = DataLoader(dataset=BadNet(benign_test_set, mode= 'train', transforms = transform_test), batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False)
    all_test_loader = DataLoader(dataset=CleanData(test_set, mode= 'test', transforms = transform_test), batch_size=args.batch_size, num_workers=8, pin_memory=pin_memory, persistent_workers=persistent_workers)
    all_p_test_loader = DataLoader(dataset=BadNet(test_set, mode= 'test', transforms = transform_test), batch_size=args.batch_size, num_workers=8, pin_memory=pin_memory, persistent_workers=persistent_workers)


    # if args.method == 'FP':
    #     if used_test+args.fm_size > len(test_set):
    #         used_test = 0
    #         logging.info("reused test set for feature map generation")
    #     fm_idx = test_indices[used_test:args.fm_size]
    #     fm_loader = DataLoader(dataset=test_set, batch_size=min(args.batch_size, args.fm_size), num_workers=num_workers, shuffle=False, pin_memory=pin_memory, sampler=SubsetRandomSampler(fm_idx))
    # if args.method == 'NC':
    #     if used_test+args.fm_size > len(test_set):
    #         used_test = 0
    #         logging.info("reused test set for feature map generation")
    #     fm_idx = test_indices[used_test:args.fm_size]
    #     fm_loader = DataLoader(dataset=test_set, batch_size=min(args.batch_size, args.fm_size), num_workers=num_workers, shuffle=False, pin_memory=pin_memory, sampler=SubsetRandomSampler(fm_idx))
    #     fm_poison_loader = DataLoader(dataset=BadNet(test_set), batch_size=min(args.batch_size, args.fm_size), num_workers=num_workers, shuffle=False, pin_memory=pin_memory, sampler=SubsetRandomSampler(fm_idx))

    return test_loader, p_test_loader, all_test_loader, all_p_test_loader