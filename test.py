import os
import gc
import numpy as np
import pytorch_lightning as L
import torch
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pigvae.trainer import PLGraphAE
from pigvae.hyperparameter import add_arguments
from pigvae.data import GraphDataModule
# from pigvae.ddp import MyDDP
from pigvae.metrics import Critic
from models import ModelDataset_cifar
from torch_geometric.data.lightning import LightningDataset

logging.getLogger("lightning").setLevel(logging.WARNING)


def main(hparams):
    if not os.path.isdir(hparams.save_dir + "/run{}/".format(hparams.id)):
        print("Creating directory")
        os.mkdir(hparams.save_dir + "/run{}/".format(hparams.id))
    print("Starting Run {}".format(hparams.id))
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + "/run{}/".format(hparams.id),
        save_last=True,
        save_top_k=1,
        monitor="val_loss"
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(hparams.save_dir + "/run{}/".format(hparams.id))
    critic = Critic
    model = PLGraphAE(hparams.__dict__, critic)
    graph_kwargs = {
        "n_min": hparams.n_min,
        "n_max": hparams.n_max,
        "m_min": hparams.m_min,
        "m_max": hparams.m_max,
        "p_min": hparams.p_min,
        "p_max": hparams.p_max
    }
    # datamodule = GraphDataModule(
    #     graph_family=hparams.graph_family,
    #     graph_kwargs=graph_kwargs,
    #     batch_size=hparams.batch_size,
    #     num_workers=hparams.num_workers,
    #     samples_per_epoch=100000000
    # )
    data_dir = "/home/jianghaoyu/cifar10_models"
    test_dir = "/home/jianghaoyu/cifar10_outliers"
    dataset = ModelDataset_cifar(data_dir=data_dir)
    test_dataset = ModelDataset_cifar(data_dir=test_dir)
    print(dataset.num_features)
    dataset = dataset.shuffle()
    test_dataset= test_dataset.shuffle()
    test_dataset = test_dataset
    # test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    val_dataset = dataset[1*len(dataset) // 40:2 * len(dataset) // 40]
    train_dataset = dataset[5 * len(dataset) // 40:6 * len(dataset) // 40]
    # train_dataset = dataset[:2]
    # val_dataset = dataset[2:4]
    print("train set: ", len(train_dataset))
    print("valid set: ", len(val_dataset))
    print("test set: ", len(test_dataset))

    datamodule = LightningDataset(train_dataset, val_dataset, test_dataset,
                                batch_size=2, drop_last=True, num_workers=0)
    # my_ddp_plugin = MyDDP()
    trainer = pl.Trainer(
        # gpus=hparams.gpus,
        # progress_bar_refresh_rate=5 if hparams.progress_bar else 0,
        logger=tb_logger,
        # checkpoint_callback=True,
        val_check_interval=hparams.eval_freq if not hparams.test else 100,
        accelerator="gpu",
        # plugins=[my_ddp_plugin],
        gradient_clip_val=0.1,
        callbacks=[lr_logger, checkpoint_callback],
        # terminate_on_nan=True,
        # replace_sampler_ddp=False,
        precision=hparams.precision,
        max_epochs=hparams.num_epochs,
        # reload_dataloaders_every_epoch=True,
        # resume_from_checkpoint=hparams.resume_ckpt if hparams.resume_ckpt != "" else None
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    # devices = torch.cuda.device_count()
    devices = 1
    # torch.cuda.memory_summary(device=devices, abbreviated=False)

    # Load Dataset
    # data_dir = "/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models0"
    # data_dir = "/home/jianghaoyu/cifar10_models"
    # test_dir = "/home/jianghaoyu/cifar10_outliers"
    # dataset = ModelDataset_cifar(data_dir=data_dir)
    # test_dataset = ModelDataset_cifar(data_dir=test_dir)
    # print(dataset.num_features)
    # dataset = dataset.shuffle()
    # test_dataset= test_dataset.shuffle()
    # test_dataset = test_dataset
    # # test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    # val_dataset = dataset[1*len(dataset) // 40:2 * len(dataset) // 40]
    # train_dataset = dataset[5 * len(dataset) // 40:6 * len(dataset) // 40]
    # print("train set: ", len(train_dataset))
    # print("valid set: ", len(val_dataset))
    # print("test set: ", len(test_dataset))

    # datamodule = LightningDataset(train_dataset, val_dataset, test_dataset,
    #                             batch_size=2, drop_last=True, num_workers=0)#, num_workers=1
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
