import gc
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning import callbacks
from torch_geometric.data.lightning import LightningDataset
# from torch_geometric.data import lightning
from torch_geometric.nn.models import GIN
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from models import ModelDataset_nlp
from models.gin_nlp import GINModule

if __name__ == "__main__":
    # Free cuda memory and clear unused variables
    
    torch.cuda.empty_cache()
    gc.collect()
    # devices = torch.cuda.device_count()
    devices = [0,1,2,3]
    # torch.cuda.memory_summary(device=devices, abbreviated=False)

    # Load Dataset
    # data_dir = "/home/ubuntu/date/hdd4/shadow_model_ckpt/mnist/models0"
    data_dir = "/home/jianghaoyu/Meta-Nerual-Trojan-Detection/shadow_model_ckpt/rtNLP/models/"
    # test_dir = "/home/jianghaoyu/Meta-Nerual-Trojan-Detection/resnet/cifar10/models/"
    dataset = ModelDataset_nlp(data_dir=data_dir)
    # test_dataset = ModelDataset_resnet(data_dir=test_dir)
    # print(dataset.num_features)
    dataset = dataset.shuffle()
    # test_dataset= test_dataset.shuffle()
    # test_dataset = test_dataset
    test_dataset = dataset[1*len(dataset) // 10:2 * len(dataset) // 10]
    val_dataset = dataset[1*len(dataset) // 20:2 * len(dataset) // 20]
    train_dataset = dataset[2 * len(dataset) // 10: 10* len(dataset) // 10]
    print("train set: ", len(train_dataset))
    print("valid set: ", len(val_dataset))
    print("test set: ", len(test_dataset))

    datamodule = LightningDataset(train_dataset, val_dataset, test_dataset,
                                batch_size=8, drop_last=True, num_workers=1)#, num_workers=1
    # print(datamodule)

    #Load Model
    model = GINModule(in_channels=1500, hidden_channels=128, out_channels=2,
                      )

    # Get Trainer
    checkpoint = callbacks.ModelCheckpoint(monitor="val_auroc", save_top_k=1,
                                           mode="max") 
    swa = callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    early_stop = callbacks.EarlyStopping(monitor="val_auroc",
                                         min_delta=0.00, patience=3,
                                         verbose=False, mode="max")
    trainer = L.Trainer(precision=16,
                        strategy="ddp_spawn",
                        # strategy="auto",
                        # strategy="ddp_find_unused_parameters_true",
                        accelerator='gpu',
                        devices=devices,
                        max_epochs=10, log_every_n_steps=5,
                        accumulate_grad_batches=4,
                        # auto_lr_find=True,
                        # limit_train_batches=0.4, limit_val_batches=0.4,
                        # limit_test_batches=0.01,
                        # fast_dev_run=True, num_sanity_val_steps=2,
                        # callbacks=[checkpoint, swa]
                        )

    # Train
    trainer.fit(model, datamodule)  
    # p,l = model.roc()
    # p= np.array(p[-75:])
    # l= np.array(l[-75:])
    # print(l)
    # print(p)
    # # print(type(l[0]))
    # threshold = np.median(p)
    # acc = ( (p>threshold) == l ).mean()
    # # acc = accuracy_score(l,p)
    # print(acc)
    trainer.test(ckpt_path="best", datamodule=datamodule)