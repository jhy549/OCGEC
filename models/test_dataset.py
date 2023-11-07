from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .dataset import ModelDataset

def test_ModelDataset():
    test_data_dir = "/home/dorian/dev/GBD/data/MNIST/Jumbo/models0"
    test_dataset = ModelDataset(data_dir=test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1)
    for idx, g in enumerate(test_loader):
        assert isinstance(g, Data), "returned not a pyg Data object."
        break

from .dataset import ModelToGraphDataModule
    
def test_ModelToGraphDataModule():
    test_data_dir = "/home/dorian/dev/GBD/data/MNIST/Jumbo/models0"
    dataModule = ModelToGraphDataModule(test_data_dir)
    dataModule.setup(stage="fit")
    for idx, g in enumerate(dataModule.train_dataloader()):
        assert isinstance(g, Data), "returned not a pyg Data object."
        break