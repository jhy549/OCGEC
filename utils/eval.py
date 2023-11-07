from torch.utils.data import DataLoader
import lightning as pl
def evaluate(model, test_set):
    trainer = pl.Trainer()

    trainer.test(model=model, dataloaders=DataLoader(test_set)) 