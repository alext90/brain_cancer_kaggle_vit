import torch
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping


from src.data_utils import BCC_Dataloader
from src.model import BCC_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/"

BATCH_SIZE = 16
NUM_WORKERS = 7

NUM_CLASSES = 3
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4


if __name__ == "__main__":
    BCC_dataloader = BCC_Dataloader(
        DATA_DIR, DEVICE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    train_loader, val_loader, test_loader = BCC_dataloader.load_data()

    base_model = models.vgg19(pretrained=True)
    model = BCC_Model(base_model=base_model, num_classes=NUM_CLASSES, lr=LEARNING_RATE)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="mps",
        max_epochs=MAX_EPOCHS,
        logger=CSVLogger("logs", name="BCC_Model"),
        callbacks=[early_stopping],
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
