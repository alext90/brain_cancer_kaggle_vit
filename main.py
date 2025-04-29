import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from src.models import BCC_mobilenet_v2, BCC_ViT, SimpleCNN
from src.data_utils import BCC_Dataloader
from src.eval import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    for model, name in [
        (SimpleCNN(num_classes=NUM_CLASSES, lr=LEARNING_RATE), "SimpleCNN"),
        (BCC_mobilenet_v2(num_classes=NUM_CLASSES, lr=LEARNING_RATE), "MobileNetV2"),
        (BCC_ViT(num_classes=NUM_CLASSES, lr=LEARNING_RATE), "ViT"),
    ]:
        logger.info(f"Evaluate non-fine-tuned models:")
        evaluate_model(model, test_loader, logger)

        logger.info(f"Training {name} model...")
        trainer = pl.Trainer(
            accelerator="mps",
            max_epochs=MAX_EPOCHS,
            logger=CSVLogger("logs", name=name),
            callbacks=[early_stopping],
        )
        trainer.fit(model, train_loader, val_loader)

        logger.info(f"Evaluating {name} model...")
        evaluate_model(model, test_loader, logger)

        torch.save(model.state_dict(), name + ".pth")
        logger.info(f"Model {name} saved as {name}.pth")
