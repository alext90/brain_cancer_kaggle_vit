import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BCC_Dataloader:
    def __init__(
        self, data_dir: str, device: str, batch_size: int = 16, num_workers: int = 7
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.transform = {
            "train": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        }
        self.train_loader, self.val_loader, self.test_loader = self.load_data()

    def load_data(self) -> tuple:
        """
        Load the dataset and split it into training, validation, and test sets.
        Returns:
            tuple: A tuple containing the training, validation, and test data loaders.
        """
        full_data = datasets.ImageFolder(
            root=self.data_dir, transform=self.transform["train"]
        )
        
        # Calculate split
        train_size = int(0.7 * len(full_data))
        val_size = int(0.2 * len(full_data))
        test_size = len(full_data) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_data, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader
