import unittest

from torchvision import datasets

from src.data_utils import BCC_Dataloader


class TestBCCDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_dir = "data/"  # Replace with a valid test data directory
        self.device = "cpu"
        self.batch_size = 16
        self.num_workers = 0
        self.data_loader = BCC_Dataloader(
            self.data_dir,
            self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_data_split(self):
        # Test if the dataset is split correctly
        train_loader, val_loader, test_loader = self.data_loader.load_data()
        total_samples = (
            len(train_loader.dataset)
            + len(val_loader.dataset)
            + len(test_loader.dataset)
        )
        self.assertEqual(total_samples, len(datasets.ImageFolder(root=self.data_dir)))

    def test_batch_size(self):
        # Test if the data loaders return the correct batch size
        train_loader, val_loader, test_loader = self.data_loader.load_data()
        for loader in [train_loader, val_loader, test_loader]:
            for batch in loader:
                x, _ = batch
                self.assertEqual(x.size(0), self.batch_size)
                break

    def test_transforms(self):
        # Test if the transforms are applied correctly
        train_loader, _, _ = self.data_loader.load_data()
        for batch in train_loader:
            x, _ = batch
            self.assertEqual(x.size(1), 3)  # check number of channels
            self.assertEqual(x.size(2), 224)  # image siye
            self.assertEqual(x.size(3), 224)
            break


if __name__ == "__main__":
    unittest.main()
