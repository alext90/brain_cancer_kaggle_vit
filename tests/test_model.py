import unittest

import torch
from torchvision import models

from src.models import BCC_Model


class TestBCCModel(unittest.TestCase):
    def setUp(self):
        self.num_classes = 3
        self.lr = 1e-4
        self.input_shape = (1, 3, 224, 224)
        self.base_model = models.vit_b_16(pretrained=False)
        self.model = BCC_Model(
            base_model=self.base_model, num_classes=self.num_classes, lr=self.lr
        )

    def test_model_initialization(self):
        # Test if the model initializes correctly
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.model.heads.head.out_features, self.num_classes)

    def test_forward_pass(self):
        # Test if the forward pass works correctly
        dummy_input = torch.randn(self.input_shape)
        output = self.model(dummy_input)
        self.assertEqual(
            output.shape, (1, self.num_classes)
        )  # (batch_size, num_classes)

    def test_freezing_layers(self):
        # Test if the layers are frozen correctly
        for param in self.model.model.parameters():
            self.assertFalse(param.requires_grad)

        for param in self.model.model.heads.head.parameters():
            self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
