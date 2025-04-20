# MRI Brain Cancer Classification

This project classifies MRI images of brain cancer into different categories using deep learning. It leverages PyTorch and PyTorch Lightning for model training and evaluation.

## Dataset

The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset/data). Ensure the dataset is downloaded and placed in the `data/` directory.

### Running the Project
To train and evaluate the model, use the following command:

This will:

1. Load and split the dataset.
2. Train the model using the VisionTransformer architecture.
3. Evaluate the model on the test set.

### Tests
Unit tests are included to ensure the correctness of the code. To run the tests, use the following command:

`pytest tests/`  

This will execute all test cases in the tests directory and provide a summary of the results.