# Seq-2-Seq_Sp2EN

This repository contains a PyTorch implementation of a sequence-to-sequence model for translating English to Spanish.

## Directory Structure

- `config.yaml`: Configuration file containing hyperparameters and settings for the model.
- `nets/`: Directory containing the sequence-to-sequence neural network architecture.
- `Sqe2seq_layers=2_loss.png`: Image showing the loss curve during training for a 2-layer sequence-to-sequence model.
- `utils/`: Directory containing utility functions and helper scripts.
- `dataloaders/`: Directory containing data loading and preprocessing scripts.
- `README.md`: This readme file providing an overview of the project.
- `test.py`: Script for testing the trained model's translation performance.
- `deeplearning/`: Directory containing deep learning-related code and utilities.
- `Sqe2seq_layers=2_Accuracy.png`: Image showing the translation accuracy curve during training for a 2-layer sequence-to-sequence model.
- `train.py`: Script for training the sequence-to-sequence model.

## Getting Started

1. Install the required dependencies by running: `pip install -r requirements.txt`
2. Set up your dataset and adjust `config.yaml` with appropriate settings.
3. Train the model using `train.py`.
4. Evaluate the model's translation quality using `test.py`.

## Usage

- Modify `config.yaml` to adjust hyperparameters, data paths, and training settings.
- The `nets/` directory contains the sequence-to-sequence model architecture.
- The `utils/` directory includes helpful utility functions.
- The `dataloaders/` directory contains data preprocessing code.
- `train.py` is used for model training.
- `test.py` is used for evaluating the translation performance.

## Acknowledgements

This project is inspired by the field of neural machine translation and utilizes the power of the PyTorch library.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
