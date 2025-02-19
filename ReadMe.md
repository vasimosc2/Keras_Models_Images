# CIFAR-10 Model Training

This project trains 6 different deep learning models on the CIFAR-10 dataset and evaluates them based on accuracy, precision, and recall. The models include:

- Simple CNN
- Deeper CNN
- ResNet CNN
- CNN with Dropout
- CNN with Batch Normalization
- CNN with Global Average Pooling

We train and evaluate each of the models saving their:
- Accuary
- Precission
- Recal
- Model Size
- Number of FLOPs (floating-point operations in the model)

## Setup

To set up the environment, install the required dependencies:

python3.11 -m venv venv
or
python -m venv venv

```bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

pip freeze