# LUP-SAC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

You can run the federated learning simulation using the main execution script. Use the `--dataset` and `--attack` arguments to configure your experiment.

### Image Data (MNIST)
Run an experiment on the MNIST dataset against a Label-Flip attack:
```bash
python main.py --dataset mnist --attack label_flip --epochs 100
```

### Tabular/IoT Data (ToN-IoT)
Run an experiment on the ToN-IoT dataset against a ByzMean optimization attack:
```bash
python main.py --dataset ton_iot --attack byzMean --epochs 100
```

---

## Configuration Arguments

The framework is highly customizable. Below are the available command-line arguments you can pass during execution:

### Byzantine Attack Parameters
* `--attack` (str): The Byzantine attack method to simulate. Options include: `random`, `noise`, `label_flip`, `sign_flip`, `byzMean`, `min_max`, `min_sum`, `lie`, `mpfa`. (Default: `label_flip`)
* `--num_byzs` (int): Number of malicious Byzantine nodes. (Default: `25`)
* `--agg_rule` (str): The gradient aggregation rule fallback. (Default: `Mean`)

### Federated Learning Parameters
* `--epochs` (int): Total number of global communication rounds/epochs. (Default: `100`)
* `--num_users` (int): Total number of clients in the federated network. (Default: `50`)
* `--frac` (float): The fraction of clients selected per round. (Default: `1.0`)
* `--local_iter` (int): Number of local training iterations per client. (Default: `1`)
* `--local_bs` (int): Local batch size for client training. (Default: `256`)
* `--lr` (float): Local learning rate. (Default: `0.01`)
* `--momentum` (float): SGD momentum parameter. (Default: `0.9`)

### System & Data Parameters
* `--dataset` (str): Name of the dataset to use (`mnist` or `ton_iot`). (Default: `mnist`)
* `--num_classes` (int): Number of output classes for the dataset. (Default: `10`)
* `--iid` (int): Set to `1` for IID data distribution, or `0` for non-IID. (Default: `0`)
* `--skew` (float): Data skewness parameter for non-IID settings. (Default: `0.5`)
* `--optimizer` (str): Type of local optimizer to use. (Default: `sgd`)
* `--device` (str): Compute device. Set to `cuda:0` for GPU or `cpu`. (Default: `cpu`)
