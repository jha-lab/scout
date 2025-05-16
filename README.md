# scout
[ACM Transactions on Computing for Heathcare, 2023] ScouT:  Synthetic Counterfactuals via Spatiotemporal Transformers for Actionable Healthcare


![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)
![CUDA](https://img.shields.io/badge/cuda-v11.1.1-76b900)
![License](https://img.shields.io/badge/license-Clear%20BSD-green)

This is the official repository for the paper [SCouT: Synthetic Counterfactuals via Spatiotemporal Transformers for Actionable Healthcare](https://arxiv.org/abs/2207.04208).

![SCouT Architecture](scout.png)


## Environment setup

The following shell script creates an anaconda environment called "scout" and installs all the required packages. 
```shell
source env_setup.sh
```

## Dataset Setup

Data is setup as a N x T x D numpy matrix `data.npy` where:
    -- N is units
    -- T is total time interval
    -- D is the total covariates

Along with the data matrix provide a binary matrix `mask.npy` of shape N x T indicating missing measurement by 1

See ``/synthetic_data/synethtic_data_noise_1'' for an example

## Experiment Config Setup

Experiment configurations in SCouT are defined using YAML files. These config files specify the model architecture, modeling parameters, and training settings. An example configuration is provided in `experiment_config/synthetic.yaml`.

### Architecture Parameters
- `feature_dim`: Number of covariates in the dataset
- `cont_dim`: Number of continuous covariates
- `discrete_dim`: Number of discrete covariates
- `hidden_size`: Hidden size of the transformer model
- `n_layers`: Number of layers in the SCouT model
- `n_heads`: Number of attention heads in the transformer

### Modeling Parameters
- `K`: Number of donor units to use
- `seq_range`: Total units (donor + target)
- `pre_int_len`: Length of pre-intervention interval to model
- `post_int_len`: Length of post-intervention interval to model
- `time_range`: Total time interval period
- `interv_time`: Time point of intervention
- `target_id`: ID of the target unit
- `lowrank`: Whether to use low-rank approximation
- `rank`: Rank of the low-rank approximation

### Training Parameters
- `batch_size`: Batch size for training
- `lr`: Learning rate
- `weight_decay`: Weight decay for regularization
- `warmup_steps`: Number of warmup steps for learning rate scheduler

## SCouT Library

The core of the SCouT library is the `SCOUT` class which handles model initialization, training, and prediction. Here's how to use the key functionalities:

### Initialization
```python
scout = SCOUT(config_path="./experiment_config/synthetic.yaml",
              op_dir="./logs/",
              random_seed=42,
              datapath="./synthetic_data/synthetic_data_noise_1/",
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

### Training
The `fit()` method handles both pretraining on donor units and finetuning on the target unit:
```python
scout = scout.fit(pretrain=True, pretrain_iters=10000, finetune_iters=1000)
```

### Prediction
After training, `.predict()` generates counterfactual predictions for post-intervention period:
```python
counterfactual_predictions = scout.predict()
```

### Attention Analysis
Extract donor attention weights to analyze which donor units influenced the predictions:
```python
attention_weights = scout.return_attention(interv_time=1600)
```

### Loading from Checkpoint
Load a previously trained model:
```python
scout.load_model_from_checkpoint("./logs/finetune/checkpoint.pt")
```

## Demo

The repository includes `demo.ipynb`, a Jupyter notebook that demonstrates SCouT's core functionality using synthetic data. The notebook shows:

- Setting up model configuration and paths
- Initializing and training the SCouT model
- Generating counterfactual predictions
- Visualizing results comparing observed data, ground truth, and counterfactual predictions

## Third Party Code

We are grateful to huggingface for their [transformers](https://github.com/huggingface/transformers) library, which our SCouT model builds upon. Our architecture leverages their BERT implementation as the foundation for our spatiotemporal transformer models.

## Citations

Please cite the paper and star this repo if you find it useful, thanks! Feel free to contact bdedhia@princeton.edu or open an issue if you have any questions. 
Cite our work using the following bitex entry:
```bibtex
@misc{dedhia2022scoutsyntheticcounterfactualsspatiotemporal,
      title={SCouT: Synthetic Counterfactuals via Spatiotemporal Transformers for Actionable Healthcare}, 
      author={Bhishma Dedhia and Roshini Balasubramanian and Niraj K. Jha},
      year={2022},
      eprint={2207.04208},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2207.04208}, 
}
```













