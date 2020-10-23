# GCN-VAE

This is a TensorFlow implementation of the uncertainty-GNN model as described in our paper:
 
[Xujiang Zhao](https://zxj32.github.io/), Feng Chen, Shu Hu, Jin-Hee Cho [Deep Learning based Scalable Inference of Uncertain Opinions](https://zxj32.github.io/data/NIPS2020_Uncertainty.pdf), NIPS 2020 (##Spotlight##)

A multi-source uncertainty framework of GNN that reflecting various types of uncertainties in both deep learning and belief/evidence theory domains for node classification predictions.

![Uncertainty Framework](framework.png)


## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/zxj32/uncertainty-GNN
   cd uncertainty-GNN
   ```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
   ```sh
   pip install -r requirements.txt 
   ```

## Requirements
* TensorFlow (1.0 or later)
* python 3.6.9
* networkx
* scikit-learn
* scipy
* numpy
* pickle

## Run the demo

```bash
python S-GCN.py
```

## Dataset

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node)
* an N by E binary label matrix (E is the number of classes).

Have a look at the `load_data()` function in `input_data.py` for an example.

Our date processing is same as [GCN code](https://github.com/tkipf/gcn)

In this example, we load epinioin data. The original datasets can be found here (epinion data): http://www.trustlet.org/downloaded and here (traffic data): http://inrix.com/publicsector.asp


## Models

You can choose between the following models: 
* `GCN-based`: gcn_vae_epinion.py
* `GCN-VAE-based`: gcn_epinion.py

## Question

If you have any question, please feel free to contact me. Email is good for me. 

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{xujiang2020uncertainties,
  title={Uncertainty Aware Semi-Supervised Learning on Graph Data},
  author={Zhao, Xujiang and Chen, Feng and Hu, Shu and Cho, Jin-Hee},
  booktitle={Advances in neural information processing systems},
  year={2020}
}
```

