# The Extended Contextual Graph Markov Model (E-CGMM)
### a.k.a. making CGMM process arbitrary edge features

## Description
Here you will find all you need to replicate the experiments in our code (**please look at previous releases**).

If you happen to use or modify this code, please remember to cite our paper:

Atzeni Daniele, Bacciu Davide, Errica Federico, Micheli Alessio: *Modeling Edge Features with Deep Bayesian Graph Networks*, IJCNN, 2021.

### Usage

This repo builds upon [PyDGN](https://github.com/diningphil/PyDGN), a framework to easily develop and test new DGNs.
See how to construct your dataset and then train your model there.

This repo assumes PyDGN 1.0.3 is used. Compatibility with future versions is not guaranteed.

The evaluation is carried out in two steps:
- Generate the unsupervised graph embeddings
- Apply a classifier on top

We designed two separate experiments to avoid recomputing the embeddings each time. First, use the `config_CGMM_Embedding.yml` config file to create the embeddings,
specifying the folder where to store them in the parameter `embeddings_folder`. Then, use the `config_CGMM_Classifier.yml` config file to launch
the classification experiments.

## Launch Exp:

#### Build dataset and data splits (follow PyDGN tutorial and use the data splits provided there)
For instance:

    pydgn-dataset --config-file DATA_CONFIGS/config_PROTEINS_custom_transform.yml

#### Train the model

    pydgn-train  --config-file MODEL_CONFIGS/config_ECGMM_Embedding.yml 
    pydgn-train  --config-file MODEL_CONFIGS/config_ECGMM_Classifier.yml 
