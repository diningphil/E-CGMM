# The Extended Contextual Graph Markov Model (E-CGMM)
### a.k.a. making CGMM process arbitrary edge features

## Description
This library is a clone of our [PyDGN](https://github.com/diningphil/PyDGN) repository (pre-release of version 0.4.1) for easy experimentation and reproducibility with Deep Graph Networks.
Here you will find all you need to replicate the experiments in our code.

If you happen to use or modify this code, please remember to cite our paper:

[Atzeni Daniele, Bacciu Davide, Errica Federico, Micheli Alessio: *Modeling Edge Features with Deep Bayesian Graph Networks*, IJCNN, 2021.

## Installation:
(We assume **git** and **Miniconda/Anaconda** are installed)

First, make sure gcc 5.2.0 is installed: ``conda install -c anaconda libgcc=5.2.0``. Then, ``echo $LD_LIBRARY_PATH`` should always contain ``:/home/[your user name]/miniconda3/lib``. Then run from your terminal the following command:

    source setup/install.sh [<your_cuda_version>]

Where `<your_cuda_version>` is an optional argument that can be either `cpu`, `cu92`, `cu101`, `cu102`, `cu110` for Pytorch 1.7.0. If you do not provide a cuda version, the script will default to `cpu`. The script will create a virtual environment named `pydgn`, with all the required packages needed to run our code. **Important:** do NOT run this command using `bash` instead of `source`!

Remember that [PyTorch MacOS Binaries dont support CUDA, install from source if CUDA is needed](https://pytorch.org/get-started/locally/)

## Usage:

### Preprocess your dataset (see also Wiki)
    python build_dataset.py --config-file [your data config file]

### Launch an experiment in debug mode (see also Wiki)
    python launch_experiment.py --config-file [your exp. config file] --splits-folder [the splits MAIN folder] --data-splits [the splits file] --data-root [root folder of your data] --dataset-name [name of the dataset] --dataset-class [class that handles the dataset] --max-cpus [max cpu parallelism] --max-gpus [max gpu parallelism] --gpus-per-task [how many gpus to allocate for each job] --final-training-runs [how many final runs when evaluating on test. Results are averaged] --result-folder [folder where to store results]

To debug your code it is useful to add `--debug` to the command above. Notice, however, that the CLI will not work as expected here, as code will be executed sequentially. After debugging, if you need sequential execution, you can use `--max-cpus 1 --max-gpus 1 --gpus-per-task [0/1]` without the `--debug` option.  

### How to replicate the experiments
- First, you should download and preprocess the datasets (see guide above) using the config files in `ECGMM_DATA_CONFIGS`.

- Then, you are ready to launch the experiments. In `ECGMM_MODEL_CONFIGS` you will find two types of experiment configuration files for each dataset and model. The first to be used should be the one that does not contain the word "classifier" in its name. For example, the `config_ECGMM_NCI1.yml` config file will launch an experiment that will store graph embeddings in the `/data/errica/ECGMM_EMBEDDINGS` folder (you should change this according to your own filesystem). The `config_ECGMMClassifier_NCI1.yml` config file, instead, will launch experiments using the graph embeddings stored in the `/data/errica/ECGMM_EMBEDDINGS` folder (again, please change the reference in the config file).

- If you have any inquiry or something does not work as expected (this repo is basically PyDGN but stripped of all non-relevant content), please contact us via email. We will do our best to help you.

### How to stop your experiment
If you are in debug mode, a `CTRL-C` will be enough. Otherwise, you have to use `ray stop` to kill all processes.

## Contributing
**This research software is provided as-is**.
If you find a bug, please open an issue to report it, and we will do our best to solve it. For generic/technical questions, please email us rather than opening an issue.

## License:
E-CGMM is GPL 3.0 licensed, as written in the LICENSE file.

## Troubleshooting

If you get errors like ``/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found``:
* make sure gcc 5.2.0 is installed: ``conda install -c anaconda libgcc=5.2.0``
* ``echo $LD_LIBRARY_PATH`` should contain ``:/home/[your user name]/[your anaconda/miniconda folder name]/lib``
* after checking the above points, you can reinstall everything with pip using the ``--no-cache-dir`` option
