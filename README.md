# GCN with Global DP

## Description

Code accompanying "Privacy-Preserving Graph Convolutional Networks for Text Classification" paper, run with `python GCN.py`. Configurations to be specified in `settings.py`.

## Setup

```bash
$ sudo apt-get install python3-dev
```

```bash
$ pip install torch==1.6.0+cpu --find-links https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.txt
```

## Run

Experiments can be run with `./run.sh`, by default specified for the Cora dataset. Other datasets and parameters can be modified in this bash script, including alternating between private and non-private settings.

An additional script is included for the Pokec dataset, `run_pokec.sh`, due to additional download and preprocessing steps.

By default, results and other output information will be saved in `./out`, while datasets saved to `./data`. This can be changed in `settings.py`, in the `out_dir` and `root_dir` variables, respectively.
