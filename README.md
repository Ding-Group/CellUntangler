# Mixed-curvature Variational Auto-Encoders

### PyTorch implementation

## Overview

This repository contains a PyTorch implementation of CellUntangler.

## Installation

Install Python 3.7+.
To install all dependencies, make sure you have installed [conda](https://docs.conda.io/en/latest/miniconda.html), and run

```bash
make conda
conda activate pt
make download_data
```


## Structure

* `data/` - Data folder. Contains a script necessary for downloading the datasets, and the downloaded data.
* `mathematica/` - Mathematica scripts (various formula derivations, etc).
* `mt/` - Source folder (stands for Master Thesis).
  * `data/` - Data loading, preprocessing, batching, and pre-trained embeddings.
  * `mvae/` - Model directory. Note that models heavily use inheritance!
  * `visualization/` - Utilities for visualization of latent spaces or training statistics.
* `tests/` - (A few) unit tests.
* `README.md` - This manual.
* `LICENSE` - Apache Standard License 2.0.
* `requirements.txt` - Required Python packages.
* `THIRD_PARTY.md` - List of third party software used in this thesis.

## Usage

In the `tutorials` directory, we provide examples of using CellUntangler.
