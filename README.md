# CellUntangler

### PyTorch implementation

## Overview

This repository contains a PyTorch implementation of CellUntangler.

## Installation

Install Python 3.7+.

1. Clone the repository using `!git clone https://github.com/Ding-Group/CellUntangler`.
2. Install the dependencies found in `requirements.txt` with `pip install -r CellUntangler/requirements.txt` or if the working directory is CellUntangler, `pip install -r requirements.txt.`

## Structure

* `mathematica/` - Mathematica scripts (various formula derivations, etc).
* `mt/` - Source folder.
  * `data/` - Data loading, preprocessing, batching, and pre-trained embeddings.
  * `mvae/` - Model directory. Note that models heavily use inheritance!
  * `visualization/` - Utilities for visualization of latent spaces or training statistics.
* `tests/` - (A few) unit tests.
* `README.md` - This manual.
* `LICENSE` - Apache Standard License 2.0.
* `requirements.txt` - Required Python packages.
* `THIRD_PARTY.md` - List of third party software used in this thesis.
* celluntangler_hela_tutorial.ipynb - A tutorial for using CellUntangler on a dataset of HeLa cells.

## Usage

There are different subspaces available for the latent space in CellUntangler which are:

* Euclidean, specified as e
* Spherical space, specified as s
* Stereographically projected sphere, specified as d
* Hyperbolic with the Lorentz model, specified as h
* Poincare, specified as p
* Hyperbolic with the Lorentz model and the rotated hyperbolic wrapped normal distribution (RoWN), specified as r
* Universal, specified as u

To specifiy the dimension of the space, an integer is placed after the letter. *E.g.*, `e2` denotes the Euclidean space with a dimension of 2.

The different subspaces are separated by `,` or `, `. *E.g.*, `r2,e10` or `r2, e10` denotes a latent space with two subspaces where the first subspace is the hyperbolic space with the RoWN and a dimension of 2 and the second subspace is the Euclidean subspace with a dimension of 10.

In the `CellUntangler` directory, we provide specific examples of using CellUntangler.
