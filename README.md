# CellUntangler

### PyTorch implementation

## Overview

This repository contains a PyTorch implementation of CellUntangler.

## Installation

Install Python 3.10+.

1. Clone the repository using `!git clone https://github.com/Ding-Group/CellUntangler`.
2. Install the dependencies found in `requirements.txt` with `pip install -r CellUntangler/requirements.txt` or if the working directory is CellUntangler, `pip install -r requirements.txt.`

## Structure

* `src/` - Source folder.
  * `data/` - Data loading, preprocessing and batching.
  * `celluntangler/` - Model directory. Note that models heavily use inheritance!
  * `visualization/` - Utilities for visualization of latent spaces.
* `README.md` - This manual.
* `LICENSE` - Apache Standard License 2.0.
* `requirements.txt` - Required Python packages.
* `THIRD_PARTY.md` - List of third party software used.
* `celluntangler_hela_tutorial.ipynb` - A tutorial for using CellUntangler on a dataset of HeLa cells.

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

In the `CellUntangler` directory, we provide specific examples of using CellUntangler. These examples separate the cell cycle signals into one latent subspace from the non-cell cycle specific signals in another subspace.
We discuss the case of using one subspace or more than two subspaces below.

Given that the latent space is composed of $k$ different subspaces, $\mathbf{z}=(\mathbf{z}^1,\mathbf{z}^2,\ldots,\mathbf{z}^k)$, we decopmose $\mathbf{x}$ into $k$ components, $\mathbf{x}^1,\mathbf{x}^2,\ldots,\mathbf{x}^k$.
Each component $\mathbf{x}^j$ is used to output the parameters of the posterior distribution $p(\mathbf{x}^j\mid\mathbf{z}^j)$.

Importantly, CellUntangler has two parameters, `component_subspaces` and `component_no_grads`, which are `None` by default. These are the settings in the examples.

When using one latent subspace, two latent subspaces but with a different model, or more than two latent subspaces, `component_subspaces` must be specified.
`component_subspaces` is a dictionary where the keys are the indices of component $\mathbf{x}^j$, and the values are lists of indices of the subspace or subspaces $\mathbf{x}^l$ that should be used to reconstruct the component $\mathbf{x}^j$.
Indexing starts from 0, so both $\mathbf{x}^1$ and $\mathbf{z}^1$ have index 0, $\mathbf{x}^2$ and $\mathbf{z}^2$ have index 1, and so on until $\mathbf{x}^k$ and $\mathbf{z}^k$, which have index $k-1$.

*E.g.*, if we have the latent space `r2, e10`, we might specify `component_subspaces` as `{0: [0], 1: [0, 1]}` which means $\mathbf{z}^1$ should be used to reconstruct $\mathbf{x}^1$ while both $\mathbf{z}^1$ and $\mathbf{z}^2$ should be used to reconstruct $\mathbf{x}^2$.
Alternatively, `{0: [0], 1: [1]}` means $\mathbf{z}^1$ should be used to reconstruct $\mathbf{x}^1$ and $\mathbf{z}^2$ should be used to reconstruct $\mathbf{x}^2$.
For one latent subspace, `component_subspaces` should always be `{0: [0]}`.

For more than two subspaces, `component_subspaces` may look like `{[0]: [0], 1: [1], $\ldots$, k-1: [k-1]}` in which case, we use $\mathbf{z}^j$ to reconstruct $\mathbf{x}^j$ except for $\mathbf{x}^k$ where we use $\mathbf{z}^1,\mathbf{z}^2,\ldots,\mathbf{z}^k$.
We may also specify multiple latent components for other $x^j$ besides $z^k$, such as `{0: [1], 1:[0, 1], $\ldots$, k-1: [0, 1, $\ldots$, k-1]}`. Now, $\mathbf{z^1}$ and $\mathbf{z}^2$ are used to reconstruct $\mathbf{x}^2$.
Importantly, the dictionary should always contain the keys $0, 1, \ldots, k-1$ unless more than one subspace $\mathbf{z}^j$ is used to capture the same component $\mathbf{x}^j$.
In this case, two or more components $x^j$ will be repeated.
The repeated component should only appear once in `component_subspaces`.
This is the only instance in which an index for component $\mathbf{x}^j$ should be omitted.

The subspaces for the same signal should follow after each other.
*E.g.*, if we use `s2` and `e1` to captue the cell cycle and `e10` to capture non-cell cycle specific signals, the latent space should be specified as `s1, e1, e10` or `e1, s1, e10` not `s1, e10, e1`.
The dictionary for this example would look like `{0: [0, 1], 2: [0, 1, 2]` or `{1: [0, 1], 2: [0, 1, 2]}`. As `s1` and `e1` are meant to capture the same signal, the cell cycle, we only list the index of either s1 or e1. We use both `s` and `e1` to reconstruct the component $x^j$ that should capture the cell cycle and `s1`, `e1`, and `e10` to reconstruct the component $\mathbf{x}^j$ which should capture the non-cell cycle specific signals.
