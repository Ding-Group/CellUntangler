import geomstats.visualization as visualization
import math as math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os as os
import scanpy as sc

from matplotlib.patches import Circle
from scipy.signal import savgol_filter

from .helpers import lorentz_to_poincare


tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
COLOR_NAMES = tableau_colors + list(mcolors.CSS4_COLORS.keys())


def visualize_poincare_from_lorentz(embeddings,
                                    desired_obs_all,
                                    embedding_type="discrete",
                                    curvature=-1.0,
                                    cmap=plt.cm.viridis,
                                    grid_lines=False,
                                    origin=(0, 0),
                                    cat_colors=None,
                                    c_bar_label="",
                                    bbox_to_anchor=None,
                                    s=None):
  poincare_coordinates = lorentz_to_poincare(embeddings, curvature)
  fig = visualize_poincare(poincare_coordinates, desired_obs_all, curvature, embedding_type, cmap, grid_lines, cat_colors=cat_colors, c_bar_label=c_bar_label, bbox_to_anchor=bbox_to_anchor, s=s)
  return fig

def visualize_poincare(poincare_coordinates,
                       desired_obs_all,
                       curvature=-1.0,
                       embedding_type="discrete",
                       cmap=plt.cm.viridis,
                       grid_lines=False,
                       origin=(0, 0),
                       cat_colors=None,
                       c_bar_label="",
                       bbox_to_anchor=None,
                       s=None):

  x_p_1 = poincare_coordinates[:, 0]
  x_p_2 = poincare_coordinates[:, 1]
  circle = visualization.PoincareDisk(coords_type="ball")
  print(round(curvature, 1))
  circle = Circle(origin, radius=1/math.sqrt(abs(curvature)), color='black', fill=False)
  print(round(1/math.sqrt(abs(curvature)),1))
  # circle.set_origin((0.5, 0.4))
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.axes.xaxis.set_visible(grid_lines)
  ax.axes.yaxis.set_visible(grid_lines)
  # ax.grid(grid_lines, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

  ax.set_xlim((-1/math.sqrt(abs(curvature)), 1/math.sqrt(abs(curvature))))
  ax.set_ylim((-1/math.sqrt(abs(curvature)), 1/math.sqrt(abs(curvature))))

  # circle.set_ax(ax)
  # circle.draw(ax=ax)
  ax.add_artist(circle)

  if embedding_type == "discrete":
    # categories = np.unique(desired_obs_all)
    categories = desired_obs_all.cat.categories
    print(categories)
    if cat_colors is None:
      cat_colors = COLOR_NAMES[0:len(categories)]
    for i, cat in enumerate(categories):
      cat_indices = desired_obs_all == cat
      if s:
        ax.scatter(x_p_1[cat_indices], x_p_2[cat_indices], label=cat, color=cat_colors[i], s=s)
      else:
        ax.scatter(x_p_1[cat_indices], x_p_2[cat_indices], label=cat, color=cat_colors[i])
  else:
    scatter = ax.scatter(x_p_1, x_p_2, c=desired_obs_all, cmap=cmap)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(c_bar_label)
  if bbox_to_anchor is not None:
    ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor)
  else:
    ax.legend()
  # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
  # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
  return fig

def compute_umap(adata,
                  l_neighbors,
                  color,
                  embeddings_key,
                  n_pcs=None,
                  use_original_umap=False,
                  save_figure=False,
                  additional_save_information=[],
                  palette=None,
                  title=""):
  """
  l_eighbors: The number of neighbors to use when creating the neighborhood map.
  n_pcs: Dimension to use when computing neighborhood map.
  color: A list of colors to display the UMAP in.
  embeddings_key: The key to use to get the embeddings to use to compute the neighborhood map.
  save_figure: True to save the figure. False otherwie.
  additional_save_information: A list of strings to include in the save name for the figure.
  palette: A palette specifying the color for the observition.
  title: A title for the figure.
  """
  if n_pcs is None:
      n_pcs = adata.obsm[embeddings_key].shape[1]
  if not use_original_umap:
    sc.pp.neighbors(adata, n_neighbors=l_neighbors, n_pcs=n_pcs, use_rep=embeddings_key)
    sc.tl.umap(adata)

  save = None
  if save_figure:
    information = [str(n_pcs),
                      embeddings_key,
                      str(l_neighbors),
                    "original" if use_original_umap else "computed"]+additional_save_information
    save = "_" + "_".join(information) + ",.png"
  
  sc.pl.umap(adata, color=color, save=save, palette=palette, title=title)


def plot_gene_change(ordered_adata, phase, gene_name, colors=None, labels=None, save_path=None, ccPhase_palette=None, layer=""):
    if layer:
        gene_expression = ordered_adata[:, gene_name].layers[layer].squeeze(-1)
    else:
        gene_expression = ordered_adata[:, gene_name].X.squeeze(-1)
    print(gene_expression)
    if labels is not None:
        categories = labels.cat.categories

        for i, cat in enumerate(categories):
            cat_indices = labels.values == cat

            plt.vlines(np.where(cat_indices==True),
                       np.zeros(np.sum(cat_indices==True)),
                       gene_expression[cat_indices],
                       color=colors[cat_indices],
                       label=cat)
    else:
        plt.vlines(np.arange(0, len(ordered_adata)), np.zeros(len(ordered_adata)), gene_expression, colors=colors)

    if ccPhase_palette:
        color = ccPhase_palette[phase.replace("_", ".")]
    else:
        color = 'r'

    plt.plot(savgol_filter(gene_expression,70,2), color=color)

    plt.ylabel(f'${gene_name}$ expression')
    plt.xlabel('Ordering along pseudotime')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.21, 1))
    plt.tight_layout()
    # plt.savefig(os.path.join(save_path, f'savgol_{gene_name}_{phase}_ordered_x_mb_normalize_log_italicize.png'))
    plt.savefig(os.path.join(save_path, f'savgol_{gene_name}_{phase}_ordered_normalize_log.png'))
    plt.show()
    plt.close()
    # sc.pp.pca(adata)
    # sc.pl.pca(adata, color=['CCNE2','ccPhase'])