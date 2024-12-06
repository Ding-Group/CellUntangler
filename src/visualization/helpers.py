import math as math
import numpy as np
import os as os


def split_embeddings(model_name, embeddings):
  """
  Splits the concatenated embeddings into the latent subspaces.
  Returns a dictionary where the keys are of the form componenti_latent space
   and the values are the embeddings for that latent space.
  E.g., for r2, e10, a dictionary with the keys component1_e2 and component2_e10 would be returned.
  The value for component1_r2 would be the hyperbolic embeddings.
  The value for component_e10 would be the Euclidean embeddings.
  """
  component_embeddings = {}
  components = model_name.split(",")
  components = [c.strip() for c in components]
  start = 0
  for i, c in enumerate(components):
    latent_space = c[0]
    dim = int(c[1:])
    if latent_space == "s":
      dim = dim + 1
    elif latent_space == "h":
      dim = dim + 1
    elif latent_space == "r":
      dim = dim + 1
    print(c, dim)
    component_embeddings[f"component{i+1}_{c}"]=embeddings[:,start:start+dim]
    start = start + dim
  return component_embeddings


def make_dir(path, dir_name):
  """
  Makes a directory if it does not exist.
  """
  if not os.path.exists(os.path.join(path, dir_name)):
      os.mkdir(os.path.join(path, dir_name))


def get_mvae_pseudotime(embeddings, curvature, origin, rotation_angle, flip=False, new_origin=None):
    """
    Gets the pseudotime of hyperbolic embeddings .
    Requires that the embeddings are in the hyperbolic space with dimension 2 in the Lorentz model with some curvature.
    After, projects the Lorentz coordinates to Poincare coordinates.
    If new_origin is not None, the origin is translated from (0, 0) to the new origin.
    The pseudotime is then measured as being the angle from the origin.
    Optionally, the cells can be rotated by the rotation_angle.
    They may also be flipped so that cells assigned a pseudotime of 0 get a pseusotime of 360 and cells assigned a pseudotime
     of 360 get assigned a pseudotime of 0.
    The pseudotime is in degrees.
    """
    poincare_coordinates=lorentz_to_poincare(embeddings, curvature)
    # translated_poincare_coordinates=poincare_translation(np.array(origin),poincare_coordinates,curvature)
    # rotated_poincare_coordinates=rotate_coordinates(poincare_coordinates,
                                                    # math.radians(rotation_angle),
                                                    # origin,
                                                    # 1)
    if new_origin:
        poincare_coordinates = poincare_translation(np.array(new_origin), poincare_coordinates, curvature)
    mvae_pseudotime = get_degrees_angles(poincare_coordinates, origin)
    rotated_mvae_pseudotime = mvae_pseudotime + rotation_angle
    greater_than_360 = rotated_mvae_pseudotime > 360
    less_than_0 = rotated_mvae_pseudotime < 0
    rotated_mvae_pseudotime[greater_than_360] = rotated_mvae_pseudotime[greater_than_360] - 360
    rotated_mvae_pseudotime[less_than_0] = rotated_mvae_pseudotime[less_than_0] + 360
    if flip:
      rotated_mvae_pseudotime=360-rotated_mvae_pseudotime
    # else:
      # mvae_pseudotime=get_degrees_angles(rotated_poincare_coordinates,origin)
    # return rotated_poincare_coordinates, mvae_pseudotime
    return poincare_coordinates, rotated_mvae_pseudotime


def poincare_translation(v, x, curvature):
    """
    Computes the translation of x when we move v to the center.
    Hence, the translation of u with -u should be the origin.
    """
    abs_curvature = abs(curvature)
    xsq = (x ** 2).sum(axis=1)
    vsq = (v ** 2).sum()
    xv = (x * v).sum(axis=1)
    a = np.matmul((abs_curvature * xsq + 2 * abs_curvature * xv + 1).reshape(-1, 1),
                  v.reshape(1, -1)) + (1 - abs_curvature * vsq) * x
    b = (abs_curvature**2) * xsq * vsq + 2 * abs_curvature * xv + 1

    return np.dot(np.diag(1. / b), a)


def get_degrees_angles(coordinates, origin=(0, 0)):
  """
  Gets the angle of each of the coordinates from the given origin.
  Returns the angles in degrees.
  """
  x_vals = coordinates[:, 0]-origin[0]
  y_vals = coordinates[:, 1]-origin[1]
  # print(x_vals)
  # print(y_vals)
  angles = np.zeros(coordinates.shape[0])
  for i in range(coordinates.shape[0]):
    angles[i] = math.atan2(y_vals[i], x_vals[i])
    angles[i] = math.degrees(angles[i])
    if angles[i] < 0:
      angles[i] = 360 + angles[i]
    angles[i] = round(angles[i], 0)
  return angles


def lorentz_to_poincare(embeddings, curvature=-1.0):
  """
  Converts lorentz coordinates to coordinates on the Poincare ball.
  x: An array of x coordinates.
  y: An array of y coordinates.
  z: An array of z coordinates.

  Returns the coordinates on the Poincare ball, x_p_1 and x_p_2.
  """
  # curvature = math.sqrt(abs(curvature))
  # x_p_1 = y / (1 + x)
  # x_p_2 = z / (1 + x)
  # x_p_1 = (1/curvature * y) / (1/curvature + x)
  # x_p_2 = (1/curvature * z) / (1/curvature + x)
  return embeddings[:, 1:] / (1 + math.sqrt(abs(curvature)) * embeddings[:, 0:1])
  # x = embeddings[:, 0]
  # y = embeddings[:, 1]
  # z = embeddings[:, 2]
  # x_p_1 = y / (1 + math.sqrt(abs(curvature)) * x)
  # x_p_2 = z / (1 + math.sqrt(abs(curvature)) * x)
  # return x_p_1, x_p_2