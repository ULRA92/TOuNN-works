import jax.numpy as jnp
import numpy as np


# ------- FOURIER LENGTH SCALE (For Generating Features) -------- #
def computeFourierMap(mesh, fourierMap):
    """
  Compute a Fourier-based coordinate mapping for material distribution.

  Used for modifying thermal conductivity fields using a periodic mapping.
  """
    coordnMapSize = (mesh.ndim, fourierMap['numTerms'])
    freqSign = np.random.choice([-1., 1.], coordnMapSize)
    stdUniform = np.random.uniform(0., 1., coordnMapSize)
    wmin = 1. / (2 * fourierMap['maxRadius'])
    wmax = 1. / (2 * fourierMap['minRadius'])  # Frequency ~ 1/R
    wu = wmin + (wmax - wmin) * stdUniform
    coordnMap = np.einsum('ij,ij->ij', freqSign, wu)
    return coordnMap


# ----------------- #
def applyFourierMap(xy, fourierMap):
    """
  Apply Fourier mapping to coordinate points.

  Used for modifying spatial patterns of thermal conductivity fields.
  """
    if fourierMap['isOn']:
        c = jnp.cos(2 * np.pi * jnp.einsum('ij,jk->ik', xy, fourierMap['map']))
        s = jnp.sin(2 * np.pi * jnp.einsum('ij,jk->ik', xy, fourierMap['map']))
        xy = jnp.concatenate((c, s), axis=1)
    return xy


# ------- THERMAL CONDUCTIVITY PROJECTION -------- #
def applyThermalConductivityProjection(k, conductivityProj):
    """
  Apply projection to the thermal conductivity field.

  - Used to sharpen transitions in high/low conductivity regions.
  - Similar to density projection but applied to `k(x)` instead of `rho(x)`.
  """
    if conductivityProj['isOn']:
        b = conductivityProj['sharpness']
        nmr = jnp.tanh(0.5 * b) + jnp.tanh(b * (k - 0.5))
        k = 0.5 * nmr / np.tanh(0.5 * b)
    return k


# ------- SYMMETRY -------- #
def applySymmetry(x, symMap):
    """
  Apply symmetry transformations to coordinate points.

  Used to enforce symmetric thermal distributions if required.
  """
    if symMap['YAxis']['isOn']:
        xv = x[:, 0].at[:].set(symMap['YAxis']['midPt'] + jnp.abs(x[:, 0] - symMap['YAxis']['midPt']))
    else:
        xv = x[:, 0]
    if symMap['XAxis']['isOn']:
        yv = x[:, 1].at[:].set(symMap['XAxis']['midPt'] + jnp.abs(x[:, 1] - symMap['XAxis']['midPt']))
    else:
        yv = x[:, 1]
    x = jnp.stack((xv, yv)).T
    return x


# -------------------------- #
def applyRotationalSymmetry(xyCoordn, rotationalSymmetry):
    """
  Apply rotational symmetry constraints to coordinate points.

  Used to ensure rotationally symmetric thermal distributions.
  """
    if rotationalSymmetry['isOn']:
        dx = xyCoordn[:, 0] - rotationalSymmetry['centerCoordn'][0]
        dy = xyCoordn[:, 1] - rotationalSymmetry['centerCoordn'][1]
        radius = jnp.sqrt((dx) ** 2 + (dy) ** 2)
        angle = jnp.arctan2(dy, dx)
        correctedAngle = jnp.remainder(angle, np.pi * rotationalSymmetry['sectorAngleDeg'] / 180.)
        x, y = radius * jnp.cos(correctedAngle), radius * jnp.sin(correctedAngle)
        xyCoordn = jnp.stack((x, y)).T
    return xyCoordn


# -------------------------- #
def applyExtrusion(xy, extrusion):
    """
  Apply extrusion constraints to coordinate points.

  Used to model periodic patterns in thermal distributions.
  """
    if extrusion['X']['isOn']:
        xv = xy[:, 0] % extrusion['X']['delta']
    else:
        xv = xy[:, 0]
    if extrusion['Y']['isOn']:
        yv = xy[:, 1] % extrusion['Y']['delta']
    else:
        yv = xy[:, 1]
    x = jnp.stack((xv, yv)).T
    return x
