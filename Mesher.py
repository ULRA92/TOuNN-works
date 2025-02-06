import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Helvetica','Times New Roman'
import jax.numpy as jnp
import jax

class RectangularGridMesher:
    # --------------------------#
    def __init__(self, ndim, nelx, nely, elemSize, bcSettings, physics='thermal'):
        """
        Initializes a structured grid mesh for heat transfer or structural analysis.

        - ndim: Number of spatial dimensions.
        - nelx, nely: Number of elements along x and y axes.
        - elemSize: Tuple (dx, dy) defining element size.
        - bcSettings: Dictionary containing boundary condition settings.
        - physics: 'thermal' (default) or 'structural'.
        """
        self.meshType = 'gridMesh'
        self.ndim = ndim
        self.nelx = nelx
        self.nely = nely
        self.elemSize = elemSize
        self.bcSettings = bcSettings
        self.physics = physics

        self.numElems = self.nelx * self.nely
        self.elemArea = self.elemSize[0] * self.elemSize[1] * jnp.ones((self.numElems))
        self.totalMeshArea = jnp.sum(self.elemArea)
        self.numNodes = (self.nelx + 1) * (self.nely + 1)
        self.nodesPerElem = 4  # Quadrilateral elements

        # Degrees of freedom per node: 1 for thermal, 2 for structural
        self.dofsPerNode = 1 if physics == 'thermal' else 2
        self.ndof = self.bcSettings['dofsPerNode'] * self.numNodes

        # Generate mesh structure
        self.edofMat, self.nodeIdx, self.elemNodes, self.nodeXY, self.bb = self.getMeshStructure()
        self.elemCenters = self.generatePoints()
        self.processBoundaryCondition()

        self.fig, self.ax = plt.subplots()

        # Bounding box for the mesh
        self.bb = {
            'xmax': self.nelx * self.elemSize[0],
            'xmin': 0.,
            'ymax': self.nely * self.elemSize[1],
            'ymin': 0.
        }

    # --------------------------#
    def getMeshStructure(self):
        """
        Generates element-node connectivity and other mesh structure data.
        """
        n = self.bcSettings['dofsPerNode'] * self.nodesPerElem
        edofMat = np.zeros((self.nelx * self.nely, n), dtype=int)

        if self.bcSettings['dofsPerNode'] == 1:  # Thermal case
            nodenrs = np.reshape(np.arange(0, self.ndof), (1 + self.nelx, 1 + self.nely)).T
            edofVec = np.reshape(nodenrs[0:-1, 0:-1] + 1, self.numElems, 'F')
            edofMat = np.tile(edofVec, (4, 1)).T + np.tile(
              np.array([0, self.nely + 1, self.nely, -1]).reshape(4, 1), self.numElems
            ).T

        iK = tuple(np.kron(edofMat, np.ones((n, 1))).flatten().astype(int))
        jK = tuple(np.kron(edofMat, np.ones((1, n))).flatten().astype(int))
        nodeIdx = (iK, jK)

        elemNodes = np.zeros((self.numElems, self.nodesPerElem))
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                elemNodes[el, :] = np.array([n1 + 1, n2 + 1, n2, n1])

        nodeXY = np.zeros((self.numNodes, 2))
        ctr = 0
        for i in range(self.nelx + 1):
            for j in range(self.nely + 1):
                nodeXY[ctr, 0] = self.elemSize[0] * i
                nodeXY[ctr, 1] = self.elemSize[1] * j
                ctr += 1

        return edofMat, nodeIdx, elemNodes, nodeXY, {}

    # --------------------------#
    def generatePoints(self, res=1):
        """
        Generates points for evaluating temperature fields within elements.
        """
        xy = np.zeros((res ** 2 * self.numElems, 2))
        ctr = 0
        for i in range(res * self.nelx):
            for j in range(res * self.nely):
                xy[ctr, 0] = self.elemSize[0] * (i + 0.5) / res
                xy[ctr, 1] = self.elemSize[1] * (j + 0.5) / res
                ctr += 1
        return xy

    # --------------------------#
    def processBoundaryCondition(self, fixedTempNodes=None, heatFluxNodes=None, heatSourceNodes=None):
        """
        Processes boundary conditions for heat transfer.
        """
        if not hasattr(self, 'bc'):  # Ensure bc exists before using it
            self.bc = {}

        # Ensure boundary condition keys exist
        self.bc['fixed'] = fixedTempNodes if fixedTempNodes is not None and len(fixedTempNodes) > 0 else []
        self.bc['flux'] = heatFluxNodes if heatFluxNodes is not None and len(heatFluxNodes) > 0 else []
        self.bc['source'] = heatSourceNodes if heatSourceNodes is not None and len(heatSourceNodes) > 0 else []

        # 🔥 Fix: Ensure 'heat' key is present
        self.bc['heat'] = np.zeros((self.ndof, 1))  # Default heat source as zero

        # If heat source nodes exist, update heat values
        if heatSourceNodes is not None:
            for node, value in heatSourceNodes:
                self.bc['heat'][node] = value  # Assign heat source value

        # Compute free nodes (excluding fixed temperature nodes)
        self.bc['free'] = np.setdiff1d(np.arange(self.ndof), self.bc['fixed'])

    # --------------------------#
    def plotFieldOnMesh(self, field, titleStr):
        """
        Plots temperature distribution on the mesh.
        """
        plt.ion()
        plt.clf()

        plt.imshow(
            -np.flipud(field.reshape((self.nelx, self.nely)).T),
            cmap='hot', interpolation='none'
        )
        plt.colorbar(label="Temperature (°C)")
        plt.xlabel("X-axis (m)")
        plt.ylabel("Y-axis (m)")
        plt.title(titleStr)
        plt.grid(False)
        plt.pause(0.01)
        self.fig.canvas.draw()
## ..................##

class UnstructuredMesher:
  def __init__(self, bcFiles, physics='thermal'):
    """
    Initializes an unstructured mesh for heat transfer or structural analysis.

    - bcFiles: Dictionary containing file paths for boundary conditions.
    - physics: 'thermal' (default) or 'structural'.
    """
    self.bcFiles = bcFiles
    self.meshProp = {}
    self.meshType = 'unstructuredMesh'
    self.ndim = 2  # 2D mesh
    self.physics = physics

    # Degrees of freedom per node: 1 for thermal, 2 for structural
    self.dofsPerNode = 1 if physics == 'thermal' else 2

    self.readMeshData()
    self.fig, self.ax = plt.subplots()

  # -----------------------#
  def readMeshData(self):
    """Reads mesh and boundary condition data from files."""
    self.bc = {}
    self.nodesPerElem = 4  # Quadrilateral mesh

    # Read thermal boundary conditions (or force conditions for structural cases)
    if self.physics == 'thermal':
      with open(self.bcFiles['heatFluxFile']) as f:
        self.bc['flux'] = np.array([float(line.rstrip()) for line in f]).reshape(-1, 1)
    else:
      with open(self.bcFiles['forceFile']) as f:
        self.bc['force'] = np.array([float(line.rstrip()) for line in f]).reshape(-1, 1)

    self.ndof = self.bc['flux'].shape[0] if self.physics == 'thermal' else self.bc['force'].shape[0]
    self.numNodes = int(self.ndof / self.dofsPerNode)  # Adjusted for thermal DOFs

    # Read fixed nodes (fixed temperature or fixed displacement)
    with open(self.bcFiles['fixedFile']) as f:
      self.bc['fixed'] = np.array([int(line.rstrip()) for line in f]).reshape(-1)

    self.bc['free'] = np.setdiff1d(np.arange(self.ndof), self.bc['fixed'])

    # Read node coordinates
    self.nodeXY = np.zeros((self.numNodes, self.ndim))
    ctr = 0
    with open(self.bcFiles['nodeXYFile']) as f:
      for line in f:
        self.nodeXY[ctr, :] = list(map(float, line.rstrip().split('\t')))
        ctr += 1

    # Read element connectivity
    with open(self.bcFiles['elemNodesFile']) as f:
      self.numElems = int(f.readline().rstrip())
      self.elemSize = np.zeros((2))
      self.elemSize[0], self.elemSize[1] = list(map(float, f.readline().rstrip().split('\t')))
      self.elemArea = self.elemSize[0] * self.elemSize[1] * jnp.ones((self.numElems))
      self.totalMeshArea = jnp.sum(self.elemArea)

      self.elemNodes = np.zeros((self.numElems, self.nodesPerElem), dtype=int)
      self.edofMat = np.zeros((self.numElems, self.nodesPerElem * self.dofsPerNode), dtype=int)

      ctr = 0
      for line in f:
        self.elemNodes[ctr, :] = list(map(int, line.rstrip().split('\t')))
        if self.physics == 'structural':
          self.edofMat[ctr, :] = np.array(
            [[2 * self.elemNodes[ctr, i], 2 * self.elemNodes[ctr, i] + 1] for i in range(self.nodesPerElem)]
          ).reshape(-1)
        else:  # Thermal case
          self.edofMat[ctr, :] = self.elemNodes[ctr, :]

        ctr += 1

    # Compute element centers
    self.elemCenters = np.zeros((self.numElems, self.ndim))
    for elem in range(self.numElems):
      nodes = self.elemNodes[elem, :]
      self.elemCenters[elem, 0] = np.mean(self.nodeXY[nodes, 0])
      self.elemCenters[elem, 1] = np.mean(self.nodeXY[nodes, 1])

    # Compute bounding box
    self.bb = {
      'xmin': np.min(self.nodeXY[:, 0]),
      'xmax': np.max(self.nodeXY[:, 0]),
      'ymin': np.min(self.nodeXY[:, 1]),
      'ymax': np.max(self.nodeXY[:, 1])
    }

  # -----------------------#
  def generatePoints(self, res=1, includeEndPts=False):
    """Generates additional points inside each element for evaluations."""
    endPts = 2 if includeEndPts else 0
    resMin, resMax = (0, res + 2) if includeEndPts else (1, res + 1)

    points = np.zeros((self.numElems * (res + endPts) ** 2, 2))
    ctr = 0
    for elm in range(self.numElems):
      nodes = self.elemNodes[elm, :]
      xmin, xmax = np.min(self.nodeXY[nodes, 0]), np.max(self.nodeXY[nodes, 0])
      ymin, ymax = np.min(self.nodeXY[nodes, 1]), np.max(self.nodeXY[nodes, 1])
      delX = (xmax - xmin) / (res + 1.)
      delY = (ymax - ymin) / (res + 1.)
      for rx in range(resMin, resMax):
        xv = xmin + rx * delX
        for ry in range(resMin, resMax):
          points[ctr, 0] = xv
          points[ctr, 1] = ymin + ry * delY
          ctr += 1
    return points

  # -----------------------#
  def plotFieldOnMesh(self, field, titleStr, res=1):
    """Plots the field (e.g., temperature distribution) on the mesh."""
    y = self.nodeXY[:, 0]
    z = self.nodeXY[:, 1]

    def quatplot(y, z, quadrangles, values, ax=None, **kwargs):
      """Helper function to plot quadrilateral elements with color mapping."""
      if not ax:
        ax = plt.gca()
      yz = np.c_[y, z]
      verts = yz[quadrangles]
      pc = matplotlib.collections.PolyCollection(verts, **kwargs)
      pc.set_array(values)
      ax.add_collection(pc)
      ax.autoscale()
      ax.set_aspect('equal')
      return pc

    plt.ion()
    plt.clf()

    pc = quatplot(y, z, np.asarray(self.elemNodes), -field, ax=None, edgecolor="black", cmap="hot")
    plt.colorbar(label="Temperature (°C)")
    plt.title(titleStr)
    plt.pause(0.001)
    plt.show()
    self.fig.canvas.draw()

