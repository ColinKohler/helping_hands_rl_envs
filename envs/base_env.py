import numpy as np
import numpy.random  as npr

class BaseEnv(object):
  # Motion primatives
  PICK_PRIMATIVE = 0
  PLACE_PRIMATIVE = 1
  PUSH_PRIMATIVE = 2

  # Shape types
  CUBE = 0
  SPHERE = 1
  CYLINDER = 2
  CONE = 3

  '''
  Base Arm RL environment.
  '''
  def __init__(self, seed, workspace, max_steps, heightmap_size):
    # Set random numpy seed
    npr.seed(seed)

    # Setup environment
    self.workspace = workspace
    self.workspace_size = np.linalg.norm(self.workspace[0,1] - self.workspace[0,0])
    self.max_steps = max_steps

    # Setup heightmap
    self.heightmap_size = heightmap_size
    self.heightmap_shape = (self.heightmap_size, self.heightmap_size, 1)
    self.heightmap_resolution = self.workspace_size / self.heightmap_size

  def _getShapeName(self, shape_type):
    ''' Get the shape name from the type (int) '''
    if shape_type == CUBE: return 'cube'
    elif shape_type == SPHERE: return 'sphere'
    elif shape_type == CYLINER: return 'cylinder'
    elif shape_type == CONE: return 'cone'
    else: return 'unknown'

  def _getPrimativeHeight(self, motion_primative, x, y, offset=0.01):
    '''
    Get the z position for the given action using the current heightmap.
    Args:
      - motion_primative: Pick/place motion primative
      - x: X coordinate for action
      - y: Y coordinate for action
      - offset: How much to offset the action along approach vector
    Returns: Valid Z coordinate for the action
    '''
    x_pixel, y_pixel = self._getPixelsFromPos(x, y)
    local_region = self.heightmap[max(y_pixel - 30, 0):min(y_pixel + 30, self.heightmap_size), \
                                  max(x_pixel - 30, 0):min(x_pixel + 30, self.heightmap_size)]
    safe_z_pos = np.max(local_region) + self.workspace[2][0]
    safe_z_pos = safe_z_pos - offset if motion_primative == PICK_PRIMATIVE else safe_z_pos + offset

    return safe_z_pos

  def _getPixelsFromPos(self, x, y):
    '''
    Get the x/y pixels on the heightmap for the given coordinates
    Args:
      - x: X coordinate
      - y: Y coordinate
    Returns: (x, y) in pixels corresponding to coordinates
    '''
    x_pixel = (x - self.workspace[0][0]) / self.heightmap_resolution
    y_pixel = (y - self.workspace[1][0]) / self.heightmap_resolution

    return int(x_pixel), int(y_pixel)


  def _isPointInWorkspace(self, p):
    '''
    Checks if the given point is within the workspace

    Args:
      - p: [x, y, z] point

    Returns: True in point is within workspace, False otherwise
    '''
    return p[0] > self.workspace[0][0] - 0.1 and p[0] < self.workspace[0][1] + 0.1 and \
           p[1] > self.workspace[1][0] - 0.1 and p[1] < self.workspace[1][1] + 0.1 and \
           p[2] > self.workspace[2][0] and p[2] < self.workspace[2][1]