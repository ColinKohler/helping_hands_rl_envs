import os
import numpy as np
import pybullet as pb
from scipy.ndimage import rotate

from helping_hands_rl_envs.pybullet.robots.robot_base import RobotBase
from helping_hands_rl_envs.pybullet.utils import constants

class UR5_Hydrostatic(RobotBase):
  '''

  '''
  def __init__(self):
    super().__init__()
    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    self.home_positions_joint = self.home_positions[1:7]
    self.max_force = 200.
    self.max_velocity = 0.35
    self.num_dofs = 6

    self.wrist_index = 8
    self.finger_a_index = 11
    self.finger_b_index = 14
    self.end_effector_index = 17
    self.gripper_z_offset = 0.05
    self.gripper_joint_limit = [0., 0.3] #TODO optimize

    self.urdf_filepath = os.path.join(constants.URDF_PATH, 'ur5/ur5_hydrostatic_gripper.urdf')

  def initialize(self):
    ''''''
    self.id = pb.loadURDF(self.urdf_filepath, useFixedBase=True)
    pb.resetBasePositionAndOrientation(self.id, [0,0,0.1], [0,0,0,1])

    self.gripper_closed = False
    self.holding_obj = None
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    pb.enableJointForceTorqueSensor(self.id, self.wrist_index)
    pb.enableJointForceTorqueSensor(self.id, self.finger_a_index)
    pb.enableJointForceTorqueSensor(self.id, self.finger_b_index)

    self.openGripper()

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(1, 7):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)

  def reset(self):
    self.gripper_closed = False
    self.holding_obj = None
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]
    self.moveToJ(self.home_positions_joint)
    self.openGripper()

  def controlGripper(self, open_ratio, max_it=100):
    p1, p2 = self._getGripperJointPosition()
    target = open_ratio * (self.gripper_joint_limit[1] - self.gripper_joint_limit[0]) + self.gripper_joint_limit[0]
    self._sendGripperCommand(target, target)
    it = 0
    while abs(target - p1) + abs(target - p2) > 0.001:
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1 - p1_) < 0.0001 and abs(p2 - p2_) < 0.0001):
        return
      p1 = p1_
      p2 = p2_

  def getGripperOpenRatio(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2) / 2
    ratio = (mean - self.gripper_joint_limit[0]) / (self.gripper_joint_limit[1] - self.gripper_joint_limit[0])
    return ratio

  def closeGripper(self, max_it=100, primative=constants.PICK_PRIMATIVE):
    ''''''
    if primative == constants.PULL_PRIMATIVE:
      self.gripper_close_force = 20
    else:
      self.gripper_close_force = 10
    p1, p2 = self._getGripperJointPosition()
    target = self.gripper_joint_limit[0]
    self._sendGripperCommand(target, target, self.gripper_close_force)
    self.gripper_closed = True
    it = 0
    while abs(target-p1) + abs(target-p2) > 0.001:
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1 - p1_) < 0.0001 and abs(p2 - p2_) < 0.0001):
        return False
      p1 = p1_
      p2 = p2_
    return True

  def adjustGripperCommand(self):
    '''
    TODO: As currently implemented this causes the gripper to loosen it's grip,
    and drop things, bcause it resets the gripper close target from 0 to a small
    offset on the current position. It is not clear that this gripper should have
    and adjust function because there is no mechanical constraint that the fingers
    have the same joint angle as with the robotiq or other grippers
    '''
    pass

  def checkGripperClosed(self):
    limit = self.gripper_joint_limit[1]
    p1, p2 = self._getGripperJointPosition()
    if (limit - p1) + (limit - p2) > 0.001:
      return
    else:
      self.holding_obj = None

  def openGripper(self):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    target = self.gripper_joint_limit[1]
    self._sendGripperCommand(target, target)
    self.gripper_closed = False
    self.holding_obj = None
    it = 0
    if self.holding_obj:
      pos, rot = self.holding_obj.getPose()
    while abs(target-p1) + abs(target-p2) > 0.001:
      if self.holding_obj and it < 5:
        self.holding_obj.resetPose(pos, rot)
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1_, p2_ = self._getGripperJointPosition()
      if p1 >= p1_ and p2 >= p2_:
        return False
      p1 = p1_
      p2 = p2_
    return True

  def gripperHasForce(self):
    return pb.getJointState(self.id, self.wrist_index)[2][2] > 100

  def getWristForce(self):
    wrist_rot = pb.getMatrixFromQuaternion(pb.getLinkState(self.id, self.wrist_index - 1)[5])
    wrist_rot = np.array(list(wrist_rot)).reshape((3,3))

    wrist_force = np.array(list(pb.getJointState(self.id, self.wrist_index)[2][:3]))
    wrist_force = np.dot(wrist_rot, wrist_force)

    wrist_moment = np.array(list(pb.getJointState(self.id, self.wrist_index)[2][3:]))
    wrist_moment = np.dot(wrist_rot, wrist_moment)

    return wrist_force, wrist_moment

  def getFingerForce(self):
    wrist_rot = pb.getMatrixFromQuaternion(list(pb.getLinkState(self.id, self.wrist_index)[5]))
    wrist_rot = np.array(list(wrist_rot)).reshape((3,3))

    finger_a_force = np.array(list(pb.getJointState(self.id, self.finger_a_index)[2][:3]))
    finger_a_force = np.dot(wrist_rot, finger_a_force)

    finger_b_force = np.array(list(pb.getJointState(self.id, self.finger_b_index)[2][:3]))
    finger_b_force = np.dot(wrist_rot, finger_b_force)

    return finger_a_force, finger_b_force

  def getPickedObj(self, objects):
    if not objects:
      return None

    # TODO: Fingers here have a lot of links so might need to check more than just these two
    #       for contact. Unsure...
    for obj in objects:
      # check the contact force normal to count the horizontal contact points
      contact_points = pb.getContactPoints(self.id, obj.object_id, 13) + \
                       pb.getContactPoints(self.id, obj.object_id, 16)
      horizontal = list(filter(lambda p: abs(p[7][2]) < 0.4, contact_points))
      if len(horizontal) >= 2:
        return obj

  def getGripperImg(self, img_size, workspace_size, obs_size_m):
    gripper_state = self.getGripperOpenRatio()
    gripper_rz = pb.getEulerFromQuaternion(self._getEndEffectorRotation())[2]

    im = np.zeros((img_size, img_size))
    gripper_half_size = 4 * workspace_size / obs_size_m
    gripper_half_size = round(gripper_half_size / 128 * img_size)
    gripper_max_open = 35 * workspace_size / obs_size_m

    anchor = img_size // 2
    d = int(gripper_max_open / 128 * img_size * gripper_state)
    im[int(anchor - gripper_half_size):int(anchor + gripper_half_size), int(anchor - d // 2 - gripper_half_size):int(anchor - d // 2 + gripper_half_size)] = 1
    im[int(anchor - gripper_half_size):int(anchor + gripper_half_size), int(anchor + d // 2 - gripper_half_size):int(anchor + d // 2 + gripper_half_size)] = 1
    im = rotate(im, np.rad2deg(gripper_rz), reshape=False, order=0)

    return im

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:self.num_dofs]

  def _getGripperJointPosition(self):
    p1 = pb.getJointState(self.id, self.finger_a_index)[0]
    p2 = pb.getJointState(self.id, self.finger_b_index)[0]
    return p1, p2

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 targetVelocities=[0.] * num_motors,
                                 forces=[self.max_force] * num_motors,
                                 positionGains=[self.position_gain] * num_motors,
                                 velocityGains=[1.0] * num_motors)

  def _sendGripperCommand(self, target_pos1, target_pos2, force=10):
    pb.setJointMotorControlArray(self.id,
                                 [self.finger_a_index, self.finger_b_index],
                                 pb.POSITION_CONTROL,
                                 targetPositions=[target_pos1, target_pos2],
                                 forces=[force, force])
