import os
import copy
import math
import numpy as np
import numpy.random as npr
from collections import deque, namedtuple
from attrdict import AttrDict
from threading import Thread

import pybullet as pb
import pybullet_data

import helping_hands_rl_envs
import time
from helping_hands_rl_envs.simulators.pybullet.robots.robot_base import RobotBase
from helping_hands_rl_envs.simulators import constants

jointInfo = namedtuple("jointInfo",
                       ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])
jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

class UR5_Hydrostatic(RobotBase):
  '''

  '''
  def __init__(self):
    super(UR5_Hydrostatic, self).__init__()
    # Setup arm and gripper variables
    self.max_forces = [150, 150, 150, 28, 28, 28, 30, 30]
    self.gripper_close_force = [30] * 2
    self.gripper_open_force = [30] * 2
    self.end_effector_index = 22

    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    self.home_positions_joint = self.home_positions[1:7]
    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)

    self.gripper_joint_limit = [0, 0.3] #TODO optimize
    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()

    # ###############################################
    # ## fake robotiq 85
    # # the open length of the gripper. 0 is closed, 0.085 is completely opened
    # self.robotiq_open_length_limit = [0, 0.085]
    # # the corresponding robotiq_85_left_knuckle_joint limit
    # self.robotiq_joint_limit = [0.715 - math.asin((self.robotiq_open_length_limit[0] - 0.010) / 0.1143),
    #                             0.715 - math.asin((self.robotiq_open_length_limit[1] - 0.010) / 0.1143)]
    #
    # self.robotiq_controlJoints = ["robotiq_85_left_knuckle_joint",
    #                       "robotiq_85_right_knuckle_joint",
    #                       "robotiq_85_left_inner_knuckle_joint",
    #                       "robotiq_85_right_inner_knuckle_joint",
    #                       "robotiq_85_left_finger_tip_joint",
    #                       "robotiq_85_right_finger_tip_joint"]
    # self.robotiq_main_control_joint_name = "robotiq_85_left_inner_knuckle_joint"
    # self.robotiq_mimic_joint_name = [
    #   "robotiq_85_right_knuckle_joint",
    #   "robotiq_85_left_knuckle_joint",
    #   "robotiq_85_right_inner_knuckle_joint",
    #   "robotiq_85_left_finger_tip_joint",
    #   "robotiq_85_right_finger_tip_joint"
    # ]
    # self.robotiq_mimic_multiplier = [1, 1, 1, 1, -1, -1]
    # self.robotiq_joints = AttrDict()

  def initialize(self):
    ''''''
    ur5_urdf_filepath = os.path.join(self.root_dir, 'simulators/urdf/ur5/ur5_hydrostatic_gripper_v2.urdf')
    self.id = pb.loadURDF(ur5_urdf_filepath, [0,0,0.1], [0,0,0,1])
    # self.is_holding = False
    self.gripper_closed = True
    self.holding_obj = None
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(1, 7):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)
    elif i in range(11, 14):
        self.gripper_joint_names.append(str(joint_info[1]))
        self.gripper_joint_indices.append(i)

      # elif i in range(14, self.num_joints):
      #   info = pb.getJointInfo(self.id, i)
      #   jointID = info[0]
      #   jointName = info[1].decode("utf-8")
      #   jointType = jointTypeList[info[2]]
      #   jointLowerLimit = info[8] #TODO what is this?
      #   jointUpperLimit = info[9]
      #   jointMaxForce = info[10]
      #   jointMaxVelocity = info[11]
      #   singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
      #                          jointMaxVelocity)
      #   self.robotiq_joints[singleInfo.name] = singleInfo

  def reset(self):
    self.gripper_closed = False
    self.holding_obj = None
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

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
        # mean = (p1 + p2) / 2 - 0.001
        # self._sendGripperCommand(mean, mean)
        return False
      p1 = p1_
      p2 = p2_
    return True

  def adjustGripperCommand(self):
    '''TODO: As currently implemented this causes the gripper to loosen it's grip,
    and drop things, bcause it resets the gripper close target from 0 to a small
     offset on the current position. It is not clear that this gripper should have
     and adjust function because there is no mechanical constraint that the fingers
     have the same joint angle as with the robotiq or other grippers'''
    # p1, p2 = self._getGripperJointPosition()
    # mean = (p1 + p2) / 2 - 0.01
    # self._sendGripperCommand(mean, mean, self.gripper_close_force)
    return

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

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:6]

  def _getGripperJointPosition(self):
    p1 = pb.getJointState(self.id, self.gripper_joint_indices[0])[0]
    p2 = pb.getJointState(self.id, self.gripper_joint_indices[1])[0]
    return p1, p2

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 [0.]*num_motors, self.max_forces[:-2], [0.02]*num_motors, [1.0]*num_motors)

  def _sendGripperCommand(self, target_pos1, target_pos2, force = 2):
    pb.setJointMotorControlArray(self.id,
                                 self.gripper_joint_indices,
                                 pb.POSITION_CONTROL,
                                 targetPositions=[target_pos1, target_pos2],
                                 forces=[force, force])
    # pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
    #                              targetPositions=[target_pos1, target_pos2], forces=self.gripper_open_force)
