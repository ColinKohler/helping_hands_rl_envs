import pybullet as pb
import time
import sys
import os
sys.path.append(os.getcwd())
physicsClient = pb.connect(pb.GUI)
pb.setGravity(0,0,-10)

'''Test URDF'''
# robot = pb.loadURDF('helping_hands_rl_envs/pybullet/urdf/ur5/ur5_hydrostatic_gripper.urdf')
# num_joints = pb.getNumJoints(robot)
# print("num joints : {}".format(num_joints))
# for i in range(num_joints):
#     joint_info = pb.getJointInfo(robot, i)
#     print("joint {} info : {}".format(i,joint_info))
#
# #link info
# info = pb.getDynamicsInfo(robot,10)
# print("Link info")
# print(info)
#
# #open gripper
# pb.setJointMotorControlArray(robot, [11,14], pb.POSITION_CONTROL, [1,1])
# pb.setRealTimeSimulation(True)

'''Test control class'''
from helping_hands_rl_envs.pybullet.robots.ur5_hydrostatic import UR5_Hydrostatic
import numpy as np
# from helping_hands_rl_envs.simulators.pybullet.robots.ur5_robotiq import UR5_Robotiq
# robot = UR5_Robotiq()
robot = UR5_Hydrostatic()
robot.initialize()
pb.setRealTimeSimulation(True)
robot.closeGripper()
time.sleep(2)
robot.openGripper()
time.sleep(2)
# time.sleep(2)
# robot.openGripper()
# time.sleep(2)
# robot.closeGripper()

#get IK for pose
# robot.end_effector_index = 12
pos = [0.0, 0.0, 0.2]
rot = [1.0, 0.0, 0.0, 0.0]

for x in np.linspace(0.2,0.7,100):
    pos[0] = x
    joints = robot._calculateIK(pos, rot)
    print("commanded psition : {}, {}".format(pos, rot))
    print("joints: {}".format(joints))
    robot._sendPositionCommand(joints)
    time.sleep(0.05)

input("hit enter to quit")
pb.disconnect()
