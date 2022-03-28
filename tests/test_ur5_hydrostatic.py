import pybullet as pb
import time

physicsClient = pb.connect(pb.GUI)
pb.setGravity(0,0,-10)

'''Test URDF'''
# robot = pb.loadURDF('helping_hands_rl_envs/simulators/urdf/ur5/ur5_hydrostatic_gripper.urdf')
# num_joints = pb.getNumJoints(robot)
# print("num joints : {}".format(num_joints))
# for i in range(num_joints):
#     joint_info = pb.getJointInfo(robot, i)
#     print("joint {} info : {}".format(i,joint_info))
'''Test control class'''
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.robots.robot_base import RobotBase
from helping_hands_rl_envs.simulators.pybullet.robots.ur5_hydrostatic import UR5_Hydrostatic
robot = UR5_Hydrostatic()
robot.initialize()
pb.setRealTimeSimulation(True)
time.sleep(2)
robot.openGripper()
time.sleep(2)
robot.closeGripper()

input("hit enter to quit")
pb.disconnect()
