import pybullet as pb
import time
import sys
import os

#only works if in the top level dir is the project dir
sys.path.append(os.getcwd())
from helping_hands_rl_envs.pybullet.robots.ur5_hydrostatic import UR5_Hydrostatic
import numpy as np

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
# from helping_hands_rl_envs.simulators.pybullet.robots.ur5_robotiq import UR5_Robotiq
# robot = UR5_Robotiq()
robot = UR5_Hydrostatic()
robot.initialize()
pb.setRealTimeSimulation(True)
robot.closeGripper()
time.sleep(0.1)

print("Joint Info:")
for i in range(pb.getNumJoints(robot.id)):
    print("Joint {}: {}".format(i, pb.getJointInfo(robot.id,i)[12]))


start_pos = [0.2, 0.0, 0.2]
end_pos = [0.6, 0.0, 0.2]
rot = [0.0, 0.0, 0.0, 1.0]
start_joints = robot._calculateIK(start_pos, rot)
print("Calculated Joints: \n{}".format(start_joints))

joints_raw = pb.calculateInverseKinematics(robot.id, robot.end_effector_index, start_pos, rot)
print("PB calculated joints: \n{}".format(joints_raw))
#reset to start pose
# for i, joint_pos in enumerate(start_joints):
#     pb.resetJointState(robot.id, i, joint_pos)
# [pb.resetJointState(robot.id, idx, start_joints[idx]) for idx in range(len(start_joints))]
# pb.resetJointState(robot.id, robot.gripper_joint_indices[0], robot.gripper_joint_limit[1])
# pb.resetJointState(robot.id, robot.gripper_joint_indices[1], robot.gripper_joint_limit[1])
# robot.moveToJ(start_joints)
robot.moveTo(start_pos, rot)
# robot.openGripper()
time.sleep(0.5)
#get ee position
for i in [robot.end_effector_index]:
    print("****")
    print("Index : {}".format(i))
    ee_pos, ee_rot, _, _, ee_u_pos, ee_u_rot     = pb.getLinkState(robot.id, i)
    print("Desired Pose")
    print(start_pos, rot)
    print("Actua; Pose")
    print(ee_pos, ee_rot)
    # print(ee_u_pos, ee_u_rot)

joint_positions = list(zip(*pb.getJointStates(robot.id, robot.arm_joint_indices)))[0]
print("Actual Joint Positions: \n{}".format(joint_positions))
#get IK for pose
# robot.end_effector_index = 12

# for x in np.linspace(0.2,0.7,100):
#     pos[0] = x
#     joints = robot._calculateIK(pos, rot)
#     print("commanded psition : {}, {}".format(pos, rot))
#     print("joints: {}".format(joints))
#     robot._sendPositionCommand(joints)
#     time.sleep(0.05)

input("hit enter to quit")
pb.disconnect()
