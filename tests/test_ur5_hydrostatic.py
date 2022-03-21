import pybullet as pb

physicsClient = pb.connect(pb.GUI)
pb.setGravity(0,0,-10)
robot = pb.loadURDF('helping_hands_rl_envs/simulators/urdf/ur5/ur5_hydrostatic_gripper.urdf')
input("hit enter to quit")
pb.disconnect()
