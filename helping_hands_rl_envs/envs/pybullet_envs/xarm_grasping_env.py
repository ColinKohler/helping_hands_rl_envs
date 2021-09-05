import pybullet as pb
import numpy as np
import copy

from nuro_arm.robot.robot_arm import RobotArm
from helping_hands_rl_envs.simulators.pybullet.objects.random_household_object_200 import RandomHouseHoldObject200

class TopDownDepthCamera:
    def __init__(self, workspace):
        '''
        Note
        ----
        Axis of image must align with axis of workspace
        '''
        cam_z = 10
        ws_center = workspace[:2].mean(axis=1)
        cam_pos = np.array((*ws_center, cam_z))
        cam_target = np.array((*ws_center, 0))
        self.view_mtx = pb.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraUpVector=(-1,0,0),
            cameraTargetPosition=cam_target
        )
        extents = workspace[:2,1] - workspace[:2,0]
        target_size = np.amin(extents)
        aspect = extents[1]/extents[0]
        fov = np.degrees(2 * np.arctan(target_size/2) / cam_z)
        self.proj_mtx = pb.computeProjectionMatrixFOV(fov, aspect, cam_z-1, cam_z)

    def get_heightmap(self, height, width):
        depth = pb.getCameraImage(width=width,
                                  height=height,
                                  viewMatrix=self.view_mtx,
                                  projectionMatrix=self.proj_mtx,
                                  renderer=pb.ER_TINY_RENDERER)[3]
        depth = np.array(depth)
        return np.abs(depth - np.max(depth))

class TopDownDepthEnv:
    def __init__(self,
                 n_objects=8,
                 render=False,
                 workspace=((-0.15, 0.25),(-0.2, 0.2),(0, 0.1)),
                 heightmap_size=128,
                 n_rotations=16,
                 z_heuristic_mode='offset',
                 seed=None,
                 max_episode_steps=5,
                 base_height=0.06,
                 **kwargs,
                ):
        print('WARNING: roll value may fall outside -PI/2, PI/2, need to change this later')
        self.seed = seed if seed is not None else np.random.randint(10000)
        np.random.randint(self.seed)

        self.max_episode_steps = max_episode_steps
        self.episode_step_count = 0
        # self.workspace = np.array(((0.115, 0.19),
                                   # (-0.075, 0.075),
                                   # (0, 0.1)))
        self.workspace = np.array(workspace)
        self.base_height = base_height
        self.robot = RobotArm('sim', headless=not render, realtime=False)
        self.robot._sim.reset_robot_base([0,0,base_height])
        # add base platform
        vis_id = pb.createVisualShape(pb.GEOM_BOX,
                                      halfExtents=[0.078, 0.07, base_height/2])
        col_id = pb.createCollisionShape(pb.GEOM_BOX,
                                         halfExtents=[0.078, 0.07, base_height/2])
        pb.createMultiBody(0, col_id, vis_id, [0,0,base_height/2])

        self.rest_jpos = [0, -0.8, 0.8, 0.8, 0]
        self.pregrasp_jpos = [0, 0.2, 0.8, 0.8, 0]

        assert z_heuristic_mode in ('offset', 'mean')
        self.z_heuristic_mode = z_heuristic_mode
        self.pick_z_offset = -0.02
        self.min_pick_z = 0.006
        self.minimum_lift_height = base_height
        self.sensor = TopDownDepthCamera(self.workspace)

        self.n_objects = n_objects
        self.objects = []

        self.heightmap_size = np.array((heightmap_size, heightmap_size))
        self.heightmap_resolution = np.subtract(*self.workspace[:2,::-1].T)/self.heightmap_size
        self.rotations = np.linspace(0, np.pi, num=n_rotations, endpoint=False)

        self.obj_scales = (0.5, 1)

        # for se2 mask
        self.se2_base_obstruction_x = 0.1
        self.se2_base_obstruction_y = 0.09
        self.se2_max_reach_radius = 0.185
        self.se2_max_pivot_theta = 2.

        # heightmap must ignore xarm
        self.zeroed_px_high = int(self._position_to_pixel((self.se2_base_obstruction_x, 0))[0])
        self.zeroed_py_low = int(self._position_to_pixel((0, -self.se2_base_obstruction_y))[1])
        self.zeroed_py_high = int(self._position_to_pixel((0, self.se2_base_obstruction_y))[1])

        if render:
            self.show_workspace()

    def get_se2_mask(self):
        '''mask: True if position is se2 feasible
        '''
        pixels = np.moveaxis(np.mgrid[:self.heightmap_size[0], :self.heightmap_size[1]], 0, -1)
        positions = self._pixel_to_position(pixels)
        return self.valid_se2_position(positions)

    def valid_se2_position(self, position):
        is_valid = np.bitwise_and.reduce(
            (np.linalg.norm(position[...,:2], axis=-1) < self.se2_max_reach_radius,
             np.abs(np.arctan2(position[...,1], position[...,0])) < self.se2_max_pivot_theta,
             np.bitwise_not(
                 np.bitwise_and(np.abs(position[...,0]) < self.se2_base_obstruction_x,
                                np.abs(position[...,1]) < self.se2_base_obstruction_y)
             ),
            )
        )
        return is_valid

    def reset(self):
        self.episode_step_count = 0
        self._reset_scene()
        self.heightmap = None

        for _ in range(self.n_objects):
            new_obj = self.place_object()
            self.objects.append(new_obj)
        return self.get_observation()

    def _reset_scene(self):
        # clear objects
        for obj in self.objects:
            pb.removeBody(obj.id_)
        self.objects = []

        # # reset robot
        # self.robot.move_arm_jpos(self.rest_jpos)
        # self.robot.open_gripper()

    def place_object(self):
        while 1:
            pos = np.random.uniform(self.workspace[:,0], self.workspace[:,1])
            pos[2] = self.workspace[2,1]
            if not self.valid_se2_position(pos):
                continue
            euler = np.random.uniform(0, 2*np.pi, size=(3,))
            quat = pb.getQuaternionFromEuler(euler)
            scale = np.random.uniform(*self.obj_scales)
            obj_cand = RandomHouseHoldObject200(pos, quat, scale)
            if obj_cand.let_fall(self.valid_se2_position):
                return obj_cand
            pb.removeBody(obj_cand.id_)

    def step(self, a):
        self.episode_step_count += 1
        x,y,th,prim = a
        pxy = self._position_to_pixel(np.array((x,y)))

        z = self.calc_z(pxy)

        # theta will come in from 0 to PI but we want -PI/2 to PI/2
        th -= np.pi/2

        # compensate for base rotation so it is axis aligned
        yaw = np.arctan2(y, x)
        th += yaw

        self.robot.open_gripper()
        self.robot.move_arm_jpos([yaw] + self.pregrasp_jpos[1:-1] + [th])
        self.robot.move_hand_to((x,y,z), pitch_roll=(np.pi, th))
        self.robot.close_gripper()
        self.robot.move_arm_jpos(self.rest_jpos)

        reward = self.get_reward()
        done = self.is_done()
        return self.get_observation(), reward, done

    def is_done(self):
        if self.episode_step_count >= self.max_episode_steps:
            return True

        # there must be at least one object still in workspace
        for obj in self.objects:
            COM = np.mean(obj.getAABB()[:,:2], axis=0)
            if ((COM > self.workspace[:2, 0]).any() \
                or (COM < self.workspace[:2,1]).any()):
                return False
        return True

    def get_reward(self):
        '''Calculates reward after pick action

        R=1 if some object is above some threshold in the z dimension, R=0
        otherwise.
        '''
        for obj in self.objects:
            pos = obj.get_position()
            if pos[2] > self.minimum_lift_height:
                # remove object
                pb.removeBody(obj.id_)
                self.objects.remove(obj)
                return 1
        return 0

    def _pixel_to_position(self, pxy):
        '''Converts pixel positions of top_down image to positions in the
        simulator, where the image is indexed like I[px,py].

        :pxy: array_like with length 2
        '''
        xy = np.add(pxy * self.heightmap_resolution, self.workspace[:2,0])
        return xy

    def _position_to_pixel(self, xy):
        pxy = np.subtract(xy, self.workspace[:2,0]) / self.heightmap_resolution
        return pxy

    def calc_z(self, pxy):
        '''Uses local patch of height map to determine z value, does not add
        offset'''
        lower = np.clip(pxy - self.heightmap_size/20, 0, self.heightmap_size-1).astype(int)
        upper = np.clip(pxy + self.heightmap_size/20, 0, self.heightmap_size-1).astype(int)
        local_patch = self.heightmap[lower[0]:upper[0],lower[1]:upper[1]]
        max_z_local = np.amax(local_patch)

        if self.z_heuristic_mode == 'offset':
            z = max_z_local + self.pick_z_offset
        else:
            z = (max_z_local)/2
        return max(z, self.min_pick_z) + self.workspace[2][0]

    def get_heightmap(self):
        hmap = self.sensor.get_heightmap(*self.heightmap_size)
        hmap[:self.zeroed_px_high,self.zeroed_py_low:self.zeroed_py_high] = 0
        return hmap

    def get_observation(self):
        self.heightmap = self.get_heightmap()

        is_holding = False
        in_hand_img = np.zeros((1,10,10))
        return is_holding, in_hand_img, self.heightmap[None]

    def show_workspace(self):
        '''Add visuals to see workspace in simulator
        '''
        for i in range(4):
            x = [0,1,1,0]
            y = [0,0,1,1]
            start = (self.workspace[0][x[i]], self.workspace[1][y[i]], 0.002)
            end = (self.workspace[0][x[(i+1)%4]], self.workspace[1][y[(i+1)%4]], 0.002)
            pb.addUserDebugLine(start, end, lineColorRGB=(0,0,0), lineWidth=3)

    def saveEnvToFile(self, *args, **kwargs):
        pass

def createXArmGraspingEnv(config):
    return TopDownDepthEnv(**config)

if __name__ == "__main__":
    env = TopDownDepthEnv(render=True, )
    import time
    import matplotlib.pyplot as plt
    # obs = env.reset()[2][0]

    # plt.figure()
    # plt.imshow(mask)
    # plt.axis('off')
    # plt.title('SE(2) Feasibility Mask')
    # plt.tight_layout(0)
    # # plt.imshow(obs)
    # plt.show()
    # exit()

    def argmax2d(img):
        flat = img.reshape(-1)
        return np.unravel_index(np.argmax(flat), img.shape)

    while 1:
        obs = env.reset()
        done = False
        while not done:
            x, y = env._pixel_to_position(argmax2d(obs[2][0]))
        # while 1:
            # x,y = np.random.uniform(env.workspace[:2,0],env.workspace[:2,1])
            # print(x,y)
            # if env.valid_se2_position(np.array((x,y))):
                # break
            th = np.random.uniform(0, 15*np.pi/16)
            obs, r, done = env.step(np.array((x, y, th, 0)))
        # f = plt.figure()
        # plt.imshow(obs)
        # plt.plot(y,x, 'r.')
        # plt.axis('off')
        # plt.show()
