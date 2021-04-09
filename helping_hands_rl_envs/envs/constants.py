import numpy as np

from helping_hands_rl_envs.envs.pybullet_envs.block_picking_env import createBlockPickingEnv
from helping_hands_rl_envs.envs.pybullet_envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.block_adjacent_env import createBlockAdjacentEnv
from helping_hands_rl_envs.envs.pybullet_envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.pyramid_stacking_env import createPyramidStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_2_env import createHouseBuilding2Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_4_env import createHouseBuilding4Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_5_env import createHouseBuilding5Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_x_env import createHouseBuildingXEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_2_env import createImproviseHouseBuilding2Env
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_3_env import createImproviseHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_discrete_env import createImproviseHouseBuildingDiscreteEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_random_env import createImproviseHouseBuildingRandomEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_1_deconstruct_env import createHouseBuilding1DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_2_deconstruct_env import createHouseBuilding2DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_3_deconstruct_env import createHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_4_deconstruct_env import createHouseBuilding4DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_x_deconstruct_env import createHouseBuildingXDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_2_deconstruct_env import createImproviseHouseBuilding2DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_3_deconstruct_env import createImproviseHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_discrete_deconstruct_env import createImproviseHouseBuildingDiscreteDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_random_deconstruct_env import createImproviseHouseBuildingRandomDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_block_stacking_env import createRampBlockStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_house_building_1_env import createRampHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_house_building_2_env import createRampHouseBuilding2Env
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_house_building_3_env import createRampHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_house_building_4_env import createRampHouseBuilding4Env
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_improvise_house_building_2_env import createRampImproviseHouseBuilding2Env
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_improvise_house_building_3_env import createRampImproviseHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_block_stacking_deconstruct_env import createRampBlockStackingDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_house_building_1_deconstruct_env import createRampHouseBuilding1DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_house_building_2_deconstruct_env import createRampHouseBuilding2DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_house_building_3_deconstruct_env import createRampHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_house_building_4_deconstruct_env import createRampHouseBuilding4DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_improvise_house_building_2_deconstruct_env import createRampImproviseHouseBuilding2DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_improvise_house_building_3_deconstruct_env import createRampImproviseHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.random_picking_env import createRandomPickingEnv
from helping_hands_rl_envs.envs.pybullet_envs.random_stacking_env import createRandomStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.multi_task_env import createMultiTaskEnv
from helping_hands_rl_envs.envs.pybullet_envs.household_envs.two_view_drawer_teapot_env import createTwoViewDrawerTeapotEnv
from helping_hands_rl_envs.envs.pybullet_envs.household_envs.mview_drawer_teapot_env import createMViewDrawerTeapotEnv
from helping_hands_rl_envs.envs.pybullet_envs.cup_stacking_env import createCupStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.bowl_stacking_env import createBowlStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.household_envs.shelf_bowl_stacking_env import createShelfBowlStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.household_envs.shelf_plate_stacking_env import createShelfPlateStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.household_envs.drawer_shelf_plate_stacking_env import createDrawerShelfPlateStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.household_envs.block_bin_packing_env import createBlockBinPackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.random_block_picking_env import createRandomBlockPickingEnv
from helping_hands_rl_envs.envs.pybullet_envs.random_household_picking_env import createRandomHouseholdPickingEnv


CREATE_NUMPY_ENV_FNS = {
  'block_picking' : createBlockPickingEnv,
  'block_stacking' : createBlockStackingEnv,
  'block_adjacent' : createBlockAdjacentEnv,
  'brick_stacking' : createBrickStackingEnv,
  'pyramid_stacking' : createPyramidStackingEnv,
  'house_building_1' : createHouseBuilding1Env,
  'house_building_2' : createHouseBuilding2Env,
  'house_building_3' : createHouseBuilding3Env,
  'house_building_4' : createHouseBuilding4Env,
  'house_building_5' : createHouseBuilding5Env,
}

CREATE_PYBULLET_ENV_FNS = {
  'block_picking' : createBlockPickingEnv,
  'block_stacking' : createBlockStackingEnv,
  'block_adjacent' : createBlockAdjacentEnv,
  'brick_stacking' : createBrickStackingEnv,
  'pyramid_stacking' : createPyramidStackingEnv,
  'house_building_1' : createHouseBuilding1Env,
  'house_building_2' : createHouseBuilding2Env,
  'house_building_3' : createHouseBuilding3Env,
  'house_building_4' : createHouseBuilding4Env,
  'house_building_5' : createHouseBuilding5Env,
  'house_buliding_x' : createHouseBuildingXEnv,
  'improvise_house_building_2' : createImproviseHouseBuilding2Env,
  'improvise_house_building_3' : createImproviseHouseBuilding3Env,
  'improvise_house_building_discrete' : createImproviseHouseBuildingDiscreteEnv,
  'improvise_house_building_random' : createImproviseHouseBuildingRandomEnv,
  'house_building_1_deconstruct' : createHouseBuilding1DeconstructEnv,
  'house_building_2_deconstruct' : createHouseBuilding2DeconstructEnv,
  'house_building_3_deconstruct' : createHouseBuilding3DeconstructEnv,
  'house_building_4_deconstruct' : createHouseBuilding4DeconstructEnv,
  'house_building_x_deconstruct' : createHouseBuildingXDeconstructEnv,
  'improvise_house_building_2_deconstruct' : createImproviseHouseBuilding2DeconstructEnv,
  'improvise_house_building_3_deconstruct' : createImproviseHouseBuilding3DeconstructEnv,
  'improvise_house_building_discrete_deconstruct' : createImproviseHouseBuildingDiscreteDeconstructEnv,
  'improvise_house_building_random_deconstruct' : createImproviseHouseBuildingRandomDeconstructEnv,
  'random_picking' : createRandomPickingEnv,
  'random_stacking' : createRandomStackingEnv,
  'multi_task' : createMultiTaskEnv,
  'ramp_block_stacking': createRampBlockStackingEnv,
  'ramp_house_building_1': createRampHouseBuilding1Env,
  'ramp_house_building_2': createRampHouseBuilding2Env,
  'ramp_house_building_3': createRampHouseBuilding3Env,
  'ramp_house_building_4': createRampHouseBuilding4Env,
  'ramp_improvise_house_building_2': createRampImproviseHouseBuilding2Env,
  'ramp_improvise_house_building_3': createRampImproviseHouseBuilding3Env,
  'ramp_block_stacking_deconstruct': createRampBlockStackingDeconstructEnv,
  'ramp_house_building_1_deconstruct': createRampHouseBuilding1DeconstructEnv,
  'ramp_house_building_2_deconstruct': createRampHouseBuilding2DeconstructEnv,
  'ramp_house_building_3_deconstruct': createRampHouseBuilding3DeconstructEnv,
  'ramp_house_building_4_deconstruct': createRampHouseBuilding4DeconstructEnv,
  'ramp_improvise_house_building_2_deconstruct': createRampImproviseHouseBuilding2DeconstructEnv,
  'ramp_improvise_house_building_3_deconstruct': createRampImproviseHouseBuilding3DeconstructEnv,
  'two_view_drawer_teapot': createTwoViewDrawerTeapotEnv,
  'multi_view_drawer_teapot': createMViewDrawerTeapotEnv,
  'cup_stacking': createCupStackingEnv,
  'bowl_stacking': createBowlStackingEnv,
  'shelf_bowl_stacking': createShelfBowlStackingEnv,
  'shelf_plate_stacking': createShelfPlateStackingEnv,
  'drawer_shelf_plate_stacking': createDrawerShelfPlateStackingEnv,
  'block_bin_packing': createBlockBinPackingEnv,
  'random_block_picking': createRandomBlockPickingEnv,
  'random_household_picking': createRandomHouseholdPickingEnv,
}