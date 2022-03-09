from helping_hands_rl_envs.envs.multi_task_env import createMultiTaskEnv

from helping_hands_rl_envs.envs.block_structure_envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.block_structure_envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.block_structure_envs.pyramid_stacking_env import createPyramidStackingEnv
from helping_hands_rl_envs.envs.block_structure_envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.block_structure_envs.house_building_2_env import createHouseBuilding2Env
from helping_hands_rl_envs.envs.block_structure_envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.block_structure_envs.house_building_4_env import createHouseBuilding4Env
from helping_hands_rl_envs.envs.block_structure_envs.house_building_5_env import createHouseBuilding5Env
from helping_hands_rl_envs.envs.block_structure_envs.house_building_x_env import createHouseBuildingXEnv
from helping_hands_rl_envs.envs.block_structure_envs.improvise_house_building_2_env import createImproviseHouseBuilding2Env
from helping_hands_rl_envs.envs.block_structure_envs.improvise_house_building_3_env import createImproviseHouseBuilding3Env
from helping_hands_rl_envs.envs.block_structure_envs.improvise_house_building_discrete_env import createImproviseHouseBuildingDiscreteEnv
from helping_hands_rl_envs.envs.block_structure_envs.improvise_house_building_random_env import createImproviseHouseBuildingRandomEnv

from helping_hands_rl_envs.envs.deconstruct_envs.block_stacking_deconstruct_env import createBlockStackingDeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.house_building_1_deconstruct_env import createHouseBuilding1DeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.house_building_2_deconstruct_env import createHouseBuilding2DeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.house_building_3_deconstruct_env import createHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.house_building_4_deconstruct_env import createHouseBuilding4DeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.house_building_x_deconstruct_env import createHouseBuildingXDeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.improvise_house_building_2_deconstruct_env import createImproviseHouseBuilding2DeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.improvise_house_building_3_deconstruct_env import createImproviseHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.improvise_house_building_discrete_deconstruct_env import createImproviseHouseBuildingDiscreteDeconstructEnv
from helping_hands_rl_envs.envs.deconstruct_envs.improvise_house_building_random_deconstruct_env import createImproviseHouseBuildingRandomDeconstructEnv

from helping_hands_rl_envs.envs.realistic_envs.object_grasping import createObjectGrasping
from helping_hands_rl_envs.envs.realistic_envs.block_picking_env import createBlockPickingEnv
from helping_hands_rl_envs.envs.realistic_envs.block_bin_packing_env import createBlockBinPackingEnv
from helping_hands_rl_envs.envs.realistic_envs.random_block_picking_env import createRandomBlockPickingEnv
from helping_hands_rl_envs.envs.realistic_envs.random_household_picking_env import createRandomHouseholdPickingEnv
from helping_hands_rl_envs.envs.realistic_envs.random_block_picking_clutter_env import createRandomBlockPickingClutterEnv
from helping_hands_rl_envs.envs.realistic_envs.random_household_picking_clutter_env import createRandomHouseholdPickingClutterEnv
from helping_hands_rl_envs.envs.realistic_envs.bottle_tray_env import createBottleTrayEnv
from helping_hands_rl_envs.envs.realistic_envs.box_palletizing_env import createBoxPalletizingEnv
from helping_hands_rl_envs.envs.realistic_envs.covid_test_env import createCovidTestEnv

from helping_hands_rl_envs.envs.ramp_envs.ramp_block_stacking_env import createRampBlockStackingEnv
from helping_hands_rl_envs.envs.ramp_envs.ramp_house_building_1_env import createRampHouseBuilding1Env
from helping_hands_rl_envs.envs.ramp_envs.ramp_house_building_2_env import createRampHouseBuilding2Env
from helping_hands_rl_envs.envs.ramp_envs.ramp_house_building_3_env import createRampHouseBuilding3Env
from helping_hands_rl_envs.envs.ramp_envs.ramp_house_building_4_env import createRampHouseBuilding4Env
from helping_hands_rl_envs.envs.ramp_envs.ramp_improvise_house_building_2_env import createRampImproviseHouseBuilding2Env
from helping_hands_rl_envs.envs.ramp_envs.ramp_improvise_house_building_3_env import createRampImproviseHouseBuilding3Env
from helping_hands_rl_envs.envs.ramp_envs.ramp_block_stacking_deconstruct_env import createRampBlockStackingDeconstructEnv
from helping_hands_rl_envs.envs.ramp_envs.ramp_house_building_1_deconstruct_env import createRampHouseBuilding1DeconstructEnv
from helping_hands_rl_envs.envs.ramp_envs.ramp_house_building_2_deconstruct_env import createRampHouseBuilding2DeconstructEnv
from helping_hands_rl_envs.envs.ramp_envs.ramp_house_building_3_deconstruct_env import createRampHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.ramp_envs.ramp_house_building_4_deconstruct_env import createRampHouseBuilding4DeconstructEnv
from helping_hands_rl_envs.envs.ramp_envs.ramp_improvise_house_building_2_deconstruct_env import createRampImproviseHouseBuilding2DeconstructEnv
from helping_hands_rl_envs.envs.ramp_envs.ramp_improvise_house_building_3_deconstruct_env import createRampImproviseHouseBuilding3DeconstructEnv

from helping_hands_rl_envs.envs.bumpy_envs.bumpy_box_palletizing_env import createBumpyBoxPalletizingEnv
from helping_hands_rl_envs.envs.bumpy_envs.bumpy_house_building_4_env import createBumpyHouseBuilding4Env

from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_picking import createCloseLoopBlockPickingEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_reaching import createCloseLoopBlockReachingEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_stacking import createCloseLoopBlockStackingEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_pulling import createCloseLoopBlockPullingEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_house_building_1 import createCloseLoopHouseBuilding1Env
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_picking_corner import createCloseLoopBlockPickingCornerEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_drawer_opening import createCloseLoopDrawerOpeningEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_household_picking import createCloseLoopHouseholdPickingEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_household_picking_cluttered import createCloseLoopHouseholdPickingClutteredEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_pushing import createCloseLoopBlockPushingEnv
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_in_bowl import createCloseLoopBlockInBowlEnv

from helping_hands_rl_envs.envs.force_envs.force_block_picking import createForceBlockPickingEnv
from helping_hands_rl_envs.envs.force_envs.force_block_pulling import createForceBlockPullingEnv
from helping_hands_rl_envs.envs.force_envs.force_block_picking_corner import createForceBlockPickingCornerEnv

CREATE_ENV_FNS = {
  'block_picking' : createBlockPickingEnv,
  'block_stacking' : createBlockStackingEnv,
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
  'block_stacking_deconstruct': createBlockStackingDeconstructEnv,
  'house_building_1_deconstruct' : createHouseBuilding1DeconstructEnv,
  'house_building_2_deconstruct' : createHouseBuilding2DeconstructEnv,
  'house_building_3_deconstruct' : createHouseBuilding3DeconstructEnv,
  'house_building_4_deconstruct' : createHouseBuilding4DeconstructEnv,
  'house_building_x_deconstruct' : createHouseBuildingXDeconstructEnv,
  'improvise_house_building_2_deconstruct' : createImproviseHouseBuilding2DeconstructEnv,
  'improvise_house_building_3_deconstruct' : createImproviseHouseBuilding3DeconstructEnv,
  'improvise_house_building_discrete_deconstruct' : createImproviseHouseBuildingDiscreteDeconstructEnv,
  'improvise_house_building_random_deconstruct' : createImproviseHouseBuildingRandomDeconstructEnv,
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
  'object_grasping': createObjectGrasping,
  'block_bin_packing': createBlockBinPackingEnv,
  'random_block_picking': createRandomBlockPickingEnv,
  'random_household_picking': createRandomHouseholdPickingEnv,
  'random_block_picking_clutter': createRandomBlockPickingClutterEnv,
  'random_household_picking_clutter': createRandomHouseholdPickingClutterEnv,
  'bottle_tray': createBottleTrayEnv,
  'box_palletizing': createBoxPalletizingEnv,
  'bumpy_box_palletizing': createBumpyBoxPalletizingEnv,
  'bumpy_house_building_4': createBumpyHouseBuilding4Env,
  'covid_test': createCovidTestEnv,
  'close_loop_block_picking': createCloseLoopBlockPickingEnv,
  'close_loop_block_reaching': createCloseLoopBlockReachingEnv,
  'close_loop_block_stacking': createCloseLoopBlockStackingEnv,
  'close_loop_block_pulling': createCloseLoopBlockPullingEnv,
  'close_loop_house_building_1': createCloseLoopHouseBuilding1Env,
  'close_loop_block_picking_corner': createCloseLoopBlockPickingCornerEnv,
  'close_loop_drawer_opening': createCloseLoopDrawerOpeningEnv,
  'close_loop_household_picking': createCloseLoopHouseholdPickingEnv,
  'close_loop_clutter_picking': createCloseLoopHouseholdPickingClutteredEnv,
  'close_loop_block_pushing': createCloseLoopBlockPushingEnv,
  'close_loop_block_in_bowl': createCloseLoopBlockInBowlEnv,
  'force_block_picking' : createForceBlockPickingEnv,
  'force_block_pulling' : createForceBlockPullingEnv,
  'force_block_picking_corner' : createForceBlockPickingCornerEnv,
}
