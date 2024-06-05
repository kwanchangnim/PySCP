import numpy as np
from lib.utils.PlanetData import Moon
from .ConstraintConfig import LinkageConfig, ConstraintConfig
from .project_constants import *
from .MissionConfigBase import MissionConfig
import matplotlib.pyplot as plt
from itertools import product, combinations


class MissionConfigThreeDof(MissionConfig):

    def __init__(self):
        super().__init__()

        self.all_control_bound = np.array([[-9999, -9999,-9999], 
                                           [ 9999,  9999, 9999]])
        # self.all_control_bound = np.array([[-np.pi*2, -np.pi*2,     0], 
        #                                    [ np.pi*2,  np.pi*2, 15000]])

        # each list should have the same number of elemetns as the state 
        self.all_linkage_config = LinkageConfig(
            [-0.01, -0.01, -0.01, -0.001, -0.001, -0.001],
            [ 0.01,  0.01,  0.01,  0.001,  0.001,  0.001]
        )
        # each list should have the same number of elemetns as the control 
        self.all_control_linkage_config = LinkageConfig(
            [-0.01, -0.01, -0.01],
            [ 0.01,  0.01,  0.01]
        )

        descent_approach_az, alt_parking_orbit, alt_braking, range_braking, rx_brk_0, ry_brk_0, rz_brk_0, range_rate_brk_0, vx_brk_0, vy_brk_0, vz_brk_0,rx_app_0, ry_app_0, rz_app_0, range_rate_app_0, vx_app_0, vy_app_0, vz_app_0 = get_dispersed()

        # BRAKE PHASE CONFIG
        braking_isb = np.array([[rx_brk_0, ry_brk_0, rz_brk_0, vx_brk_0, vy_brk_0, vz_brk_0], 
                                [rx_brk_0, ry_brk_0, rz_brk_0, vx_brk_0, vy_brk_0, vz_brk_0]])
        braking_fsb = np.array([[rx_app_0-200, ry_app_0-200, alt_approach-200, vx_app_0-30, vy_app_0-30, alt_rate_approach-5], 
                                [rx_app_0+200, ry_app_0+200, alt_approach+200, vx_app_0+30, vy_app_0+30, alt_rate_approach+5]])
        braking_sb = np.array([[-range_braking, -range_braking, alt_approach-200, -velocity_braking, -velocity_braking, -velocity_braking], 
                                [ range_braking,  range_braking, alt_braking+500,  velocity_braking,  velocity_braking,  velocity_braking]])
        braking_t0b = np.array([0, 0])
        braking_tfb = np.array([500, 2000])
        braking_config = ConstraintConfig(braking_isb, braking_fsb, braking_sb, self.all_control_bound, braking_t0b, braking_tfb)
        R_braking = np.diag(np.array([1.0, 1.0, 1.0]))*300
        N_braking = 25

        self.R_p.append(R_braking) # ORDER MATTERS!
        self.N_phases.append(N_braking)
        self.phase_config_list.append(braking_config)

        # APPOACH PHASE CONFIG
        # approach_isb = np.array([[rx_app_0, ry_app_0, rz_app_0, vx_app_0, vy_app_0, vz_app_0], 
        #                          [rx_app_0, ry_app_0, rz_app_0, vx_app_0, vy_app_0, vz_app_0]])
        approach_isb = braking_fsb
        approach_fsb = np.array([[-10, -10, alt_land-10, -3, -3, alt_rate_land-0.5], 
                                [ 10,  10, alt_land+10,  3,  3, alt_rate_land+0.5]])
        approach_sb = np.array([[-range_approach, -range_approach, alt_land-10, -vel_approach, -vel_approach, -vel_approach], 
                                [ range_approach,  range_approach, alt_approach+100,  vel_approach,  vel_approach,  vel_approach]])
        approach_t0b = np.array([0, 0])
        approach_tfb = np.array([5, 600])
        approach_config = ConstraintConfig(approach_isb, approach_fsb, approach_sb, self.all_control_bound, approach_t0b, approach_tfb)
        R_approach = np.diag(np.array([1.0, 1.0, 1.0]))*100
        N_approach = 30

        self.R_p.append(R_approach)
        self.N_phases.append(N_approach)
        self.phase_config_list.append(approach_config)

        # TERMINAL DESCENT PHASE CONFIG
        descent_isb = approach_fsb
        # descent_isb = np.array([[-5, -5, alt_land, 1, 1, alt_rate_land], 
        #                         [-5, -5, alt_land, 1, 1, alt_rate_land]])
        descent_fsb= np.array([[0, 0, 0, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0]])
        descent_sb = np.array([[-10, -10,        0, -100, -100, -100], 
                            [ 10,  10, alt_land+10,  100,  100,  100]])
        descent_t0b = np.array([0, 0])
        descent_tfb = np.array([5, 100])
        descent_config = ConstraintConfig(descent_isb, descent_fsb, descent_sb, self.all_control_bound, descent_t0b, descent_tfb)
        R_descent =  np.diag(np.array([1.0, 1.0, 1.0]))*100
        N_descent = 30

        self.R_p.append(R_descent)
        self.N_phases.append(N_descent)
        self.phase_config_list.append(descent_config)