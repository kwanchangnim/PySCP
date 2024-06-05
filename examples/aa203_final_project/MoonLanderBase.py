"""
author: xmaple
date: 2021-11-06
"""
import matplotlib.pyplot as plt
import numpy as np
from lib.core.Entrance import PySCP
from lib.utils.PlanetData import Earth, Moon
from lib.utils.Scale import Scale
from lib.utils.Structs import PhaseInfo, Linkage

from examples.aa203_final_project.project_constants import *
import jax
import jax.numpy as jnp

# https://github.com/xmaple04/PySCP

class MoonLander:
    n_state = None
    m_control = None
    scale = None
    def __init__(self, mission_config):
        if self.n_state is None or self.m_control is None:
            raise RuntimeError('n and m need to be set in child class init')
        
        self.mission_config = mission_config

        # scale
        if self.scale is None:
            self.scale = Scale(state_name=['non']*self.n_state, control_name=['non']*self.m_control)
        self.phases = []
        self.linkages = []
        self.control_linkages = []

        # setup each phase
        for phase_idx, phase_config in enumerate(self.mission_config.phase_config_list):
            phase = PhaseInfo()
            phase.init_state_bound = phase_config.init_state_bound
            phase.final_state_bound = phase_config.final_state_bound
            phase.state_bound = phase_config.state_bound
            phase.control_bound = phase_config.control_bound
            phase.t0_bound = phase_config.t0_bound
            phase.tf_bound = phase_config.tf_bound
            phase.dynamicsFunc = self.Dynamics
            phase.pathFunc = self.Path
            phase.scale = self.scale
            phase.trState = np.ones((self.n_state,)) * 1
            phase.trControl = np.ones((self.m_control,)) * 1
            phase.trSigma = 1    
            self.phases.append(phase)

            # link states to the previous phase if one existss
            if phase_idx > 0:
                for link_idx in range(len(self.mission_config.all_linkage_config.lower_bound)):
                    lb = self.mission_config.all_linkage_config.lower_bound[link_idx]
                    ub = self.mission_config.all_linkage_config.upper_bound[link_idx]
                    link = Linkage(phase_idx-1, phase_idx, [link_idx], np.array([lb,ub]))
                    self.linkages.append(link)
            
            # link controls to the previous phase if one existss
            if phase_idx > 0:
                for link_idx in range(len(self.mission_config.all_control_linkage_config.lower_bound)):
                    lb = self.mission_config.all_control_linkage_config.lower_bound[link_idx]
                    ub = self.mission_config.all_control_linkage_config.upper_bound[link_idx]
                    link = Linkage(phase_idx-1, phase_idx, [link_idx], np.array([lb,ub]))
                    self.control_linkages.append(link)
        
        self.dynamics_jacobian = jax.jacobian(self.jnp_dynamics, argnums=(0, 1))

    def jnp_dynamics(self, x, u):
        raise NotImplementedError()
    
    def Dynamics(self, x, u, t, auxdata):
        raise NotImplementedError()

    def Path(self, phases):
        path = None
        return [path]

    def objective(self, Vars, refTraj, weights4State, weights4Control):
        # weights4State and weights4Control are for psuedospectral approximations

        objective = 0
        for phase_index in range(len(Vars)):
            R = self.mission_config.R_p[phase_index]
            segments = Vars[phase_index].control.shape[0]
            delta_tau = 2/segments
            for i in range(segments):
                # unpack vars for convenience
                # x_ref = refTraj[phase_index].state[i]
                # x_var = refTraj[phase_index].state[i]
                sigma_ref = refTraj[phase_index].sigma
                sigma_var = refTraj[phase_index].sigma
                u_ref = refTraj[phase_index].control[i]
                u_var = refTraj[phase_index].control[i]

                # sum up the cost function sigma*u.T @ R @ u linearized about the trajectory
                objective += 1/2*sigma_ref*np.transpose(u_ref) @ R @ u_ref * delta_tau
                objective += sigma_ref*np.transpose(u_ref) @ R * delta_tau @ (u_var - u_ref)
                objective += 1/2*np.transpose(u_ref) @ R @ u_ref*delta_tau*(sigma_var-sigma_ref)

        return objective
