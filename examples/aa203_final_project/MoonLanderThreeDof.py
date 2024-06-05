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
from .MoonLanderBase import MoonLander

from examples.aa203_final_project.project_constants import *
import jax
import jax.numpy as jnp
    
class MoonLanderThreeDof(MoonLander):
    
    def __init__(self, mission_config):
        self.n_state = 6
        self.m_control = 3
        state_scale_names = ['length', 'length', 'length', 'velocity', 'velocity', 'velocity']
        control_scale_names = ['force', 'force', 'force']
        self.scale = Scale(state_name=state_scale_names, control_name=control_scale_names)
        
        super().__init__(mission_config)

        print(self.phases)

        # labels for plotting
        self.state_labels = [
            r'$x$', r'$y$', r'$z$', 
            r'$vx$', r'$vy$', r'$vz$'
        ]
        self.control_labels = [
            r'$Tx$', r'$Ty$', r'$Tz$'
        ]

    def jnp_dynamics(self, x, u):
        Tx = u[0] # thrust in each axis
        Ty = u[1]
        Tz = u[2]
        # r = x[0:3]      # m 
        v = x[3:6]        # m/s 
        mass = mass_0     # kg
        
        r_dot = v
        v_dot = jnp.array([
            1/mass*Tx,
            1/mass*Ty,
            1/mass*Tz - Moon.g0
        ])
        return jnp.hstack((r_dot, v_dot))
    
    def Dynamics(self, x, u, t, auxdata):
        Tx = u[0] # thrust in each axis
        Ty = u[1]
        Tz = u[2]
        # r = x[0:3]      # m 
        v = x[3:6]        # m/s 
        mass = mass_0     # kg
        
        r_dot = v
        v_dot = np.array([
            1/mass*Tx,
            1/mass*Ty,
            1/mass*Tz - Moon.g0
        ])
        f = np.hstack((r_dot, v_dot))
        A, B = self.dynamics_jacobian(x,u)
        return f, A, B