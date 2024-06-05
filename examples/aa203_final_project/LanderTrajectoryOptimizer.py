"""
author: xmaple
date: 2021-11-06
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from lib.core.Entrance import PySCP
from lib.utils.PlanetData import Earth, Moon
from lib.utils.Scale import Scale
from lib.utils.Structs import PhaseInfo, Linkage
import pickle


from examples.aa203_final_project.project_constants import *
import jax
import jax.numpy as jnp
from .MoonLanderThreeDof import MoonLanderThreeDof
from .MissionConfigThreeDof import MissionConfigThreeDof


def run():

    num_permutations = 1
    # path relative to main.py file...
    data_filename = 'report'
    subfolder = 'monte_carlo/'
    data_dir = "examples/aa203_final_project/data/"
    if not os.path.exists(data_dir+subfolder): 
        # if the demo_folder directory is not present  
        # then create it. 
        os.makedirs(data_dir+subfolder) 
        
    # mission_config = MissionConfigThreeDof()
    for perm_num in range(num_permutations):
        mission_config = MissionConfigThreeDof()
        # mission_config.apply_dispersion()
        print(mission_config.phase_config_list[0].init_state_bound)
        vehicle = MoonLanderThreeDof(mission_config)
        # mission_config = MissionConfigSixDof()
        # vehicle = MoonLanderSixDof(mission_config)
        PSConfigs = []
        for n in range(len(mission_config.N_phases)):
            PSConfigs.append ({
                'meshName': 'ZOH',  # discretization method
                'ncp': mission_config.N_phases[n],  # number of discrete points
            })
        
        setup = {
            'model': vehicle,
            'meshConfig': PSConfigs,
            'initialization': 'linear', # [linear, specify, integration] are the options
            'maxIteration': 2,
            'verbose': 2,
            'weightNu': [3e3]*len(mission_config.N_phases), # cost on dynamics violations
            'weightTrustRegion': [5e-4]*len(mission_config.N_phases),
            'weightPath': [1e2]*len(mission_config.N_phases),
            'weightBoundary': [1e2]*len(mission_config.N_phases),
            'weightLinkage': 1e2,
            # 'plot_interval': 1
        }


        prob = PySCP(setup)
        prob.solve()
        # prob.plotXU(traj=prob.result.solution, show=True) # non-dimensional, -1 to 1, 1 to 3, 3 to 5
        # prob.plotXU(traj=prob.result.solutionIntegrated, show=False)
        prob.plotXU(traj=prob.result.solutionDimension,
                    show=True,
                    save=False,
                    # matlab_path='examples/aa203_final_project/project.mat',
                    state_name=vehicle.state_labels,
                    control_name=vehicle.control_labels,
                    legend=['solutionDimension (PySCP solver ref trajectory)'])
                    # legend=['solutionIntegrated (segment-wise open loop)', 'solutionDimension (PySCP solver ref trajectory)'])
        prob.print()
        # prob.plotPeroformanceHistory()
        
        continue
        
        def save_pickle(path, fname, data):
            pfile = open(path+fname, 'ab')
            b = pickle.dumps(data)
            pickle.dump(b, pfile)
            pfile.close()
            
        def load_pickle(path, fname):
            pfile = open(path+fname, 'rb')    
            b = pickle.load(pfile)
            var = pickle.loads(b)
            return var

        run_data = {
            'sol_dim': prob.result.solutionDimension,
            'sol_int': prob.result.solutionIntegrated,
            'sol': prob.result.solution,
            'err_hist': prob.result.errorHistory,
            'obj': prob.result.objective,
            'state_labels': vehicle.state_labels,
            'control_labels': vehicle.control_labels,
            'psconfigs': PSConfigs,
            'mconfig': mission_config,
        }

        # save_pickle(data_dir+subfolder, f'{data_filename}_{perm_num}', run_data)


# print('test')
# tmp = MissionConfigThreeDof()
# ax = plt.figure().add_subplot(projection='3d')
# # draw cube
# r = [-1, 1]
# for s, e in combinations(np.array(list(product(r, r, r))), 2):
#     if np.sum(np.abs(s-e)) == r[1]-r[0]:
#         ax.plot(*zip(s, e), color="b")



if __name__ == '__main__':
    run()


