"""
author: xmaple
date: 2021-11-06
"""
import pickle


from examples.aa203_final_project.project_constants import *
from .plot_helper import *


def analyze():

    # path relative to main.py file...
    data_filename = 'report'
    subfolder = 'monte_carlo/'
    data_dir = "examples/aa203_final_project/data/"
    # if not os.path.exists(data_dir+subfolder): 
    #     # if the demo_folder directory is not present  
    #     # then create it. 
    #     os.makedirs(data_dir+subfolder) 

    
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

    # run_data = {
    #     'sol_dim': prob.result.solutionDimension,
    #     'sol_int': prob.result.solutionIntegrated,
    #     'sol': prob.result.solution,
    #     'err_hist': prob.result.errorHistory,
    #     'obj': prob.result.objective,
    #     'state_labels': vehicle.state_labels,
    #     'control_labels': vehicle.control_labels,
    #     'psconfigs': PSConfigs,
    #     'mconfig': mission_config,
    # }

    # concat data

    num_files = 10
    for num in range(num_files):

        run_data = load_pickle(data_dir+subfolder, f'{data_filename}_{num}')
        
        state_labels = run_data['state_labels']
        control_labels = run_data['control_labels']
        solutionDimension = run_data['sol_dim']
        mission_config = run_data['mconfig']
        
    #     if num < num_files-1:
    #         plotXU_NEW(traj=solutionDimension,
    #                 )
    #     else:
    #         plotXU_NEW(traj=solutionDimension,
    #                 show=True,
    #                 save=False,
    #                 state_name=state_labels,
    #                 control_name=control_labels,
    #                 show_segments=True,
    #                 # show_boxes=True,
    #                 # mission_config=mission_config
    #                 # legend=['solutionDimension (PySCP solver ref trajectory)'],
    #                 # grid=True
    #                 )


    # ax_3d = None

    # for num in range(num_files):

    #     run_data = load_pickle(data_dir+subfolder, f'{data_filename}_{num}')


    #     state_labels = run_data['state_labels']
    #     control_labels = run_data['control_labels']
    #     solutionDimension = run_data['sol_dim']

    #     if num <num_files-1:
    #         ax_3d = plot_3d_flat(traj=solutionDimension, ax = ax_3d)
    #     else:
    #         plot_3d_flat(traj=solutionDimension,
    #                      show_axes=True,
    #                      show=True,
    #                      ax=ax_3d)

    ax_range = None

    for num in range(num_files):

        run_data = load_pickle(data_dir+subfolder, f'{data_filename}_{num}')


        state_labels = run_data['state_labels']
        control_labels = run_data['control_labels']
        solutionDimension = run_data['sol_dim']

        if num <num_files-1:
            ax_range = plot_alt_range(traj=solutionDimension, ax = ax_range)
        else:
            plot_alt_range(traj=solutionDimension,
                         show_axes=True,
                         show=True,
                         ax=ax_range,
                         show_boxes=True,
                         mission_config=mission_config,
                         show_segments=True
                        )




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


