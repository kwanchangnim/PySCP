import numpy as np
import matplotlib.pyplot as plt
from lib.utils.Structs import PhaseData

def plot_3d_flat(traj, **kwargs):

    # if kwargs.get('show_segments'):
    state_t_c, state_c, control_t_c, control_c = concat_traj(traj)

    if kwargs.get('ax') is not None:
        ax = kwargs.get('ax')
    else:
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    ax.plot(state_c[:, 0], state_c[:, 1], state_c[:, 2], label='position')

    # if initial_conditions is not None:
    #     ax.scatter(initial_conditions[0], initial_conditions[1], initial_conditions[2], color="g")
    # if final_conditions is not None:
    #     ax.scatter(final_conditions[0], final_conditions[1], final_conditions[2], color="g")

    if kwargs.get('show_axes'):
        x, y, z = np.zeros((3,3))
        u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
        ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1)

    if kwargs.get('show'):
        # ax.axis('equal')
        plt.show()

    return ax

def plot_alt_range(traj, **kwargs):

    # if kwargs.get('show_segments'):
    state_t_c, state_c, control_t_c, control_c = concat_traj(traj)

    
    n_phases = len(traj)
    range_f = np.zeros((n_phases,))
    alt_f = np.zeros((n_phases,))
    for iPhase in range(n_phases-1):
        if isinstance(traj[iPhase], PhaseData):
            xt = traj[iPhase].xtao
            state = traj[iPhase].state
            ut = traj[iPhase].utao
            control = traj[iPhase].control
        else:
            xt, state, ut, control = traj[iPhase]

        range_f[iPhase] = np.linalg.norm(state[-1,:2])
        alt_f[iPhase] = state[-1,2]


    if kwargs.get('ax') is not None:
        ax = kwargs.get('ax')
    else:
        ax = plt.figure().add_subplot()
        ax.set_xlabel('range')
        ax.set_ylabel('altitude')
    
    rng = np.linalg.norm(state_c[:, :2], axis=1)

    ax.plot(rng, state_c[:, 2])

    if kwargs.get('show_boxes'):
        if kwargs.get('mission_config'):
            mission_config = kwargs.get('mission_config')
            n_phases = len(traj)
            for idx in range(n_phases):
                fsb = mission_config.phase_config_list[idx].final_state_bound
                # plt.plot([tf[idx],tf[idx]], [fsb[0,ix], fsb[1,ix]])
                range_min = np.linalg.norm(fsb[0,:2])
                range_max = np.linalg.norm(fsb[1,:2])
                if range_min > range_max:
                    r_min = range_max
                    range_max = range_min
                    range_min = r_min
                if range_min == range_max:
                    range_min *= -1
                alt_min = fsb[0,2]
                alt_max = fsb[1,2]
                plot_box(ax, range_min, range_max, alt_min, alt_max)

        if kwargs.get('show_segments'):
            for idx in range(n_phases-1):
                # need the range at the segment boundary
                ax.axvline(range_f[idx], color='grey', linestyle='dashed')
                # ax.axhline(alt_f[idx], color='grey', linestyle='dashed')



    if kwargs.get('show'):
        # ax.axis('equal')
        # TODO flip axis?
        plt.show()

    return ax
    

def concat_traj(traj):
    n_phases = len(traj)
    state_t_c = None
    control_t_c = None
    state_c = None
    control_c = None

    tf = np.zeros((n_phases,))
    for iPhase in range(n_phases):
        if isinstance(traj[iPhase], PhaseData):
            xt = traj[iPhase].xtao
            state = traj[iPhase].state
            ut = traj[iPhase].utao
            control = traj[iPhase].control
        else:
            xt, state, ut, control = traj[iPhase]

        # xdim = state.shape[1]
        # udim = control.shape[1]
        if xt[0] == -1:
            xt = xt + 2*iPhase
            ut = ut + 2*iPhase
        else:
            if iPhase > 0:
                xt = xt + tf[iPhase-1]
                ut = ut + tf[iPhase-1]
        tf[iPhase] = xt[-1]
        
        if state_t_c is None:
            state_t_c = np.copy(xt)
            state_c = np.copy(state)
            control_t_c = np.copy(ut)
            control_c = np.copy(control)
        else:
            state_t_c = np.append(state_t_c, xt)
            state_c = np.vstack((state_c, state))
            control_t_c = np.append(control_t_c, ut)
            control_c = np.vstack((control_c, control))
            

    return state_t_c, state_c, control_t_c, control_c


def plotXU(traj, **kwargs):
    figsize = (10, 6)

    state_t_c, state_c, control_t_c, control_c = concat_traj(traj)

    n_phases = len(traj)
    # plot states in one figure, and controls in another
    tf = np.zeros((n_phases,))
    for iPhase in range(n_phases):
        if isinstance(traj[iPhase], PhaseData):
            xt = traj[iPhase].xtao
            state = traj[iPhase].state
            ut = traj[iPhase].utao
            control = traj[iPhase].control
        else:
            xt, state, ut, control = traj[iPhase]

        xdim = state.shape[1]
        udim = control.shape[1]
        if xt[0] == -1:
            xt = xt + 2*iPhase
            ut = ut + 2*iPhase
        else:
            if iPhase > 0:
                xt = xt + tf[iPhase-1]
                ut = ut + tf[iPhase-1]
        tf[iPhase] = xt[-1]

        stateyscale = np.ones((xdim,))
        controlyscale = np.ones((udim,))
        if kwargs.get('stateyscale') is not None:
            stateyscale = kwargs.get('stateyscale')
        if kwargs.get('controlyscale') is not None:
            controlyscale = kwargs.get('controlyscale')

        grid = False
        if kwargs.get('grid') is not None:
            grid = kwargs.get('grid')
        # if self.setup['plotStyle'] == 'grid':
        if grid:
            plt.figure(1, figsize=figsize)

        xrow = int(np.sqrt(xdim))
        xcol = int(np.ceil(xdim / xrow))
        urow = int(np.sqrt(udim))
        ucol = int(np.ceil(udim / urow))
        for ix in range(xdim):
            # if self.setup['plotStyle'] == 'grid':
            if grid:
                plt.subplot(xrow, xcol, ix + 1)
            else:
                plt.figure(ix, figsize=figsize)
            plt.plot(xt, state[:, ix] * stateyscale[ix], color=kwargs.get('color'), marker=kwargs.get('marker'), linewidth=kwargs.get('linewidth'))

            plt.xlabel('time/s')
            if kwargs.get('state_name'):
                plt.ylabel(kwargs.get('state_name')[ix])
            else:
                plt.ylabel('state {:d}'.format(ix))
            if kwargs.get('legend'):
                plt.legend(kwargs.get('legend'))
        if kwargs.get('save'):
            plt.tight_layout()
            plt.savefig('state.eps', dpi=600)

        # if self.setup['plotStyle'] == 'grid':
        if grid:
            plt.figure(2, figsize=figsize)
        for iu in range(udim):
            # if self.setup['plotStyle'] == 'grid':
            if grid:
                plt.subplot(urow, ucol, iu + 1)
            else:
                plt.figure(xdim + iu, figsize=figsize)
            plt.plot(ut, control[:, iu] * controlyscale[iu], color=kwargs.get('color'), marker=kwargs.get('marker'))
            
            plt.xlabel('time/s')
            if kwargs.get('state_name'):
                plt.ylabel(kwargs.get('control_name')[iu])
            else:
                plt.ylabel('control {:d}'.format(iu))

            if kwargs.get('legend'):
                plt.legend(kwargs.get('legend'))
        if kwargs.get('save'):
            plt.tight_layout()
            plt.savefig('control.eps', dpi=600)

    if kwargs.get('show'):
        plt.show()



def plot_box(ax, x1,x2,y1,y2):
    x = [x1, x1, x2, x2, x1]
    y = [y1, y2, y2, y1, y1]
    ax.plot(x, y, 'blue')


def plotXU_NEW(traj, **kwargs):
    figsize = (10, 6)

    state_t_c, state_c, control_t_c, control_c = concat_traj(traj)

    print(state_c[0])

    xdim = state_c.shape[1]
    udim = control_c.shape[1]

    n_phases = len(traj)
    # plot states in one figure, and controls in another
    tf = np.zeros((n_phases,)) # this is the intersection of phases!
    for iPhase in range(n_phases):
        if isinstance(traj[iPhase], PhaseData):
            xt = traj[iPhase].xtao
            state = traj[iPhase].state
            ut = traj[iPhase].utao
            control = traj[iPhase].control
        else:
            xt, state, ut, control = traj[iPhase]

        xdim = state.shape[1]
        udim = control.shape[1]
        if xt[0] == -1:
            xt = xt + 2*iPhase
            ut = ut + 2*iPhase
        else:
            if iPhase > 0:
                xt = xt + tf[iPhase-1]
                ut = ut + tf[iPhase-1]
        tf[iPhase] = xt[-1]

    plt.figure(1, figsize=figsize)

    xrow = int(np.sqrt(xdim))
    xcol = int(np.ceil(xdim / xrow))
    urow = int(np.sqrt(udim))
    ucol = int(np.ceil(udim / urow))
    for ix in range(xdim):
        plt.subplot(xrow, xcol, ix + 1)
        plt.plot(state_t_c, state_c[:, ix], color=kwargs.get('color'), marker=kwargs.get('marker'), linewidth=kwargs.get('linewidth'))
        
        if kwargs.get('show_boxes'):
            if kwargs.get('mission_config'):
                mission_config = kwargs.get('mission_config')
                for idx in range(n_phases):
                    fsb = mission_config.phase_config_list[idx].final_state_bound
                    plt.plot([tf[idx],tf[idx]], [fsb[0,ix], fsb[1,ix]])

        if kwargs.get('show_segments'):
            for idx in range(n_phases-1):
                plt.axvline(tf[idx], color='grey', linestyle='dashed')

        plt.xlabel('time/s')
        if kwargs.get('state_name'):
            plt.ylabel(kwargs.get('state_name')[ix])
        else:
            plt.ylabel('state {:d}'.format(ix))
        if kwargs.get('legend'):
            plt.legend(kwargs.get('legend'))
    if kwargs.get('save'):
        plt.tight_layout()
        plt.savefig('state.eps', dpi=600)

    plt.figure(2, figsize=figsize)
    for iu in range(udim):
        plt.subplot(urow, ucol, iu + 1)
        plt.plot(control_t_c, control_c[:, iu], color=kwargs.get('color'), marker=kwargs.get('marker'))
        
        if kwargs.get('show_segments'):
            for idx in range(n_phases-1):
                plt.axvline(tf[idx], color='grey', linestyle='dashed')

        plt.xlabel('time/s')
        if kwargs.get('state_name'):
            plt.ylabel(kwargs.get('control_name')[iu])
        else:
            plt.ylabel('control {:d}'.format(iu))

        if kwargs.get('legend'):
            plt.legend(kwargs.get('legend'))
    if kwargs.get('save'):
        plt.tight_layout()
        plt.savefig('control.eps', dpi=600)

    if kwargs.get('show'):
        plt.show()


    