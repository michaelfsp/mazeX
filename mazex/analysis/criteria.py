"""Here come methods to select sessions, trials or steps based on criteria.
"""
def first_exit_steps_only(session):
    """Return a session where the paths have been modified such that in any
    given trial, only the first visit to a node is represented.
    """

def per_dist_avg_dir_prob(session, first_exit_only=True,
        availability='plot', plot=False):
    """Calculate the probability of direction choice on the distances space.
    Right now the way the algorithm is the following:
    
       For every point in the distances space, calculate for every direction
       number of times it is available (availability) and the number of times
       it is taken (moves) and return the 'probability' as moves/availability.
    """
    # session.maze will return a list if not all trials use the same maze
    # in which case it will be iterable
    max_dists = np.array([utils.max_distances(t.maze, t.goal) for t in session.trials]
            ).max(0)

    dir_surprisals_dims = max_dists + 1
    dir_surprisals_dims = dir_surprisals_dims.tolist()
    dir_surprisals_dims.append(4)
    dir_surprisals = np.zeros(dir_surprisals_dims)
    dir_availabilities = np.zeros(dir_surprisals_dims)

    x, y = np.meshgrid(range(dir_surprisals.shape[0]), range(dir_surprisals.shape[1]))
    xy = np.array([x.flatten(), y.flatten()]).transpose()

    for dist_coord in xy:
        #import pdb; pdb.set_trace()
        surpr, avail = dist_dir_prob(session, tuple(dist_coord),
                first_exit_only=first_exit_only)
        dir_surprisals[dist_coord[0], dist_coord[1], :] = surpr
        dir_availabilities[dist_coord[0], dist_coord[1], :] = avail

    if not plot:
        return dir_surprisals, dir_availabilities
    else:
        distances_plot(session.trial.maze, [t.goal for t in session.trials],
                plot=True, zorder=10)
        pl.hold(True)
        dx_sign = (1, 1, -1, -1)
        dy_sign = (1, -1, -1, 1)
        # scale factor that will make the maximum length of the arrow 0.5 when
        # projected to either x or y axis. 0.75 = np.sqrt(1**2/2), and 1 is
        # obviously the maximum...
        #FIXME

        if availability == 'normalizer':
            # maximum would be 1, since the surprisals are normalized
            # this is simply the value of the projection of a vector of length
            # 1 and angle 45 degr. onto x or y axes...
            max_proj = np.sqrt(1/2)
        else:
            # if not normalized the maximum surprisal is 2
            max_proj = np.sqrt((2**2)/2)
        
        # this is simply a scaling factor so that the arrows don't overlap on
        # the plot
        scale = 0.5/max_proj

        for direction in range(0, 4):
            surprisals_xy_proj = scale*np.sqrt(dir_surprisals[:,:,direction
                ]**2/2).transpose().flatten()
            pl.quiver(xy[:,0], xy[:,1], dx_sign[direction]*surprisals_xy_proj,
                   dy_sign[direction]*surprisals_xy_proj, units='xy', scale=1,
                   color='b', zorder=2) 

            if availability == 'plot':
                availabilities_xy_proj = scale*np.sqrt(dir_availabilities[:,:,direction
                    ]**2/2).transpose().flatten()
                pl.quiver(xy[:,0], xy[:,1], dx_sign[direction]*availabilities_xy_proj,
                       dy_sign[direction]*availabilities_xy_proj, units='xy', scale=1,
                       color='r', zorder=1) 


