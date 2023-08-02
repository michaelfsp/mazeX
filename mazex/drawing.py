"""
Visually represent mazes and trials
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.datamodel import Step, Trial
from mazex.analysis.distances import distances_plot

from seaborn import cubehelix_palette

# other interesting node colors '#848599'
def get_cubehelix_RGB(x, nsteps=10, vmin=0, vmax=1, scheme='green',
        hexadecimal=False):
    """Maps a scalar to an RGB tuple from the seaborn version of cubehelix.
    """
    if scheme == 'green':
        colors = cubehelix_palette(nsteps, start=.5, rot=-.75)
    elif scheme == 'pink':
        colors = cubehelix_palette(nsteps)

    # Normalize x to the range [0,1]
    x = (x - vmin) / (vmax - vmin)
    
    # Find closest index
    color_index = np.argmin(abs(x - (np.arange(nsteps)+0.5)))
    
    x_colors = colors[color_index]
    if hexadecimal:
        hex_color = '#'
        for c in x_colors:
            hex_color += hex(int(c*256)).rpartition('x')[2]
        x_colors = hex_color
    import ipdb; ipdb.set_trace() # BREAKPOINT
         
    return x_colors

# These are the cubehelix settings suggested by seaborn
#green_cubehelix_cmap = cubehelix.cmap(reverse=True, start=0, rot=0.4)
#pink_cubehelix_cmap = cubehelix.cmap(reverse=True, start=0.5, rot=-0.75)

def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height,
        fontsize):
    
    """Helper function for creating a legend for matplotlib arrow objects.
    """
    import matplotlib.patches as mpatches

    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True,
            head_width=0.75*height )
    return p

def draw_maze(M, centrality=None, edge_centrality=True, simple_nodes=True,
        nvmin=None, nvmax=None, evmin=None, evmax=None, colorbars=False,
        ax=None, **kwds):
    """Draw a maze configuration.

    Optional parameters:
    --------------------
    edge_centrality : boolean
        plotting edge centrality is now off by default

    simple_nodes : boolean
        if True, a single circle instead of the classic two concentric circles
            is used to represent a node

    nvmin : float 
        value for node centrality colormap normalization

    nvmax : float 
        value for node centrality colormap normalization

    evmin : float 
        value for edge centrality colormap normalization

    evmax : float 
        value for edge centrality colormap normalization

    colorbars : boolean
        if True, the colorbars indicating the scale of values of centrality is
            plotted next to the maze depiction.
    """
    #TODO: group all options desireable for a colormap in one function argument
    if not isinstance(M, SquareMaze):
        raise DrawException
    if not ax:
        ax = plt.gca()
    # Makes sure the drawings are square and not rectangular
    ax.set_aspect('equal')

    if np.all(centrality == None):
        if isinstance(M, SquareMaze):
            nx.draw(M, pos=M.pos, ax=ax, node_color='#797b8c', linewidths=0,
                node_size=800, width=6, **kwds)
            nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, node_color='white', linewidths=0,
                    node_size=350)
            nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, node_color='#797b8c', linewidths=0,
                    node_size=100)
            plt.tight_layout()

    else:
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        #nodes_cm = cubehelix_palette(40, as_cmap=True)
        #TODO: fix cubehelix palette parameters!!
        #nodes_cm = cubehelix_palette(40, start=.5, rot=-.75,
        #        as_cmap=True)
        nodes_cm = cubehelix_palette(60, start=0.5, rot=-0.75,
                gamma=1, hue=1, reverse=False, light=0.85, dark=0.15, as_cmap=True)

        if type(centrality) is dict and len(centrality) == len(M.nodes()):
            centralities = centrality

        elif type(centrality) in (list, np.ndarray) and len(centrality) == len(M.nodes()):
            #we need to turn the list or array into a dictionary
            centralities = {}
            for n, c in zip(M.nodes(), centrality):
                centralities[n] = c

        elif centrality == 'betweenness':
            centralities = M.betweenness_centrality

        elif centrality == 'closeness':
            centralities = M.closeness_centrality

        elif centrality == 'degree':
            centralities = M.degree_centrality

        elif centrality == 'communicability_betweenness':
            centralities = M.communicability_betweenness_centrality

        elif centrality == 'current_flow':
            centralities = M.current_flow_betweenness_centrality

        elif centrality == 'fiedler_vector':
            centralities = M.fiedler_vector_centrality

        elif centrality == 'fiedler_vector_partition':
            centralities = M.fiedler_vector_partition

        elif centrality == 'normalized_fiedler_vector':
            centralities = M.normalized_fiedler_vector_centrality

        elif centrality == 'normalized_fiedler_vector_partition':
            centralities = M.normalized_fiedler_vector_partition

        elif centrality == 'spectral_ordering':
            centralities = M.spectral_ordering_centrality
        
        if not nvmin:
            nvmin = min(list(centralities.values()))

        if not nvmax:
            nvmax = max(list(centralities.values()))

        n_colors = []
        for n in M.nodes():
            n_colors.append(centralities[n])

        if isinstance(M, SquareMaze):
            nx.draw_networkx_nodes(M, pos=M.pos, ax=ax,
                    node_color=n_colors, cmap=nodes_cm, vmin=nvmin,
                    vmax=nvmax, linewidths=0, node_size=800,
                    **kwds)

            if not simple_nodes:
                nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, node_color='white', linewidths=0,
                        node_size=350)
                nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, node_color=n_colors,
                        cmap=nodes_cm, vmin=nvmin, vmax=nvmax, linewidths=0,
                        node_size=100)

            if edge_centrality == True and str(centrality) == 'betweenness':
                edge_centralities = M.edge_betweenness_centrality
                draw_edges_with_centrality = True

            elif edge_centrality == True and str(centrality) == 'current_flow':
                edge_centralities = M.edge_current_flow_betweenness_centrality
                draw_edges_with_centrality = True

            elif type(edge_centrality) is dict:
                edge_centralities = edge_centrality
                draw_edges_with_centrality = True

            else:
                draw_edges_with_centrality = False

            if draw_edges_with_centrality:
                if not evmin:
                    evmin = min(list(edge_centralities.values()))

                if not evmax:
                    evmax = max(list(edge_centralities.values()))

                edges_cm = cubehelix_palette(60, start=0.5, rot=-0.75,
                        gamma=1, hue=1, reverse=False, light=0.85, dark=0.15, as_cmap=True)
                #edges_cm = cubehelix_palette(40, start=.5, rot=-.75, as_cmap=True)
                #edges_cm = cubehelix_palette(60, start=.5, rot=-.75,
                #        as_cmap=True)

                e_colors = []
                for e in M.edges():
                    e_colors.append(edge_centralities[e])

                nx.draw_networkx_edges(M, pos=M.pos, ax=ax,
                        edge_color=e_colors, edge_cmap=edges_cm, edge_vmin=evmin,
                        edge_vmax=evmax, width=6.0)
            
            else:
                nx.draw_networkx_edges(M, pos=M.pos, ax=ax,
                        width=6.0)#, edge_color='#797b8c')


            # Create the colorbar for the nodes' centrality
            # 1st: create the axes to be used for the colorbars
            ax = plt.gca()
            ax.set_xlim((-0.5, M.maze_size-0.5))
            ax.set_ylim((-0.5, M.maze_size-0.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines.clear()
            if colorbars:
                divider = make_axes_locatable(ax)
                n_cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
                
                # Create the colorbar for the nodes' centrality
                n_norm = mpl.colors.Normalize(vmin=nvmin, vmax=nvmax)
                n_ticks = [round(nvmin, 3), round(nvmax, 3)]
                n_cbar = mpl.colorbar.ColorbarBase(n_cbar_ax, cmap=nodes_cm,
                        norm=n_norm, ticks=n_ticks)
                if type(centrality) is str:
                    n_cbar.set_label('Node {} centrality'.format(centrality))
                else:
                    n_cbar.set_label('Node centrality')

                if draw_edges_with_centrality:
                    # 1st: create the axes to be used for the colorbars
                    e_cbar_ax = divider.append_axes('bottom', size='5%',
                            pad=0.05)

                    # Create the colorbar for the edges' centrality
                    e_norm = mpl.colors.Normalize(vmin=evmin,
                            vmax=evmax)
                    e_ticks = [round(evmin, 3), round(evmax, 3)]
                    e_cbar = mpl.colorbar.ColorbarBase(e_cbar_ax,
                            cmap=edges_cm, norm=e_norm, ticks=e_ticks,
                            orientation='horizontal')
                    e_cbar.set_label('Edge centrality')

            plt.tight_layout()

        else:
            raise DrawException

def draw_step(S, goal_only=False, ax=None, **kwds):
    """Represent a maze configuration.
    
    Parameters:
    -----------
    M : maze
       A maze object
    
    **kwds : dictionary
       Additional parameters to be passed onto 
       newtworkx.draw()
    """
    if not isinstance(S, Step):
        raise DrawException
    # First we obtain a list of the nodes that are neither current
    # nor goal location
    normal_nodes = list(S.trial.maze.nodes())
    # And then we remove current and goal location nodes because
    # those will be plotted differently
    if S.trial.goal != S.loc:
        normal_nodes.remove(S.trial.goal)

    draw_maze(S.trial.maze, ax=ax, nodelist=normal_nodes, **kwds)

    # Then we draw the nodes correspoding to current location
    if not goal_only:
        #  we use the heading to shape the arrow, if present
        if S.heading:
            shape = S.heading
        else:
            shape = '8'
        nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
                nodelist=[S.loc], node_color='#6161f2', node_size=350,
                linewidths=1.5, node_shape=shape)

  # ... and goal location 
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.trial.goal], node_color='#f26161', linewidths=0,
            node_size=800)
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.trial.goal], node_color='white', linewidths=0,
            node_size=350)
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.trial.goal], node_color='#f26161', linewidths=0,
            node_size=100)

def draw_trial(T, goal_only=False, ax=None, **kwds):
    """Draw a Trial. Will represent the taken path if that is available.
    """
    edges = []

    for i in range(len(T.path)-1):
        edges.append((T.path[i], T.path[i+1]))

    draw_step(T.steps[0], goal_only=goal_only, ax=ax, **kwds)

    if not goal_only:
        nx.draw_networkx_edges(T.maze, pos=T.maze.pos,
                edgelist=edges, edge_color='#61f261', width=3, style='dashed')

def draw(M, ax=None, **kwds):
    """Meta function. Calls the write draw function.
    """
    if isinstance(M, SquareMaze):
        draw_maze(M, ax=ax, **kwds)
    
    elif isinstance(M, Step):
        draw_step(M, ax=ax, **kwds)
    
    elif isinstance(M, Trial):
        draw_trial(M, ax=ax, **kwds)
    else:
        raise DrawException
