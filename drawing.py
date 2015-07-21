"""
Visually represent mazes and trials
"""
import numpy as np
import pylab as pl
import networkx as nx
from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.datamodel import Step, Trial
from mazex.analysis.distances import distances_plot

# other interesting node colors '#848599'
def get_cubehelix_RGB(x, nsteps=10, vmin=0, vmax=1, scheme='green',
        hexadecimal=False):
    """Maps a scalar to an RGB tuple from the seaborn version of cubehelix.
    """
    if scheme == 'green':
        colors = sns.cubehelix_palette(nsteps, start=.5, rot=-.75)
    elif scheme == 'pink':
        colors = sns.cubehelix_palette(nsteps)

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

def draw_maze(M, centrality=None, ax=None, hold=None, **kwds):
    """Draw a maze configuration.
    """
    if not isinstance(M, SquareMaze):
        raise DrawException
    if not ax:
        ax = pl.gca()
    # Makes sure the drawings are square and not rectangular
    ax.set_aspect('equal')

    if not centrality:
        if isinstance(M, SquareMaze):
            nx.draw(M, pos=M.pos, ax=ax, hold=True, node_color='#797b8c', linewidths=0,
                node_size=800, width=6, **kwds)
            nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, hold=hold, node_color='white', linewidths=0,
                    node_size=350)
            nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, hold=hold, node_color='#797b8c', linewidths=0,
                    node_size=100)
            pl.tight_layout()

    else:
        import matplotlib as mpl
        import seaborn.apionly as sns
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        nodes_cm = sns.cubehelix_palette(40, as_cmap=True)
        edges_cm = sns.cubehelix_palette(40, start=.5, rot=-.75, as_cmap=True)

        if centrality == 'betweenness':
            centralities = nx.centrality.betweenness_centrality(M)

        elif centrality == 'closeness':
            centralities = nx.centrality.closeness_centrality(M)
    
        vmin_nodes = min(list(centralities.values()))
        vmax_nodes = max(list(centralities.values()))
        n_colors = []
        for n in M.nodes():
            n_colors.append(centralities[n])

        #nodes_cmap = cubehelix.cmap(reverse=True, start=0.5, rot=-0.75,
        #vmin=vmin_nodes, vmax=vmax_nodes)

        if isinstance(M, SquareMaze):
            edge_centralities = nx.centrality.edge_betweenness_centrality(M)
            vmin_edges = min(list(edge_centralities.values()))
            vmax_edges = max(list(edge_centralities.values()))
            #edges_cmap = cubehelix.cmap(reverse=True, start=0, rot=0.4,
            #vmin=vmin_edges, vmax=vmax_edges)
            e_colors = []
            for e in M.edges():
                e_colors.append(edge_centralities[e])

            nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, hold=True,
                    node_color=n_colors,
                    cmap=nodes_cm, vmin=vmin_nodes,
                    vmax=vmax_nodes,linewidths=0, node_size=800, width=6,
                    **kwds)
            nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, hold=hold, node_color='white', linewidths=0,
                    node_size=350)
            nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, hold=hold, node_color=n_colors,
                    cmap=nodes_cm, vmin=vmin_nodes, vmax=vmax_nodes, linewidths=0,
                    node_size=100)
            
            nx.draw_networkx_edges(M, pos=M.pos, ax=ax, hold=hold,
                    edge_color=e_colors, edge_cmap=edges_cm, vmin=vmin_edges,
                    vmax=vmax_edges, width=6.0)

            # Create the colorbar for the nodes' centrality
            # 1st: create the axes to be used for the colorbars
            ax = pl.gca()
            print(ax.get_xlim())
            ax.set_xlim((-0.5, M.size-0.5))
            ax.set_ylim((-0.5, M.size-0.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines.clear()
            divider = make_axes_locatable(ax)
            n_cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
            e_cbar_ax = divider.append_axes('bottom', size='5%', pad=0.05)

            # Create the colorbar for the nodes' centrality
            n_norm = mpl.colors.Normalize(vmin=vmin_nodes, vmax=vmax_nodes)
            n_ticks = [round(vmin_nodes, 3), round(vmax_nodes, 3)]
            n_cbar = mpl.colorbar.ColorbarBase(n_cbar_ax, cmap=nodes_cm,
                    norm=n_norm, ticks=n_ticks)
            n_cbar.set_label('Node %s centrality' % (centrality))

            # Create the colorbar for the edges' centrality
            e_norm = mpl.colors.Normalize(vmin=vmin_edges, vmax=vmax_edges)
            e_ticks = [round(vmin_edges, 3), round(vmax_edges, 3)]
            e_cbar = mpl.colorbar.ColorbarBase(e_cbar_ax, cmap=edges_cm,
                    norm=e_norm, ticks=e_ticks, orientation='horizontal')
            e_cbar.set_label('Edge betweenness centrality')

            pl.tight_layout()

def draw_step(S, goal_only=False, ax=None, hold=None, **kwds):
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
    normal_nodes = S.trial.maze.nodes()
    # And then we remove current and goal location nodes because
    # those will be plotted differently
    if S.trial.goal != S.loc:
        normal_nodes.remove(S.trial.goal)

    draw_maze(S.trial.maze, ax=ax, hold=True, nodelist=normal_nodes, **kwds)

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
            node_size=800, hold=True)
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.trial.goal], node_color='white', linewidths=0,
            node_size=350, hold=hold)
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.trial.goal], node_color='#f26161', linewidths=0,
            node_size=100, hold=hold)

def draw_trial(T, goal_only=False, ax=None, hold=None, **kwds):
    """Draw a Trial. Will represent the taken path if that is available.
    """
    edges = []

    for i in range(len(T.path)-1):
        edges.append((T.path[i], T.path[i+1]))

    draw_step(T.steps[0], goal_only=goal_only, ax=ax, hold=True, **kwds)

    if not goal_only:
        nx.draw_networkx_edges(T.maze, pos=T.maze.pos,
                edgelist=edges, edge_color='#61f261', width=3, style='dashed')

def draw(M, ax=None, hold=None, **kwds):
    """Meta function. Calls the write draw function.
    """
    if isinstance(M, SquareMaze):
        draw_maze(M, ax=ax, hold=hold, **kwds)
    
    elif isinstance(M, Step):
        draw_step(M, ax=ax, hold=hold, **kwds)
    
    elif isinstance(M, Trial):
        draw_trial(M, ax=ax, hold=hold, **kwds)
    else:
        raise DrawException
