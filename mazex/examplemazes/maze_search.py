import mazex

maze1_str = '''
._._._._._.
| . |_._|_|
|_| | |_|_.
|_| ._. ._.
|_| . | |_|
|_._|_|_|_|
'''
maze1 = mazex.mazes.maze_from_ASCII(maze1_str)

maze2_str = '''
._._._. ._.
| |_|_._._|
| | ._. |_|
|_. ._| |_|
| . |_| |_|
|_|_| |_._|
'''
maze2 = mazex.mazes.maze_from_ASCII(maze2_str)

maze3_str = '''
._._._._._.
|_|_|_| |_|
|_| ._| ._|
| | | ._|_|
|_. |_|_._|
|_|_|_|_|_.
'''
maze3 = mazex.mazes.maze_from_ASCII(maze3_str)
