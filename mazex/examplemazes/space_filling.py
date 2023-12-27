import mazex as mx

peano_str = """
._. ._. ._.
| |_| |_| |
| ._. ._. |
|_| | | |_|
._. | | ._.
| |_| |_| |
"""

peano = mx.mazes.maze_from_ASCII(peano_str)
peano.name = "peano"
