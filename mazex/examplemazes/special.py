import mazex as mx

snake_str = """
._._._._._.
._._._._._|
|_._._._._.
._._._._._|
|_._._._._.
._._._._._|
"""

snake = mx.mazes.maze_from_ASCII(snake_str)
snake.name = "snake"

rooms_str = """
._._. ._._.
|_|_|_|_|_|
|_|_| |_|_|
._|_. ._|_.
|_|_|_|_|_|
|_|_| |_|_|
"""

rooms = mx.mazes.maze_from_ASCII(rooms_str)
rooms.name = "rooms"


rooms_shuffle_str = """
._._. ._._.
|_|_| |_|_|
|_|_|_|_|_|
|_._. ._|_.
|_|_|-|_|_|
|_|_| |_|_|
"""

rooms_shuffle = mx.mazes.maze_from_ASCII(rooms_shuffle_str)
rooms_shuffle.name = "rooms_shuffle"
