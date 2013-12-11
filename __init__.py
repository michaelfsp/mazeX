"""
MazeX
========

    MazeX is a package derived from NetworkX for the purpose of representing
    maze structures.

Using
-----

    Just write in Python

    >>> import maze as mx
    >>> S=mx.SquareMaze(5)
	#TODO: correct the following lines
	>>> G.add_edge(1,2)
    >>> G.add_node(42)

"""

import sys
if sys.version_info[:2] < (3, 0) < (3.3):
    m = "Python version 3.0 or later is required for MazeX (%d.%d detected).
    Version 3.3 or later is recommended!"
    raise ImportError(m % sys.version_info[:2])

elif sys.version_info[:2] >= (3, 0) < (3.3):
	m = "Python version 3.3 or later is required for MazeX (%d.%d detected).
    Proceed with care!"
    raise ImportWarning(m % sys.version_info[:2])

del sys

# Release data
from networkx import release

__author__   = "Michael Pereira"

__version__ = "0.0.1_git"

#These are import orderwise
from mazex.exception import  *

from mazex.classes import *

