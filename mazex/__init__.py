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
__author__   = "Michael Pereira"

__version__ = "dev_git"

import sys
if sys.version_info[:2] < (3, 0):
    m = """Python version 3.0 or later is required for MazeX (%d.%d detected).
    Version 3.3 or later is recommended!"""
    raise ImportError(m % sys.version_info[:2])

elif (3, 0) <= sys.version_info[:2] < (3, 3):
    m = """Python version 3.3 or later is required for MazeX (%d.%d detected).
    Proceed with care!"""
    raise ImportWarning(m % sys.version_info[:2])

del sys

"""
These imports are here for now only for debugging purposes.
"""

#These are import orderwise
from networkx.exception import  *
import networkx.utils
# these packages work with Python >= 2.6

import networkx.classes
from networkx.classes import *


import networkx.convert
from networkx.convert import *

import networkx.relabel
from networkx.relabel import *

import networkx.generators
from networkx.generators import *

import networkx.readwrite
from networkx.readwrite import *

#Need to test with SciPy, when available
import networkx.algorithms
from networkx.algorithms import *
import networkx.linalg

from networkx.linalg import *

import networkx.drawing
from networkx.drawing import *

"""
And here we have the imports we really need.
"""
# Agents
from mazex.agents import *

# Mazes
from mazex.mazes import *

# Drawing capabilities
from mazex.drawing import *

# Datamodel related stuff
from mazex.datamodel import *

# Simmulation stuff
from mazex.simtask import *

import mazex.analysis as analysis
import mazex.data as data
import mazex.fitting as fitting
import mazex.search as search
import mazex.examplemazes as examplemazes
"""
I believe we don't need these imports.


# MazeX exceptions
from mazex.exceptions import *

"""
