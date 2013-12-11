"""
MazeX
========

    MazeX is a package derived from NetworkX for the purpose of representing
    maze structures.

Using
-----

    Just write in Python

    >>> import maze as nx
    >>> G=nx.Graph()
    >>> G.add_edge(1,2)
    >>> G.add_node(42)
    >>> print(sorted(G.nodes()))
    [1, 2, 42]
    >>> print(sorted(G.edges()))
    [(1, 2)]
"""
#    Copyright (C) 2004-2010 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#
# Add platform dependent shared library path to sys.path
#


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
from networkx.exception import  *
import networkx.external
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
from networkx.tests.test import run as test

import networkx.drawing
from networkx.drawing import *

