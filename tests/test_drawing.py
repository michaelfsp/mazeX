import sys
# so we can import mazex.
sys.path.append('../../')
import mazex as mx
import random
import pylab as pl

class TestMaze:
    def __init__(self):
        self.maze = mx.RandomButConnectedSquareMaze(6, 37)
        self.hagent = mx.HeuristicAgent((0,0), beta=float('inf'), kappa=1, xi=0.3, heading = '^')

        self.goals = self.maze.nodes() * 3
        random.shuffle(self.goals)

        self.htrials = []
        for goal in self.goals:
            self.hagent.edge_visit_cache.clear()
            step = mx.Step(self.maze, self.hagent.loc, goal)
            htrial = mx.Trial([step], self.hagent)
            htrial.solve()
            self.htrials.append(htrial)

        self.hsession = mx.Session(self.hagent, self.htrials)

    def test_draw_maze(self):
        mx.draw(self.maze)
        pl.close()

    def test_draw_path(self):
        mx.draw(self.hsession.trials[0])
        pl.close()

    def test_draw_step(self):
        mx.draw(self.hsession.trials[0].steps[0])
        pl.close()
