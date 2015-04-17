from collections import Iterator

class FirstExitSteps(Iterator):
    def __init__(self, session):
        self.session = session
        self.i_last_trial = len(self.session.trials) -1
        self.i_trial = 0
        # we also discount the last step because agent is already at goal
        self.i_last_step = len(self.session.trials[self.i_trial].steps) -2
        self.i_step = 0
        self.visited_nodes = []


    def __next__(self):
        if self.i_trial < self.i_last_trial:
            step = self.session.trials[self.i_trial].steps[self.i_step]

            if self.i_step < self.i_last_step:
                if step.loc not in self.visited_nodes:
                    self.visited_nodes.append(step.loc)
                    self.i_step += 1
                    return step

            elif self.i_step == self.i_last_step:
                self.i_trial += 1
                self.i_step = 0
                self.i_last_step = len(self.session.trials[self.i_trial].steps) -2

                if step.loc not in self.visited_nodes:
                    # we don't need to add it to the list of visited nodes
                    #  because this is the last step already
                    self.visited_nodes.clear()
                    return step
                else:
                    self.visited_nodes.clear()

        elif self.i_trial == self.i_last_trial:
            if self.i_step == self.i_last_step:
                import ipdb; ipdb.set_trace()
                raise StopIteration


