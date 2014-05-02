import numpy as np

class Likelihood(object):
    """A class for the likelihood functions."""

    def __init__(self, session, agent_class):
        self.session = session
        self.agent_class = agent_class

    def neg_log_likelihood(self, agent_params):
        """Compute the likelihood of the data in the session gien the
        parameter set specified in the init of this likelihood method.
        """
        first_loc = self.session.trials[0].steps[0].loc
        self.agent = self.agent_class(first_loc, *agent_params)

        log_likelihood = 0
        for t in self.session.trials:
            for s in t.steps[:-1]:
                values = self.agent.evaluate(s.available_nodes, t.goal)
                choice_prob = self.agent.choice_probs(values)[s.choice]
                self.agent.loc = s.subsequent.loc

                log_likelihood += np.log(choice_prob)

        return -log_likelihood

