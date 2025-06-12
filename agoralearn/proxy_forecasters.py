"""Controllers module."""

# Author: Hamza Cherkaoui

from .criterions import criterion_2


class ProxySamplesSharing:
    def __init__(self, main_agent, agents, sigma=None, L=None):
        """
        Controller to manage the sample sharing between agents.

        Parameters
        ----------
        main_agent : forecaster
            main forecaster
        agents : list of forecaster
            List of forecaster class, i.e. a class that provide the batch_fit,
            online_fit_predict, online_predict methods.
        """
        self.main_agent = main_agent
        self.agents = agents
        self.collaborators = set(range(len(agents)))

        self.sigma = sigma
        self.L = L
        self.lbda = self.main_agent.lbda
        self.states = self.main_agent.states

    def _fetch_X_y_from_collaborators(self, idx_collaborators):
        """Return the list of samples [(X_1, y_1), ..., (X_n, y_n)]
        corresponding to the given list of index select_agents.
        """
        return [self.agents[idx].fetch_X_y() for idx in idx_collaborators]

    def _fetch_x_y_from_collaborators(self, idx_collaborators):
        """Return the list of last observed samples [(x_1, y_1), ..., (x_n, y_n)]
        corresponding to the given list of index select_agents.
        """
        return [self.agents[idx].fetch_x_y() for idx in idx_collaborators]

    def _update_list_of_collab(self):
        """Update the list of collaborators."""
        # no potential collaborators available
        if len(self.collaborators) == 0:
            return self.collaborators

        collab_flag = False

        # try current collaboration setting
        l_X_y = self._fetch_X_y_from_collaborators(self.collaborators)
        collab_flag = criterion_2(l_X_y, sigma=self.sigma, L=self.L)

        if collab_flag:
            return self.collaborators

        # heuristic to find a working collaboration setting
        while not collab_flag:

            # try to remove one agent
            for i in self.collaborators:

                collaborators = self.collaborators - {i}

                # if no collaborator
                if len(collaborators) == 0:
                    self.collaborators = set([])
                    return set([])

                l_X_y = self._fetch_X_y_from_collaborators(collaborators)
                collab_flag = criterion_2(l_X_y, sigma=self.sigma, L=self.L)

                if collab_flag:
                    return collaborators

            # if failed, remove at random one agent and restart the search
            self.collaborators.pop()

            # if no collaborator
            if len(self.collaborators) == 0:
                self.collaborators = set([])
                return set([])

    def add_collaborators(self, agents):
        """Manually add new potential collaborators."""
        n = len(self.collaborators)
        self.agents.extend(agents)
        self.collaborators |= set([n + i for i in range(n)])

    def fetch_X_y(self, ):
        """Return the gathered samples (X, y)."""
        return self.main_agent.fetch_X_y()

    def fetch_x_y(self, ):
        """Return the last observed samples (x, y)."""
        return self.main_agent.fetch_x_y()

    def batch_fit(self, X, y):
        """Fit the model using batch ridge regression."""
        self._update_list_of_collab()
        l_X_y = self._fetch_X_y_from_collaborators(self.collaborators)
        self.main_agent.batch_fit(X=X, y=y, l_X_y=l_X_y)
        return self

    def online_fit_predict(self, x, y):
        """Predict and update the model with a new observation."""
        self._update_list_of_collab()
        l_x_y = self._fetch_x_y_from_collaborators(self.collaborators)
        return self.main_agent.online_fit_predict(x, y, l_x_y=l_x_y)

    def online_predict(self, x):
        """Predict the next value given the current state (without updating)."""
        return self.main_agent.online_predict(x)
