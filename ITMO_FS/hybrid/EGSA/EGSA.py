import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Agent import Agent


class EGSA:
    """
    Performs an Evolutionary Gravitational Search-based Feature Selection
    For more details see "An Evolutionary Gravitational Search-based Feature Selection" by Mohammad Taradeh et al.

    Parameters
    ----------
    estimator : Estimator class (by default sklearn.neighbors.KNeighborsClassifier)
        The class of estimator that will be used as evaluator.
    n_agents : int (by default 20)
        Amount of search agents.
    iterations : int (by default 200)
        Amount of iterations to be performed if end condition is never met.
    stuck_iterations : int (by default 8)
        If the current best feature subset have not changed for stuck_iterations
        genetic algorithm mutation will be involved.
    stop_iterations : int (by default 30)
        If current best feature subset have not changed for stop_iterations algorithm will terminate.
    gravitational_factor : float (by default 10)
        gravitational_factor defines how fast agents are moving towards each other.
    gravitational_factor_min : float (by default 0)
        gravitational_factor scales towards gravitational_factor_min over time.
    gravitational_factor_scale : float (by default 1)
        gravitational_factor_scale defines how fast gravitational_factor scales over time.
    transfer_function : float -> float (by default lambda v: np.abs(np.tanh(v)))
        Function that defines probability of including/dropping a feature based on agent's velocity.
        Since it is probability, the domain of function should be [0, 1].
    alpha : float (by default 0.85)
        alpha defines the trade-off between the quality of feature subset and its size.
        The bigger alpha is the more agents lean towards maximizing accuracy of the evaluated estimator.
        And the smaller alpha is the more agents tend to drop selected features.
    agent_distance_metric : np.array of int -> np.array of int -> float (by default np.linalg.norm)
        Function that will be used to calculate distance between agents.
    seed: int (by default 31)
        Seed for random generator.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from EGSA import EGSA

    >>> X, y = make_classification(n_samples=250, n_features=10, random_state=0, n_informative=2, n_redundant=0)
    >>> trX = EGSA(alpha=0.8, iterations=10).fit_transform(X, y)

    >>> X.shape, trX.shape
    ((250, 10), (250, 4))

    >>> KNeighborsClassifier().fit(X, y).score(X, y)
    0.9

    >>> KNeighborsClassifier().fit(trX, y).score(trX, y)
    0.944
    """

    def __init__(self, n_agents=20, iterations=200, stuck_iterations=8, stop_iterations=30, gravitational_factor=10,
                 gravitational_factor_min=0, gravitational_factor_scale=1,
                 transfer_function=lambda v: np.abs(np.tanh(v)), alpha=0.85, estimator=KNeighborsClassifier,
                 agent_distance_metric=np.linalg.norm, seed=31):
        self.n_agents = n_agents
        self.iterations = iterations
        self.stuck_iterations = stuck_iterations
        self.stop_iterations = stop_iterations

        self.G = gravitational_factor
        self.G_max = gravitational_factor
        self.G_min = gravitational_factor_min
        self.scale = gravitational_factor_scale

        self.alpha = alpha
        self.estimator = estimator
        self.transfer_function = transfer_function
        self.distance_metric = agent_distance_metric

        self.seed = seed

        self.gbest = None
        self.iterations_since_last_update = 0

    def fit(self, X, y):
        """
        Fit to given data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            The target values.

        Returns
        -------
        fitted : EGSA
            Fitted instance of EGSA
        """

        n = len(X[0])

        agents = [self._make_agent(n) for _ in range(self.n_agents)]
        for agent in agents:
            agent.update_fitness(X, y, self.alpha)

        self._update_gbest(agents[0])

        for t in range(self.iterations):
            agents = self._crossover(agents, X, y)

            # Evaluate the fitness of all objects
            for agent in agents:
                agent.update_fitness(X, y, self.alpha)

            if self.iterations_since_last_update == self.stop_iterations:
                return

            if self.iterations_since_last_update > 8:
                temp = self.gbest.bit_flip_mutation()
                temp.update_fitness(X, y, self.alpha)

                if temp.fitness > self.gbest.fitness:
                    agents[-1] = self.gbest
                    self._update_gbest(temp)
                if temp.fitness > agents[0].fitness:
                    agents[-1] = agents[0]
                    agents[0] = temp
                elif temp.fitness > agents[-1].fitness:
                    agents[-1] = temp

            # Update global best
            self._update_gbest(agents[0])

            # Calculate Mi
            worst_fitness = agents[-1].fitness
            diff_sum = sum([a.fitness for a in agents]) - (self.n_agents * worst_fitness)
            for agent in agents:
                agent.update_mass(worst=worst_fitness, diff_sum=diff_sum)

            # Update the gravitational factor
            self.G *= self.G_max - (self.G_max - self.G_min) * np.log10(self.scale + 10 * t / self.iterations)

            # Calculate forces, update accelerations
            for agent in agents:
                forces = [self.G * agent.mass * other.mass * (other - agent) / self.distance_metric(other - agent) ** 2
                          for other in agents if (agent.bits != other.bits).all()]
                agent.update_acceleration(forces)

            # Update velocities
            for agent in agents:
                agent.update_velocity()

            # Update positions
            for agent in agents:
                agent.update_bits()

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
                The input samples.

        Returns
        -------
        X : array-like, shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """

        if self.gbest is None:
            raise Exception('transform is invoked on the instance that has not fitted yet')
        return X[:, self.gbest.to_bool()]

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        y : array-like, shape (n_samples, )
            Target values.

        Returns
        ------
        X_new : array-like, shape (n_samples, n_selected_features)
            Transformed array.
        """

        return self.fit(X, y).transform(X)

    def _make_agent(self, size):
        """
        Make new agent of given size

        Parameters
        ----------
        size

        Returns
        -------
        agent : Agent
            New agent
        """

        self.seed += 1
        return Agent(size=size, transfer_function=self.transfer_function, estimator=self.estimator, seed=self.seed)

    def _update_gbest(self, candidate):
        """
        Attempts to update current the best feature subset

        Parameters
        ----------
        candidate : Agent
            Candidate to new gbest
        Returns
        -------
            None
        """

        self.iterations_since_last_update += 1
        if self.gbest is None or self.gbest.fitness < candidate.fitness:
            self.gbest = candidate.copy()
            self.iterations_since_last_update = 0

    def _crossover(self, agents, data, target):
        """
        Applies crossover on gbest and top half of agents to increase exploration.

        Parameters
        ----------
        agents: List[Agent]
        data: array-like, shape (n_samples, n_features)
            Input samples.
        target: array-like, shape (n_samples, )
            Target values.

        Returns
        -------
        pool: List[Agent]
            New pool of agents.
        """

        pool = sorted(agents, key=lambda a: a.fitness, reverse=True)
        for i in range(self.n_agents // 2):
            new = self.gbest.crossover(pool[i])
            new.update_fitness(data, target, self.alpha)
            pool.append(new)

        return sorted(pool, key=lambda a: a.fitness, reverse=True)[:self.n_agents]
