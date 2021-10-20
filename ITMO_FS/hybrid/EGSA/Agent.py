import numpy as np


class Agent:
    """
    Agent for Evolutionary Gravitational Search-based Feature Selection.
    Simply a bit vector that stands for selected features with it's fitness, acceleration, velocity and mass.

    Parameters
    ----------
    size: int
        Size of bit vector for selected features.
    estimator : Estimator class
        The class of estimator that will be used as evaluator.
    transfer_function : float -> float
        Function that defines probability of including/dropping a feature based on agent's velocity.
        Since it is probability, the domain of function should be [0, 1].
    seed: int
        Seed for random generator.
    """
    def __init__(self, size, estimator, transfer_function, seed):
        self.estimator = estimator
        self.transfer_function = transfer_function
        self.size = size

        self.seed = seed
        self.random = np.random.default_rng(seed)
        self.bits = self.random.integers(0, 2, size)

        self.fitness = None
        self.mass = None
        self.acceleration = None
        self.velocity = np.zeros(size, dtype=np.float64)

    def count_selected(self):
        return self.bits.sum()

    def update_fitness(self, data, target, alpha):
        selected = data[:, self.to_bool()]
        fitness = self.estimator().fit(selected, target).score(selected, target)
        self.fitness = alpha * fitness + (1 - alpha) * (1 - (self.count_selected() / self.size))

    def update_mass(self, worst, diff_sum):
        diff_sum = diff_sum if diff_sum != 0 else 0.0001
        self.mass = (self.fitness - worst) / diff_sum

    def update_acceleration(self, forces):
        self.acceleration = sum([self.random.random() * f for f in forces])
        if self.mass != 0:
            self.acceleration /= self.mass
        else:
            self.acceleration /= 0.001

    def update_velocity(self):
        self.velocity *= self.random.random()
        self.velocity += self.acceleration

    def update_bits(self):
        for k, v in enumerate(self.velocity):
            if self.transfer_function(v) > self.random.random():
                self.bits[k] = 1 if self.bits[k] == 0 else 0

        if self.count_selected() == 0:
            self.bits[0] = 1

    def copy(self):
        new = Agent(self.size, transfer_function=self.transfer_function, estimator=self.estimator,
                    seed=self.seed + self.random.integers(1000))
        new.bits = self.bits.copy()
        new.fitness = self.fitness
        return new

    def to_bool(self):
        if self.bits.sum() == 0:
            self.bits = self.random.integers(0, 2, self.size)
        return [bool(x) for x in self.bits]

    def __sub__(self, other):
        new = self.copy()
        new.bits = np.array([1 if x != 0 else 0 for x in (self.bits - other.bits)])
        return new.bits

    def __str__(self):
        return ' '.join([str(self.bits), str(self.fitness)])

    def swap_mutation(self):
        i, j = self.random.integers(2, 2)
        new = self.copy()
        new.bits[i], new.bits[j] = new.bits[j], new.bits[i]
        return new

    def bit_flip_mutation(self):
        i = self.random.integers(self.size)
        new = self.copy()
        new.bits[i] = (new.bits[i] + 1) % 2
        return new

    def crossover(self, other):
        cross_len = self.size // 5
        i = self.random.integers(self.size)
        new = self.copy()
        for j in range(cross_len):
            k = (i + j) % self.size
            new.bits[k] = other.bits[k]
        return new
