import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class EnvNoise(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(action: int) -> float:
        raise NotImplementedError('Sample function not implemented!')

class GaussianEnvNoise(EnvNoise):
        def __init__(self):
            super().__init__()
        
        def sample(self, theta: NDArray[np.float64], action: int) -> float:
            return np.random.normal(theta[action], 1)


class BernoulliEnvNoise(EnvNoise):
        def __init__(self):
            super().__init__()
        
        def sample(self, theta: NDArray[np.float64], action: int) -> float:
            return np.random.uniform() < theta[action]

class GumbelEnvNoise(EnvNoise):
        def __init__(self):
            super().__init__()
        
        def sample(self, theta: NDArray[np.float64], action: int) -> float:
            return np.random.gumbel(loc=theta[action], scale=1.0)
