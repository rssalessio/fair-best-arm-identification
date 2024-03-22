import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class RewardModel(ABC):
    """Abstract reward model class. Defines the average reward of each arm.
    """
    def __init__(self, R_MAX: float, K: int):
        assert K > 0, "Number of arms needs to be strictly positive"
        self.R_MAX = R_MAX
        self.K = K
    
    @abstractmethod
    def make_model(self) -> NDArray[np.float64]:
        raise Exception('Not implemented!')

class UniformRewardModel(RewardModel):
    """ K random rewards from 0 to R_MAX """
    def __init__(self, R_MAX: float, K: int):
        super().__init__(R_MAX, K)

    def make_model(self) -> NDArray[np.float64]:
        return np.random.uniform(0, self.R_MAX, self.K)

class LinearRewardModel(RewardModel):
    """ Linear range of rewards from 0 to R_MAX (with K arms) """
    def __init__(self, R_MAX: float, K: int):
        super().__init__(R_MAX, K)

    def make_model(self) -> NDArray[np.float64]:
        return np.linspace(0, self.R_MAX, self.K)

class EqualGapRewardModel(RewardModel):
    """ Model where all the arms have the same gap R_MAX"""
    def __init__(self, R_MAX: float, K: int):
        super().__init__(R_MAX, K)

    def make_model(self) -> NDArray[np.float64]:
        THETA = np.zeros(self.K)
        THETA[-1] = self.R_MAX
        return THETA

class CustomRewardModel(RewardModel):
    """ Define a custom vector of rewards """
    def __init__(self, theta: NDArray[np.float64]):
        super().__init__(np.max(theta), len(theta))
        self.theta = theta

    def make_model(self) -> NDArray[np.float64]:
        return self.theta