import numpy as np
from numpy.typing import NDArray
from algorithms.MAB import MAB, OptimizationResult
from envs.env_noise import EnvNoise
from abc import ABC, abstractmethod
from typing import List


class BAIAlgorithm(ABC):
    def __init__(self,
                 mab: MAB,
                 fast_stopping_rule: bool = True):
        """
        Initialize a best arm identification algorithm instance.

        Args:fast
            mab (MAB): MAB instance
            fast_stopping_rule (bool, optional): if True, enables a faster algorithms that is less accurate. Defauls to true.
        """
        self.mab = mab
        
        if not fast_stopping_rule:
            self.stopping_rule =  self._slow_stopping_rule
        else:
            self.stopping_rule =  self._fast_stopping_rule

    def Cexp(self, x):
            return x + 4*np.log(1+x+ np.sqrt(2*x))
        
    def _slow_stopping_rule(self,N, delta):
        return 3 * np.sum(np.log(1 + np.log(N))) + self.K * self.Cexp(np.log(1/delta) / self.K)
    
    def _fast_stopping_rule(self,N, delta):
        return np.log(1 + np.log(N.sum())) + np.log(1/delta)
    

    @property
    def K(self) -> int:
        return self.mab.K

    @property
    def THETA(self) -> NDArray[np.float64]:
        return self.mab.THETA
    
    def Zt(self, N_t: NDArray[np.float64], Delta_hat: NDArray[np.float64], a_star_hat: int, t: int) -> float:
        T = np.max((t/N_t + t/N_t[a_star_hat])[Delta_hat>0]/Delta_hat[Delta_hat>0]**2)
        Z_t = t/(2*T)
        return Z_t
    
    @abstractmethod
    def solve(self, delta: float, FAIR: bool, VERBOSE: bool, SOLVER: str, env_noise: EnvNoise) -> List:
        pass
