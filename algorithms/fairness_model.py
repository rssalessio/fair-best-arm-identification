import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Callable

TOL = 1e-12

class FairnessModel(ABC):
    """Defines the fairness constraints used by the algorithm"""
    def __init__(self, K: int):
        self.K = K
    
    @abstractmethod
    def make_model(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        raise Exception('Not implemented!')
    
    def value(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        model = self.make_model()
        return model(theta)

class UniformFairnessModel(FairnessModel):
    """ A random vector of pre-specified rates with each p_a in [0,1/K] """
    def __init__(self, K: int):
        super().__init__(K)

    def _model(self, theta: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return np.random.uniform(0, 1 / self.K, self.K)

    def make_model(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self._model

class LinearFairnessModel(FairnessModel):
    """ A vector of pre-specified rates linearly ranging from 0 to p_max """
    def __init__(self, K: int, p_max: float):
        super().__init__(K)
        self.p_max = p_max

    def _model(self, theta: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return np.linspace(0, self.p_max, self.K)
    
    def make_model(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self._model

class EqualFairnessModel(FairnessModel):
    """ A vector of equal pre-specified rates p """
    def __init__(self, K: int, p: float):
        super().__init__(K)
        assert p <= 1/K, 'p needs to be lower than or equal to 1/K'
        self.p = p

    def _model(self, theta: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self.p * np.ones(self.K)
    
    def make_model(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self._model
    
class VectorFairnessModel(FairnessModel):
    """ A vector of pre-specified rates p """
    def __init__(self, K: int, p: NDArray[np.float64]):
        super().__init__(K)
        self.p = p

    def _model(self, theta: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self.p
    
    def make_model(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self._model

class ProportionalFairnessModel(FairnessModel):
    """ A vector of theta-dependent rates defined through the proportional fair model
        p_a(theta) = p0 * max(0,theta_a) / \sum_b max(0,theta_b)

    Args:
         (p0, float): constant value in [0,1]
         (invert, bool): If True, make the model inversely proportional. Defaults to False.
         (use_gaps, bool): If True, uses the gap Delta to compute p(theta). Defaults to False.
    """
    def __init__(self, K: int, p0: float, invert: bool = False, use_gaps: bool = False):
        super().__init__(K)
        self.p0 = p0
        self.invert = invert
        self.use_gaps = use_gaps
    
    @staticmethod
    def _prop_fair(theta: NDArray[np.float64], p0: float, invert: bool, use_gaps: bool) -> NDArray[np.float64]:
        x = np.max(theta) - theta if use_gaps else theta
        delta_min_inv= 1/x[x>0].min()
        if invert:
            x = np.nan_to_num(1/x, nan=delta_min_inv, posinf=delta_min_inv,neginf=delta_min_inv)
        return p0 * x / np.sum(x)

    def _model(self, theta: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return ProportionalFairnessModel._prop_fair(theta, self.p0, self.invert, self.use_gaps)
    
    def make_model(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self._model
    
class SoftmaxFairnessModel(FairnessModel):
    """ A vector of theta-dependent rates defined through the softmax model 
        p_a(theta) = p0 * exp(-theta_a / temperature)/\sum_b exp(-theta_b / temperature)

    Args:
         (p0, float): constant value in [0,1]
         (temperature, float): temperature value. Defaults to one.
    """
    def __init__(self, K: int, p0: float, temperature: float = 1,  invert: bool = False, use_gaps: bool = False):
        super().__init__(K)
        assert p0 >= 0 and p0 <= 1, "c needs to be in [0,1]"
        self.p0 = p0
        self.temperature = temperature
        self.invert = invert
        self.use_gaps = use_gaps

    @staticmethod
    def _softmax(theta: NDArray[np.float64], p0: float,
                 temperature: float, invert: bool, use_gaps: bool) -> NDArray[np.float64]:
        x = np.max(theta) - theta if use_gaps else theta
        delta_min_inv= 1/x[x>0].min()
        if invert:
            x = np.nan_to_num(1/x, nan=delta_min_inv, posinf=delta_min_inv,neginf=delta_min_inv)
        x = np.exp(x / temperature)
        return p0 * x / np.sum(x)
    
    def _model(self, theta: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return SoftmaxFairnessModel._softmax(theta, self.p0, self.temperature, self.invert, self.use_gaps)
    
    def make_model(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self._model

class CustomFairnessModel(FairnessModel):
    """ Custom fairness model """
    def __init__(self, K: int, model: Callable[[NDArray[np.float64]], NDArray[np.float64]]):
        super().__init__(K)
        self.model = model
    
    def make_model(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return self.model

def make_prespecified_fairness_model(w: NDArray[np.float64], p0: float, agonistic: bool) -> VectorFairnessModel:
    """Make a prespecified fairness model using the uncostrained solution w*

    Args:
        w (NDArray[np.float64]): Unconstrained solution w*
        p0 (float): Scale fairness value by p0. Needs to be in [0,1]
        agonistic (bool): If agonistic, put more weight on w_{a*} as (1+w_a*)/2. The remaining weight is split across arms.
        
                          If antagonistic, the weight is set according to (w* + 1/K)/2

    Returns:
        VectorFairnessModel: Return a fairness model
    """
    K = len(w)
    if agonistic:
        factor = 0.9   
    else:
        factor = 0.1
    winv = 1/(1e-12 + w)
    winv = winv / winv.sum()
    p = (factor * w + (1-factor) * winv)
    p = p0 * p / p.sum()
    return VectorFairnessModel(K, p)