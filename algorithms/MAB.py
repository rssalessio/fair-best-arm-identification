import numpy as np
import cvxpy as cp
import time
from numpy.typing import NDArray
from typing import NamedTuple
from algorithms.reward_model import RewardModel
from algorithms.fairness_model import FairnessModel, VectorFairnessModel

class OptimizationResult(NamedTuple):
    wstar: NDArray[np.float64]
    sol_value: float
    computation_time: float

class MAB(object):
    Delta: NDArray[np.float64] # Vector of suboptimality gaps
    THETA: NDArray[np.float64] # Vector of average rewards
    ACTIONS: NDArray[np.int64] # Arms indexes
    a_star: int                # Best arm index

    def __init__(self,
                 reward_model: RewardModel,
                 fairness_model: FairnessModel| None = None):
        """
        Initialize a MAB instance.

        Args:
            reward_model (RewardModel): Specifies the reward distribution of the arms.
            fairness_model (FairnessModel, Optional): fairness function. Needs to be continuous and vanishing in 0. Defaults to None.
        """
    
        self.reward_model = reward_model
        self.ACTIONS = np.arange(self.K)
        self.f = fairness_model.make_model() if fairness_model is not None else VectorFairnessModel(self.K, np.zeros(self.K)).make_model()
        self.THETA = reward_model.make_model()
        
        assert np.all(self.f(self.THETA) >= 0), 'f(theta) must return a non-negative vector'
            
        self.a_star: int = np.argmax(self.THETA)
        self.Delta: NDArray[np.float64] = self.THETA[self.a_star] - self.THETA

    

    @property
    def K(self) -> int:
        return self.reward_model.K

    @property
    def R_MAX(self) -> float:
        return self.reward_model.R_MAX
    
    @property
    def DELTA_MAX(self) -> float:
        return self.Delta.max()

    @property
    def DELTA_MIN(self) -> float:
        return np.sort(self.Delta)[1]
        
    def solve_T_star(self,
                    SOLVER: str = cp.MOSEK,
                    FAIR: bool = False,
                    VERBOSE: bool = False,
                    a_star: int | None = None,
                    THETA: NDArray[np.float64] | None = None,
                    TOL: float = 1e-12) -> OptimizationResult:
        """ Solve T^\star

        Args:
            SOLVER (str, optional): Type of solver used by CVXPY. Defaults to "MOSEK".
            FAIR (bool, optional): Boolean variable specifying wether to account for the fairness constraint. Defaults to False.
            VERBOSE (bool, optional): Enable verbosity of the solver. Defaults to False.
            a_star (int | None, optional): Best arm index. Defaults to None.
            THETA (NDArray[np.float64] | None, optional): Vector of average rewards. Defaults to None.
            TOL (float, optional): Tolerance used by the solver. Defaults to 1e-12.

        Returns:
            result (OptimizationResult): result containing w*, the optimization value and the computation time
        """
        if a_star is None:
            a_star = self.a_star
        if THETA is None:
            THETA = self.THETA

        assert len(THETA) == self.K, 'THETA should be of size K'

        # Compute the gaps squared
        Delta: NDArray[np.float64] = self.Delta if THETA is None else THETA[a_star] - THETA
        Delta_squared = Delta ** 2
        a_sub = self.ACTIONS[self.ACTIONS != a_star]

        if FAIR and np.isclose(self.f(THETA).sum(), 1.):
            w = self.f(THETA)
            mask = Delta > 0
            sol = np.max(1/(Delta_squared[a_sub] * w[a_sub]) + 1 / (Delta_squared[mask].min() * w[a_star]))
            return OptimizationResult(w, sol, 0.)

        # CVXPY variables
        w = cp.Variable(self.K, pos = True) 
        z = cp.Variable(pos = True)
        
        constraints = [cp.sum(w) == 1, w >= TOL]

        if FAIR:
            constraints.append(w >= self.f(THETA))
    
        for a in a_sub:
            # Numerator
            Na = cp.inv_pos(w[a]) + cp.inv_pos(w[a_star])
            constraints.append(Delta_squared[a] * z >= Na)
        
        obj = cp.Minimize(z)
        prob = cp.Problem(obj, constraints)
        start = time.time()
        sol = prob.solve(verbose = VERBOSE, solver = SOLVER)
        end = time.time()
        computation_time = end - start

        if sol is None or w.value is None:
            raise Exception('The optimization problem has failed.')

        return OptimizationResult(w.value / np.sum(w.value), sol, computation_time)

    def solve_C_star(self,
                        SOLVER: str = cp.MOSEK,
                        FAIR: bool = False,
                        VERBOSE: bool = False,
                        a_star: int | None = None,
                        THETA: NDArray[np.float64] | None = None,
                        TOL: float = 1e-12) -> OptimizationResult:
            """ Solve C^\star

            Args:
                SOLVER (str, optional): Type of solver used by CVXPY. Defaults to "MOSEK".
                FAIR (bool, optional): Boolean variable specifying wether to account for the fairness constraint. Defaults to False.
                VERBOSE (bool, optional): Enable verbosity of the solver. Defaults to False.
                a_star (int | None, optional): Best arm index. Defaults to None.
                THETA (NDArray[np.float64] | None, optional): Vector of average rewards. Defaults to None.
                TOL (float, optional): Tolerance used by the solver. Defaults to 1e-12.

            Returns:
                _type_: _description_
            """
            if a_star is None:
                a_star = self.a_star
            if THETA is None:
                THETA = self.THETA
            # Compute the gaps 
            Delta: NDArray[np.float64] = self.Delta if THETA is None else THETA[a_star] - THETA
            
            # CVXPY variables
            v = cp.Variable(self.K, pos = True) 
            a_sub = self.ACTIONS[self.ACTIONS != a_star]
            constraints = [v >= TOL]

            if FAIR:
                constraints.append(v >= self.f(THETA))
        
            for a in a_sub:
                constraints.append(v[a] >= 1/self.Delta[a]**2)
            
            obj = cp.Minimize(v@self.Delta)
            prob = cp.Problem(obj, constraints)
            start = time.time()
            sol = prob.solve(verbose = VERBOSE, solver = SOLVER)
            end = time.time()
            computation_time = end - start

            if sol is None or v.value is None:
                raise Exception('The optimization problem has failed.')

            return OptimizationResult(v.value, sol, computation_time)
        
    def solve_C_star_closed_form(self,
                    FAIR: bool = False,
                    a_star: int | None = None,
                    THETA: NDArray[np.float64] | None = None) -> OptimizationResult:
        """ Solve C^\star

        Args:
            FAIR (bool, optional): Boolean variable specifying wether to account for the fairness constraint. Defaults to False.
            a_star (int | None, optional): Best arm index. Defaults to None.
            THETA (NDArray[np.float64] | None, optional): Vector of average rewards. Defaults to None.
        Returns:
            _type_: _description_
        """
        if a_star is None:
            a_star = self.a_star
        if THETA is None:
            THETA = self.THETA
        # Compute the gaps
        Delta: NDArray[np.float64] = self.Delta if THETA is None else THETA[a_star] - THETA
        
        # CVXPY variables
        v = np.zeros(self.K)
        v[Delta>0] = 1/Delta[Delta>0]**2
        sol = v@Delta
        
        if not FAIR:
            return OptimizationResult(v, sol, 0)
        else:
            v_fair = np.where(v < self.f(THETA),self.f(THETA), v)
            v[Delta>0] = 0
            sol_fair = v_fair@Delta
            return OptimizationResult(v_fair, sol_fair, 0)