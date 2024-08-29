import numpy as np
import cvxpy as cp
from numpy.typing import NDArray
from algorithms.MAB import MAB
from algorithms.bai.bai_algorithm import BAIAlgorithm
from envs.env_noise import EnvNoise, GaussianEnvNoise
from typing import Callable

class FTaS(BAIAlgorithm):
    def __init__(self, mab: MAB, fast_stopping_rule: bool = True):
        """
        Initialize FairBAI.

        Args:
            mab (MAB): MAB instance.
            fast_stopping_rule (bool, optional): if True, enables a faster algorithms that is less accurate. Defauls to true.
        """
        super().__init__(mab, fast_stopping_rule)

    def solve(self, delta, FAIR= True, VERBOSE = False,
              SOLVER = cp.MOSEK, env_noise: EnvNoise = GaussianEnvNoise(), pre_specified_rate: bool = False, alpha: float = 0.5):
        """ 
        Run FairBAI algorithm
        """
        # Variables and counters initialization
        time_vec, sol_vec, allocation_vec  = [], [], []
        N_t = np.zeros(self.K) 
        Theta_hat = np.zeros(self.K)
        w = np.ones(self.K)/self.K
        sol = None
        pi_t = np.copy(w)
        pi_u = np.copy(w)
        t = 1

        


        
        # Initialization (sample each local arm at least once)
        for a in range(self.K):
            a_t = a
            t +=1
            N_t[a_t] += 1
            Theta_hat[a_t] += (env_noise.sample(self.THETA, a_t) - Theta_hat[a_t])/N_t[a_t]
        
        while True:  
            if pre_specified_rate:
                pi_u = self.mab.f(Theta_hat).copy()
                mask = pi_u > 0
                if np.sum(mask) != self.K:
                    pi_u[~mask] = (1 - pi_u.sum()) / (self.K - np.sum(mask))
                elif not np.isclose(np.sum(pi_u), 1.):
                    pi_u = pi_u + (1 - pi_u.sum()) / self.K


            n = 1/alpha
            coeff = t ** alpha
            pi_t = (1-(n-1)/(n*coeff)) * w + (n-1)/(n*coeff) * pi_u
            
            # Sample arm
            a_t = np.random.choice(np.arange(self.K),p=pi_t)
            
            # Update counters and estimated means
            t += 1
            N_t[a_t] += 1
            Theta_hat[a_t] += (env_noise.sample(self.THETA, a_t) - Theta_hat[a_t])/N_t[a_t]
            a_star_hat = np.argmax(Theta_hat) # Best estimated arm
            Delta_hat: NDArray[np.float64] = self.mab.Delta if Theta_hat is None else Theta_hat[a_star_hat] - Theta_hat
            # Solve optimization problem
            try:
                tstar_result = self.mab.solve_T_star(a_star = a_star_hat, THETA= Theta_hat, SOLVER=SOLVER, FAIR=FAIR, VERBOSE=VERBOSE)
                w = tstar_result.wstar
                sol_vec.append(tstar_result.sol_value)
                time_vec.append(tstar_result.computation_time)
            except Exception as e:
                if VERBOSE:
                    print(f"[Iteration {t}] Error solving optimization problem: using previous solution. Error details: {e}")
                tstar_result = None
                sol_vec.append(0)
                time_vec.append(sol_vec)
            allocation_vec.append(N_t/t)

            Z_t = self.Zt(N_t, Delta_hat, a_star_hat, t)
        
            # Stopping rule
            if tstar_result != None:
                #print("T_Z",T,"T_sol",tstar_result.sol_value)
                #if(t > tstar_result.sol_value * self.stopping_rule(N_t, delta)):
                if(Z_t > self.stopping_rule(N_t, delta)):
                    return [t, a_star_hat, time_vec, sol_vec, allocation_vec, None, None]