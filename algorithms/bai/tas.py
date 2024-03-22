import numpy as np
import cvxpy as cp
import time
from numpy.typing import NDArray
from algorithms.MAB import MAB
from algorithms.bai.bai_algorithm import BAIAlgorithm
from envs.env_noise import EnvNoise, GaussianEnvNoise

class TaS(BAIAlgorithm):
    def __init__(self,
                 mab: MAB,
                 fast_stopping_rule: bool = True):
        """
        Initialize a TaS instance.

         Args:
            mab (MAB): MAB instance.
            fast_stopping_rule (bool, optional): if True, enables a faster algorithms that is less accurate. Defauls to true.
        """
        super().__init__(mab, fast_stopping_rule)
        
    def solve(self, delta: float, u = 1, VERBOSE = False,
              FAIR=False, SOLVER = cp.MOSEK, env_noise: EnvNoise = GaussianEnvNoise(), alpha: float = 0.5, T_MAX: int = 1e12):
        """ 
        Run TaS algorithm
        """
        # Variables and counters initialization
        counter_forced, counter_track, time_vec, sol_vec, allocation_vec  = [], [], [], [], [] 
        N_t = np.zeros(self.K) 
        Theta_hat = np.zeros(self.K) 
        U_t = np.ones(self.K) 
        w = np.ones(self.K)/self.K
        sol = None
        t = 1
        
        # Initialization (sample each local arm at least once)
        for a in range(self.K):
            a_t = a
            t +=1
            N_t[a_t] += 1
            Theta_hat[a_t] += (env_noise.sample(self.mab.THETA, a_t) - Theta_hat[a_t])/N_t[a_t]
            U_t = np.argwhere(N_t <  t** alpha - self.K/2)
        
        while t < T_MAX:
            # Forced exploration
            if any(U_t): 
                counter_forced.append(t) # save forced exploratioon steps 
                a_t = np.argmin(N_t)
            # Tracking
            else:
                counter_track.append(t) 
                if type(w) is np.ndarray:
                    a_t = np.argmax(t*w - N_t)

            # Update counters and estimated means
            t += 1
            N_t[a_t] += 1
            Theta_hat[a_t] += (env_noise.sample(self.mab.THETA, a_t) - Theta_hat[a_t])/N_t[a_t]
            U_t = np.argwhere(N_t <  t** alpha - self.K/2)
            a_star_hat = np.argmax(Theta_hat) # Best estimated arm
            Delta_hat: NDArray[np.float64] = self.mab.Delta if Theta_hat is None else Theta_hat[a_star_hat] - Theta_hat
            
            try:
                tstar_result = self.mab.solve_T_star(a_star = a_star_hat, THETA= Theta_hat, FAIR=FAIR, SOLVER=SOLVER, VERBOSE=VERBOSE)
                w = tstar_result.wstar
                sol_vec.append(tstar_result.sol_value)
                time_vec.append(tstar_result.computation_time)
            except Exception as e:
                if VERBOSE:
                    print(f"[Iteration {t}] Error solving optimization problem:using previous solution. Error details {e}")
                tstar_result = None
                sol_vec.append(0)
                time_vec.append(0)

            allocation_vec.append(N_t/t)
            Z_t = self.Zt(N_t, Delta_hat, a_star_hat, t)

            # Stopping rule
            if tstar_result != None:
                #print("T_Z",T,"T_sol",tstar_result.sol_value)
                #print(f'[{t}] tstar={tstar_result.sol_value} beta={self.stopping_rule(N_t, delta)} - {tstar_result.sol_value * self.stopping_rule(N_t, delta)}')
                #if(t > tstar_result.sol_value * self.stopping_rule(N_t, delta)):
                if(Z_t > self.stopping_rule(N_t, delta)):
                    return [t, a_star_hat, time_vec, 
                            sol_vec, allocation_vec, counter_forced, counter_track]


        