import numpy as np
from numpy.typing import NDArray
from algorithms.MAB import MAB
from algorithms.bai.bai_algorithm import BAIAlgorithm
from typing import Callable
from envs.env_noise import EnvNoise, GaussianEnvNoise

class UniformBAI(BAIAlgorithm):
    def __init__(self, mab: MAB, fast_stopping_rule: bool = True):
        """
        Initialize UniformBAI.

        Args:
            mab (MAB): MAB instance.
            fast_stopping_rule (bool, optional): if True, enables a faster algorithms that is less accurate. Defauls to true.
        """
        super().__init__(mab, fast_stopping_rule)

    def solve(self, delta, FAIR: bool = True, T_MAX: int = 1e12, env_noise: EnvNoise = GaussianEnvNoise()):
        """ 
        Run UniformBAI algorithm
        """
        # Variables and counters initialization
        time_vec, sol_vec, allocation_vec  = [], [], []
        N_t = np.zeros(self.K) 
        Theta_hat = np.zeros(self.K)   
        t = 1
        
        # Initialization (sample each local arm at least once)
        for a in range(self.K):
            a_t = a
            t +=1
            N_t[a_t] += 1
            Theta_hat[a_t] += (env_noise.sample(self.mab.THETA, a_t) - Theta_hat[a_t])/N_t[a_t]
        
        while t<T_MAX:  
            # Sample arm
            p = self.mab.f(Theta_hat).copy()
            if FAIR and np.random.uniform() <= np.sum(p):
                a_t = np.random.choice(self.K, p = p / np.sum(p))
            else:
                a_t = np.random.choice(self.K)
            
            # Update counters and estimated means
            t += 1
            N_t[a_t] += 1
            Theta_hat[a_t] += (env_noise.sample(self.mab.THETA, a_t) - Theta_hat[a_t])/N_t[a_t]
            a_star_hat = np.argmax(Theta_hat) # Best estimated arm
            Delta_hat: NDArray[np.float64] = self.mab.Delta if Theta_hat is None else Theta_hat[a_star_hat] - Theta_hat
            
            # Solve optimization problem
            #try:
            #    tstar_result = self.mab.solve_T_star(a_star = a_star_hat, THETA= Theta_hat, SOLVER=SOLVER, FAIR=FAIR, VERBOSE=VERBOSE)
            #    w = tstar_result.wstar
            #    sol_vec.append(tstar_result.sol_value)
            #    time_vec.append(tstar_result.computation_time)
            #except Exception as e:
            #    print(f"[Iteration {t}] Error solving optimization problem: using previous solution. Error details: {e}")
            #    tstar_result = None
            #    sol_vec.append(0)
            #    time_vec.append(sol_vec)
            allocation_vec.append(N_t/t)
            Z_t = self.Zt(N_t, Delta_hat, a_star_hat, t)
    
            # Stopping rule
            #if tstar_result != None:
            #if(t > tstar_result.sol_value * self.stopping_rule(N_t, delta)):
            if(Z_t > self.stopping_rule(N_t, delta)):
                return [t, a_star_hat, time_vec, sol_vec, allocation_vec, None, None]
        return [t, a_star_hat, time_vec, sol_vec, allocation_vec, None, None]