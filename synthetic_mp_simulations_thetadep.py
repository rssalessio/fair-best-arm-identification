import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from algorithms.MAB import MAB
from algorithms.reward_model import LinearRewardModel, EqualGapRewardModel
from algorithms.bai.fbai import FairBAI
from algorithms.bai.tas import TaS
from algorithms.bai.uniform import UniformBAI
from algorithms.fairness_model import ProportionalFairnessModel, VectorFairnessModel, make_prespecified_fairness_model
from tqdm import tqdm
from numpy.typing import NDArray
import multiprocessing as mp
import pickle
import os
import sys



def run(idx, algorithm, args):
        np.random.seed(idx)
        return algorithm.solve(**args)


if __name__ == '__main__':
    # Convert the first command-line argument to a float
    parameter_value = float(sys.argv[1])
    print(f"Running with parameter value: {parameter_value}")
    SEED = 5
    np.random.seed(SEED)
    K = 15
    R_MAX = 5
    p0_linear = 0.7
    p0_equalgap = 0.95
    SOLVER = cp.ECOS
    NUM_CPUS = 10

    models = [
        ('Linear gaps, agonistic fairness', LinearRewardModel(R_MAX = R_MAX, K=K), ProportionalFairnessModel(K, p0=p0_linear, use_gaps=True, invert=True), 'thetadep_linear_gaps_agonistic'),
        ('Linear gaps, antagonistic fairness', LinearRewardModel(R_MAX = R_MAX, K=K), ProportionalFairnessModel(K, p0=p0_linear, use_gaps=True, invert=False), 'thetadep_linear_gaps_antagonistic'),
    ]
    N_SIM = 100
    DELTA = parameter_value
    ALPHA = 0.5
    folder_name = f'./data/synthetic/delta_{DELTA}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with mp.Pool(NUM_CPUS) as pool:
        for model in models:
            name = model[0]
            filename = model[3]
            reward_model = model[1]
            fairness_model = model[2]

            instance = MAB(reward_model=reward_model, fairness_model=None)
            [w,sol,t] = instance.solve_T_star(SOLVER=SOLVER)


            instance = MAB(reward_model=reward_model, fairness_model=fairness_model)

            tqdm_range = tqdm(range(N_SIM))
            tas = TaS(instance, fast_stopping_rule=True)
            fbai = FairBAI(instance, fast_stopping_rule=True)
            uniform_bai = UniformBAI(instance, fast_stopping_rule=True)

            methods = [
                ('F-BAI', FairBAI,  {'delta': DELTA,'FAIR': True,'VERBOSE':False, 'SOLVER':SOLVER, 'pre_specified_rate':False, 'alpha':0.5}),
                ('TaS', TaS, {'delta': DELTA,'FAIR': False,'VERBOSE':False, 'SOLVER':SOLVER, 'alpha':0.5}),
                ('Uniform Fair', UniformBAI,  {'delta': DELTA,'FAIR': True})
            ]
            results = {
                    algo_name: {
                        't_vec': [],
                        'a_star_vec': [],
                        'w_vec': [],
                        'p_err': None,
                    } for algo_name, _, _ in  methods
            }
            
            for method in methods:
                print(f'Running model {filename} - method {method[0]}')
                
                algo_name = method[0]
                res = pool.starmap(run, [(n, 
                                        method[1](MAB(reward_model=reward_model, fairness_model=fairness_model), fast_stopping_rule=True),
                                        method[2]) for n in range(N_SIM)])
                for (t, a_star_hat, time_vec, sol_vec, allocation_vec, counter_forced, counter_track) in res:
                    results[algo_name]['t_vec'].append(t)
                    results[algo_name]['a_star_vec'].append(a_star_hat)
                    results[algo_name]['w_vec'].append(allocation_vec)
                    

            for algo_name, _,_ in methods:
                results[algo_name]['t_vec'] = np.array(results[algo_name]['t_vec'])
                results[algo_name]['a_star_vec'] = np.array(results[algo_name]['a_star_vec'])
                results[algo_name]['p_err'] = instance.a_star != results[algo_name]['a_star_vec']
                
            with open(f'{folder_name}/{filename}_{DELTA}.pkl', 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
