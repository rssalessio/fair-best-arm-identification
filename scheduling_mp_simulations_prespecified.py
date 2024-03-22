import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from algorithms.MAB import MAB
from algorithms.reward_model import LinearRewardModel, EqualGapRewardModel,CustomRewardModel
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
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Or if you are using > Python 3.11:
with warnings.catch_warnings(action="ignore"):
    fxn()

def run(idx, algorithm, args):
        np.random.seed(idx)
        return algorithm.solve(**args)

if __name__ == '__main__':
    parameter_value = float(sys.argv[1])
    print(f"Running with parameter value: {parameter_value}")
    SEED = 5
    np.random.seed(SEED)
    K = 10
    p0 = 0.9
    SOLVER = cp.ECOS
    NUM_CPUS = 10
    with open('models/scheduling_env_model/circle_models.pkl', 'rb') as handle:
        THETAs = pickle.load(handle)
    models = [
        ('Pre-specified model agonistic', CustomRewardModel(THETAs[1]), lambda _: make_prespecified_fairness_model(w, p0, agonistic=True), 'Pre-specified model agonistic'),
        ('Pre-specified model antagonistic', CustomRewardModel(THETAs[1]), lambda _: make_prespecified_fairness_model(w,  p0, agonistic=False), 'Pre-specified model antagonistic'),
    ]
    N_SIM = 100
    DELTA = parameter_value
    ALPHA = 0.5
    folder_name = f'./data/scheduler/delta_{DELTA}'
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
            instance = MAB(reward_model=reward_model, fairness_model=fairness_model(w))
            [w_fair,sol_fair,t] = instance.solve_T_star(SOLVER=SOLVER, FAIR=True)
            print(f"[{name}]: 2T*_{{theta}}log(1/delta) = {2*sol*np.log(1/DELTA)}\n2T*_{{theta,p}}log(1/delta) = {2*sol_fair*np.log(1/DELTA)}")
            tqdm_range = tqdm(range(N_SIM))
            tas = TaS(instance, fast_stopping_rule=True)
            fbai = FairBAI(instance, fast_stopping_rule=True)
            uniform_bai = UniformBAI(instance, fast_stopping_rule=True)

            methods = [
                ('F-BAI', FairBAI,  {'delta': DELTA,'FAIR': True,'VERBOSE':False, 'SOLVER':SOLVER, 'pre_specified_rate':True, 'alpha':ALPHA}),
                ('TaS', TaS, {'delta': DELTA,'FAIR': False,'VERBOSE':False, 'SOLVER':SOLVER, 'alpha':ALPHA}),
                ('Uniform Fair', UniformBAI,  {'delta': DELTA,'FAIR': True})
            ]
            results = {
                    algo_name: {
                        't_vec': [],
                        'a_star_vec': [],
                        'w_vec': [],
                        'p_err': None,
                        'p':fairness_model(w)
                    } for algo_name, _, _ in  methods
            }
            
            for method in methods:
                print(f'Running model {filename} - method {method[0]}')
                
                algo_name = method[0]
                res = pool.starmap(run, [(n, 
                                        method[1](MAB(reward_model=reward_model, fairness_model=fairness_model(w)), fast_stopping_rule=True),
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