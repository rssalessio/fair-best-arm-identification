# F-BAI

This readme includes the instructions to run the experiments for the paper "Fair Best Arm Identification".

You can find the full paper [here](https://arxiv.org/pdf/2408.17313)
## Python and Libraries

To run the code make sure to use Python 3.11. The libraries needed are:
`Numpy, Cvxpy, Matplotlib, mobile-env, Scipy, tqdm, seaborn, tikzplotlib`.

Additional libraries may be needed (including latex to correctly plot the results)

## Instructions to generate the data

To run the experiment, simply run the script `run_experiment.sh` using `bash run_experiment.sh` in Linux. This script will automatically run the scripts that generate all the data
of the experiments. Double check that the `data` folder includes the data from the `synthetic`  and the `scheduler` environments. Lastly, the notebook `generate_models_env.ipynb` shows how the
environment for the scheduler is constructed.

## Instructions to plot the results

To plot the results in the synthetic model, run the notebook `analyse_result_prespecified.ipynb` for the pre-specified case, and the notebook `analyse_result_thetadep.ipynb` in the $\theta$-dependent case.

To plot the results for the scheduler, run the notebook `analyse_result_scheduling.ipynb`.

To plot the results for the price of fairness, run the notebook `scaling_study_bai.ipynb`.

All images are saved in the `images` folder.

## Instructions to make the tables

Once you have generated all the data, to make the tables that are in the paper simply run the following two jupyter notebooks:

1. `make_table_synthetic.ipynb`
2. `make_table_scheduling.ipynb`
