## simulation: construct bandit instance, compare estimation and regret of proposed algorithm and TS
from LinBandit_Real import *
import argparse
# from visualization import *
import os
import numpy as np
import random

parser = argparse.ArgumentParser(
                    prog='LinBandit with Predited Context on Simulated Env Based on Real Data')
parser.add_argument('-T', type=int, default=150)   
parser.add_argument('-n', type=int, default=10)  
parser.add_argument('-a', type=int, default=2)  # number of actions
parser.add_argument('-p', type=float, default=0.05)  # clip probability level
parser.add_argument('-l', type=float, default=1.0)  # parameter for TS
parser.add_argument('-C', type=float, default=1.0)  # parameter for UCB
parser.add_argument('--decay', action="store_true")  # whether to decay the clip probability
parser.add_argument('--decay_tmp', type=float, default=1./3)  # decay rate
parser.add_argument('--decay_coef', type=float, default=1)  # decay coefficient
parser.add_argument('--decay_cut', type=float, default=0.)  # no clipping after decay_cut * T if decay_cut > 0
parser.add_argument('--sigma', type=float, default=1.0)  # noise variance
parser.add_argument('--sigma_e', type=float, default=1./4)  # measurement error variance
parser.add_argument('--dist_ops', type=int, default=0)  # measurement error distribution option
parser.add_argument('--save_name', type=str, default='res')  # save name
parser.add_argument('--warmup', type=float, default=0.1)  # warmup period

parser.add_argument('--prob_corrupt', action='store_true') # a different setup that corrupts the context with some probability
# parser.add_argument('--theta0', nargs='+', type=int, default=[1, 0, 0, 0, 0])
# parser.add_argument('--theta1', nargs='+', type=int, default=[0, 1, -1, 1, -1])
parser.add_argument('--ind_S', type=int, default=50)  # number of burn-in steps
parser.add_argument('--save', action='store_true')  # whether to save the results
parser.add_argument('--seed', type=int, default=1)  # random seed
# parser.add_argument('--theta0', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
# parser.add_argument('--theta1', nargs='+', type=int, default=[0, 0, 2, 0, 2, 1, 2, 0, 2, 0]) 


args = parser.parse_args()

np.random.seed(seed=args.seed)
random.seed(args.seed)

decay = args.decay
T = args.T
l = args.l
C = args.C
n_experiment = args.n
n_action = args.a
sigma = args.sigma
Sigma_e = args.sigma_e

Sigma_e_list = Sigma_e
# for t in range(T):
    # Sigma_e_list[t, :, :] = Sigma_e
p_0 = args.p

algs = ['Adv_UCB', 'MEB', 'MEB_naive', 'TS', 'UCB']
# algs = ['MEB', 'MEB_naive', 'TS', 'UCB']
ind_S = args.ind_S
pi_nd_list = None
attr = {
        'TS': {'name':'TS_w_predicted_state',
               'para': {'rho2': sigma, 'l': l, 'p_0':p_0, 'decay_cut': args.decay_cut, 'warmup': args.warmup}},
        'UCB': {'name':'UCB_w_predicted_state',
                'para': {'C': C, 'l':l, 'p_0': p_0, 'decay_cut': args.decay_cut, 'warmup': args.warmup}},
        'Adv_UCB': {'name':'Adv_w_predicted_state',
                'para': {'C': C, 'l':l, 'p_0': p_0, 'decay_cut': args.decay_cut, 'warmup': args.warmup}},
        'MEB': {'name':'online_me_adjust_w_predicted_state',
               'para': {'ind_S':ind_S, 'pi_nd_list':pi_nd_list, 'l':l, 'p_0':p_0, 
                        'decay':decay, 'decay_tmp': args.decay_tmp, 'decay_coef': args.decay_coef,
                        'decay_cut': args.decay_cut, 'warmup': args.warmup}},
        'MEB_naive': {'name':'online_me_adjust_w_predicted_state',
                 'para': {'ind_S':ind_S, 'pi_nd_list':pi_nd_list, 'l':l, 'p_0':p_0, 'naive':True, 
                          'decay':decay, 'decay_tmp': args.decay_tmp, 'decay_coef': args.decay_coef,
                        'decay_cut': args.decay_cut, 'warmup': args.warmup}}
                 }
results = {}
for alg in algs:
    results[alg] = {'estimation_err_sum': np.zeros((T, n_action)),
                    'estimation_err_sum2': np.zeros((T, n_action)), 
                    'estimation_err_sum_1': np.zeros((T, n_action)),
                    'estimation_err_sum2_1': np.zeros((T, n_action)),
                    'pi_list': np.zeros(T),
                    'regret_err_sum': np.zeros(T),
                    'regret_err_sum2': np.zeros(T)}


print('Number of experiments: ', n_experiment)
for i_experiment in range(n_experiment):
    print(i_experiment, end = ' ')
    Bandit_1 = LinBandit(prob_corrupt= args.prob_corrupt)
    user_id = np.random.choice(range(1, 38))
    ## iid x_t
    # x_list = np.ones((T, d)) + np.random.multivariate_normal(mean = np.zeros(d), cov = np.eye(d), size = (T))
    Bandit_1.generate_potential_reward_history_list(T, user_id, args.sigma_e, args.sigma)
    for alg in algs:
        #print('data generated')
        alg_func = getattr(Bandit_1, attr[alg]['name'])
        history = alg_func(**(attr[alg]['para']))
        results[alg]['estimation_err_sum'] += history['estimation_err_list']
        results[alg]['estimation_err_sum2'] += history['estimation_err_list']**2
        results[alg]['estimation_err_sum_1'] += history['estimation_err_list_1']
        results[alg]['estimation_err_sum2_1'] += history['estimation_err_list_1']**2
        results[alg]['pi_list'] += history['pi_list'][:, 1]
        results[alg]['regret_err_sum'] += history['regret_list']
        results[alg]['regret_err_sum2'] += history['regret_list']**2

if args.save:
    ## save data
    import pickle

    #!!!!!! Store data, change file name !!!!!!
    if not os.path.exists('Pickle_files_real'):
        os.makedirs('Pickle_files_real')
    with open('Pickle_files_real/%s.pickle'%(args.save_name), 'wb') as handle:
        pickle.dump((results, algs, attr, args), handle)
# else:
#     # this is for Jupyter Notebook
#     oPlot = FlowLayout()
#     print(args)
#     plot_error(results, algs, oPlot, 1)
#     plot_regret(results, algs, oPlot, log = True, upper = 9)
#     oPlot.PassHtmlToCell()