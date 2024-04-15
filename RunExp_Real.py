## simulation: construct bandit instance, compare estimation and regret of proposed algorithm and TS
## run experiments on HeartStep V1 dataset
from LinBandit_Real import *
import argparse
from visualization import *

parser = argparse.ArgumentParser(
                    prog='LinBandit with Predited Context on Simulated Env Based on Real Data')
parser.add_argument('-T', type=int, default=150)   
parser.add_argument('-n', type=int, default=10)  
parser.add_argument('-a', type=int, default=2)  
parser.add_argument('-p', type=float, default=0.05) 
parser.add_argument('-l', type=float, default=1.0) 
parser.add_argument('-C', type=float, default=1.0) 
parser.add_argument('--decay', action="store_true")  
parser.add_argument('--decay_tmp', type=float, default=1./3)  
parser.add_argument('--decay_coef', type=float, default=1)  
parser.add_argument('--decay_cut', type=float, default=0.)  
parser.add_argument('--sigma', type=float, default=1.0)  
parser.add_argument('--sigma_e', type=float, default=1./4)
parser.add_argument('--dist_ops', type=int, default=0)  
parser.add_argument('--save_name', type=str, default='res')
# parser.add_argument('--theta0', nargs='+', type=int, default=[1, 0, 0, 0, 0])
# parser.add_argument('--theta1', nargs='+', type=int, default=[0, 1, -1, 1, -1])
parser.add_argument('--ind_S', type=int, default=50)
parser.add_argument('--save', action='store_true')
# parser.add_argument('--theta0', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
# parser.add_argument('--theta1', nargs='+', type=int, default=[0, 0, 2, 0, 2, 1, 2, 0, 2, 0]) 


args = parser.parse_args()

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

algs = ['TS', 'UCB', 'MEB', 'MEB_naive']
ind_S = args.ind_S
pi_nd_list = None
attr = {'TS': {'name':'TS_w_predicted_state',
               'para': {'rho2': sigma ** 2, 'l': l, 'p_0':p_0, 'decay_cut': args.decay_cut}},
        'UCB': {'name':'UCB_w_predicted_state',
                'para': {'C': C, 'l':l, 'p_0': p_0, 'decay_cut': args.decay_cut}},
        'MEB': {'name':'online_me_adjust_w_predicted_state',
               'para': {'ind_S':ind_S, 'pi_nd_list':pi_nd_list, 'l':l, 'p_0':p_0, 
                        'decay':decay, 'decay_tmp': args.decay_tmp, 'decay_coef': args.decay_coef,
                        'decay_cut': args.decay_cut}},
        'MEB_naive': {'name':'online_me_adjust_w_predicted_state',
                 'para': {'ind_S':ind_S, 'pi_nd_list':pi_nd_list, 'l':l, 'p_0':p_0, 'naive':True, 
                          'decay':decay, 'decay_tmp': args.decay_tmp, 'decay_coef': args.decay_coef,
                        'decay_cut': args.decay_cut}}
                 }
results = {}
for alg in algs:
    results[alg] = {'estimation_err_sum': np.zeros((T, n_action)),
                    'estimation_err_sum2': np.zeros((T, n_action)), 
                    'regret_err_sum': np.zeros(T),
                    'regret_err_sum2': np.zeros(T)}

for i_experiment in range(n_experiment):
    print(i_experiment, end = ' ')
    Bandit_1 = LinBandit()
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
        results[alg]['regret_err_sum'] += history['regret_list']
        results[alg]['regret_err_sum2'] += history['regret_list']**2

if args.save:
    ## save data
    import pickle

    #!!!!!! Store data, change file name !!!!!!
    with open('Pickle_files_real/%s.pickle'%(args.save_name), 'wb') as handle:
        pickle.dump((results, algs, attr, args), handle)
else:
    # this is for Jupyter Notebook
    oPlot = FlowLayout()
    print(args)
    plot_error(results, algs, oPlot, 1)
    plot_regret(results, algs, oPlot, log = True, upper = 9)
    oPlot.PassHtmlToCell()