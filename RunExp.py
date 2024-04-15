## simulation: construct bandit instance, compare estimation and regret of proposed algorithm and TS
## Run experiments on synthetic environment
from LinBandit import *
import argparse
from visualization import *

parser = argparse.ArgumentParser(
                    prog='LinBandit with Predited Context')
parser.add_argument('-T', type=int, default=50001)  
parser.add_argument('-d', type=int, default=5)      
parser.add_argument('-n', type=int, default=10)  
parser.add_argument('-a', type=int, default=2)  
parser.add_argument('-p', type=float, default=0.05) 
parser.add_argument('-l', type=float, default=1.0) 
parser.add_argument('--offset', type=float, default=0.0) 
parser.add_argument('--decay', action="store_true")  
parser.add_argument('--decay_tmp', type=float, default=1./3)  
parser.add_argument('--decay_coef', type=float, default=1)  
parser.add_argument('--decay_cut', type=float, default=0.)  
parser.add_argument('--sigma', type=float, default=1.0)  
parser.add_argument('--sigma_e', type=float, default=1./4)
parser.add_argument('--scale', type=float, default=1)  
parser.add_argument('--dist_ops', type=int, default=0)  
parser.add_argument('--save_name', type=str, default='res')
parser.add_argument('--theta0', nargs='+', type=int, default=[1, 0, 0, 0, 0])
parser.add_argument('--theta1', nargs='+', type=int, default=[0, 1, -1, 1, -1])
# parser.add_argument('--theta0', nargs='+', type=int, default=[6, 5, 5, 5, 5])
# parser.add_argument('--theta1', nargs='+', type=int, default=[5, 6, 4, 6, 4])

parser.add_argument('--ind_S', type=int, default=50)
parser.add_argument('--save', action='store_true')
# parser.add_argument('--theta0', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
# parser.add_argument('--theta1', nargs='+', type=int, default=[0, 0, 2, 0, 2, 1, 2, 0, 2, 0]) 
args = parser.parse_args()

decay = args.decay
T = args.T
d = args.d
l = args.l
n_experiment = args.n
n_action = args.a
theta = np.zeros((d, n_action))
if args.theta0 is None:
    theta[:, 0] = np.array([1, 0, 0, 0, 0])
    theta[:, 1] = np.array([0, 1, -1, 1, -1])
else:
    if len(args.theta0) == d and len(args.theta1) == d:
        theta[:, 0] = np.array(args.theta0)
        theta[:, 1] = np.array(args.theta1)
    else:
        raise Exception("Mismatched dimensions")
theta = theta * args.scale
theta += args.offset
sigma = args.sigma
Sigma_e = args.sigma_e * np.eye(d)

Sigma_e_list = np.zeros((T, d, d))
for t in range(T):
    Sigma_e_list[t, :, :] = Sigma_e
p_0 = args.p

algs = ['TS', 'UCB', 'MEB', 'MEB_naive', 'oracle']
ind_S = (np.arange(T) > args.ind_S)
pi_nd_list = 0.5 * np.ones((T, n_action))
attr = {'TS': {'name':'TS_w_predicted_state',
               'para': {'rho2': sigma ** 2, 'l': l, 'p_0':p_0, 'decay_cut': args.decay_cut}},
        'UCB': {'name':'UCB_w_predicted_state',
                'para': {'C': 1., 'l': l, 'p_0': p_0, 'decay_cut': args.decay_cut}},
        'MEB': {'name':'online_me_adjust_w_predicted_state',
               'para': {'ind_S':ind_S, 'pi_nd_list':pi_nd_list, 'l':l, 'p_0':p_0, 
                        'decay':decay, 'decay_tmp': args.decay_tmp, 'decay_coef': args.decay_coef,
                        'decay_cut': args.decay_cut}},
        'MEB_naive': {'name':'online_me_adjust_w_predicted_state',
                 'para': {'ind_S':ind_S, 'pi_nd_list':pi_nd_list, 'l':l, 'p_0':p_0, 'naive':True, 
                          'decay':decay, 'decay_tmp': args.decay_tmp, 'decay_coef': args.decay_coef,
                        'decay_cut': args.decay_cut}},
        'oracle': {'name':'oracle',
               'para': {'p_0': p_0, 'decay_cut': args.decay_cut}},
                 }
results = {}
for alg in algs:
    results[alg] = {'estimation_err_sum': np.zeros((T, n_action)),
                    'estimation_err_sum2': np.zeros((T, n_action)), 
                    'regret_err_sum': np.zeros(T),
                    'regret_err_sum2': np.zeros(T)}

for i_experiment in range(n_experiment):
    print(i_experiment, end = ' ')
    Bandit_1 = LinBandit(theta = theta, sigma = sigma)
    
    ## iid x_t
    x_list = np.ones((T, d)) + np.random.multivariate_normal(mean = np.zeros(d), cov = np.eye(d), size = (T))
    Bandit_1.generate_potential_reward_history_list(x_list = x_list, Sigma_e_list = Sigma_e_list, dist_ops = args.dist_ops)
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
    with open('Pickle_files/%s.pickle'%(args.save_name), 'wb') as handle:
        pickle.dump((results, algs, attr, args), handle)
else:
    # this is for Jupyter Notebook
    oPlot = FlowLayout()
    print(args)
    plot_error(results, algs, oPlot, 1)
    plot_regret(results, algs, oPlot, log = True, upper = 9)
    oPlot.PassHtmlToCell()