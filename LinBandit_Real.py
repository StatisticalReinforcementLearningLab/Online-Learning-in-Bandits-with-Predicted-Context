import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

## utility functions
def prob_clip(x, p_0):
    ## clip the probability value x to be between [p_0, 1-p_0]
    ## require 0<=p_0<=1/2
    if (x < p_0):
        return(p_0)
    if (x > (1-p_0)):
        return(1-p_0)
    return(x)

class LinBandit:
    def __init__(self, theta = None, n_action = None, gamma = 0.95, prob_corrupt = False):
        ## initialize a bandit model with parameters theta and phi
        ## r = <theta_a, s> + eta, eta~N(0, sigma^2)
        ## theta: array, each column is a theta_a value
        ## n_action: number of actions of the bandit
        self.theta = theta
        self.gamma = gamma
        self.n_action = n_action
        self.dim = 11
        self.n_action = 2
        self.prob_corrupt = prob_corrupt
      
    def mean_reward(self, s, a):
        ## compute mean reward from input state s and a
        theta = self.theta
        if (theta is None) or (a > theta.shape[1]-1):
            return(None)
        theta_a = self.theta[:, a]
        mu = np.dot(theta_a, s) 
        return(mu)
    
    def realized_reward(self, s, a):
        ## compute realized reward from input state s and a
        sigma = self.sigma
        eta = np.random.normal(0, sigma)
        mu = self.mean_reward(s, a)
        if mu is None:
            return(None)
        r = self.mean_reward(s, a) + eta
        return(r)
        
    def generate_potential_reward_history_list(self, T, user_id, sigma_e, sigma):
        file_location = "HeartStepV1/gee_fit/"
        n_action = 2
        theta_full = pd.read_csv("%scoeffi_pair%d.csv"%(file_location, user_id)).iloc[:, 0].values # with action = 1
        self.d = 7
        theta1 = np.zeros(7)
        theta1[:6] = theta_full[:6]
        theta1[0] += theta_full[6]
        theta1[4] += theta_full[7]
        theta1[5] += theta_full[8]
        theta1[6] = theta_full[-1] + theta_full[-2]
        theta0 = np.zeros(7)
        theta0[:6] = theta_full[:6]
        theta0[6] = theta_full[-1]
        
        
        self.theta = np.zeros((self.d, n_action))
        self.theta[:, 0] = theta0
        self.theta[:, 1] = theta1
        
        x_list = pd.read_csv("%scontext_pair%d.csv"%(file_location, user_id)).loc[:, ['sum_step.log', 'jbsteps30pre.log', 'dec.temperature', 'sd_steps60', 'home.location']]
        x_list.insert(0, 'Intercept', 1)
        x_list['dosage'] = 0

        
        
        self.n = x_list.shape[0]
        self.T = T
        if self.T > self.n:
            rep = int(self.T / self.n) + 1
            x_list = pd.concat([x_list]*rep)
        x_list = x_list.iloc[:self.T, :]
        x_list.index = range(self.T)
        
        
        # add noise to dosage (dosage has not been calculated yet but the noise is generated here)
        x_tilde_list = deepcopy(x_list)
        
        x_tilde_list['dosage'] += np.random.normal(scale=sigma_e, size = self.T) # add noise to dosage

        # x_list_1 = deepcopy(x_list)
        # x_list = x_tilde_list
        # x_tilde_list = x_list_1
        # for t in range(self.T):
            # x_tilde_list.iloc[t, :] += np.random.multivariate_normal(mean=np.zeros(self.d), cov=np.diag([sigma_e]*self.d))
        
        # residual for the reward observation noise
        # residual = pd.read_csv("%sresidual_pair%d.csv"%(file_location, user_id))
        # if self.T > self.n:
        #     residual = pd.concat([residual]*rep).iloc[:self.T, 0].values
        # else:
        #     residual = residual.iloc[:self.T, 0].values
        
        residual = np.random.normal(scale=sigma, size = self.T)
        burden_noise = np.random.normal(scale=sigma/2, size = self.T)
        # potential_reward_list = np.array([residual, residual]).reshape((2, T)).T + np.matmul(x_list, self.theta)
        
        return_list = {
            "x_list": x_list,
            "Sigma_e_list": sigma_e, # note that it is different here
            "x_tilde_list": x_tilde_list,
            # "potential_reward_list": potential_reward_list,
            "residual": residual,
            "burden_noise": burden_noise
        }
        self.potential_list = return_list
        # return(return_list)
        
    def get_cur_x_tilde(self, t, burden):
        x_tilde = deepcopy(self.potential_list['x_tilde_list'].iloc[t, :].values)
        x_tilde[-1] += burden
        if self.prob_corrupt:
            a = np.random.choice([0, 1], p = [0.9, 0.1])    
            if a == 1:
               for i in range(self.d):
                   if i % 2 == 0:
                       x_tilde[i] = 0
        return x_tilde
    def get_cur_reward(self, t, burden):
        x = deepcopy(self.potential_list['x_list'].iloc[t, :].values)
        x[-1] = burden
        return np.matmul(x, self.theta) 
    def get_cur_regret(self, t, burden, pi_t, p_0):
        x = deepcopy(self.potential_list['x_tilde_list'].iloc[t, :].values)
        x[-1] += burden
        r_0, r_1 = np.matmul(x, self.theta)
        return (1-p_0) * np.max([r_0, r_1]) + p_0 * np.min([r_0, r_1]) - pi_t * r_0 - (1-pi_t) * r_1
        
    def get_next_burden(self, t, burden, at):
        burden = self.gamma *  burden + at + self.potential_list['burden_noise'][t]# update burden
        # burden = self.gamma *  burden  + self.potential_list['burden_noise'][t]# update burden
        return burden
    def TS_w_predicted_state(self, rho2, l = 1., p_0 = 0.1, decay_cut = 0.0, warmup = 0.1):
        ## runs the TS algorithm on the bandit instance with hyperparameters rho2, l, p_0, ignoring meas. err.
        ##          p_0: minimum selection probability
        ##          rho2: parameter for TS - known noise variance in the reward model
        ##          l: parameter for TS - prior variance independent rho^2 / l
        ## returns: estimation_err_list, cumulative_regret
        ##          history also contains: x_tilde_list, potential_reward_list, at_dag_list,
        ##          theta_est_list: all estimated theta (posterior mean), n_action * T * d
        ##          pi_list: policy, T * n_action
        ##          at_list: action, T
        ##          cumulative_regret = \sum_t [E_{pi_t_dag}mu(x_t, a) - E_{pi_t}mu(x_t, a)]
        
        # variables needed for TS
        d = self.potential_list['x_list'].shape[1]
        T = self.potential_list['x_list'].shape[0]
        theta = self.theta # a d by 2 matrix
        n_action = self.n_action # always set to be 2
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        burden = 0.0 # current burden level
        estimation_err_list = np.zeros((T, n_action))
        estimation_err_list_1 = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
        ## algorithm initialization
        regret = 0. # cumulative regret up to current time
        t = 0 # current time (from 0 to T-1)
        Vt = np.zeros((n_action, d, d))  # Vt[a, :, :] = l * I + sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau \tilde X_\tau^\top
        for a in range(n_action):
            Vt[a, :, :] = l * np.eye(d)
        bt = np.zeros((n_action, d)) # bt[a, :] = sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau r_\tau
        
        ## algorithm iterations
        while t < T:
            x_tilde_t = self.get_cur_x_tilde(t, burden)
            
            ## compute posterior mean/var for all actions using Vt, bt
            post_mean_t = np.zeros((n_action, d))
            post_var_t = np.zeros((n_action, d, d))
            for a in range(n_action):
                # print(Vt[a, :, :])
                post_mean = np.matmul(np.linalg.inv(Vt[a, :, :]), bt[a, :])
                post_mean_t[a, :] = post_mean
                post_var_t[a, :, :] = rho2 * np.linalg.inv(Vt[a, :, :])
                theta_est_list[a, t, :] = post_mean
                estimation_err_list[t, a] =  (post_mean[6] - theta[6, a])**2
                # estimation_err_list[t, a] = 0
                estimation_err_list_1[t, a] = np.linalg.norm(post_mean - theta[:, a])
            
            ## compute the probability theta_0 drawn is more optimal, adjust with clipping probability to form pi_t_0
            # compute mean and var of <x_tilde_t, theta_0-theta_1> when thetas drawn from the posteriors
            post_mean_w_x_tilde = np.dot(x_tilde_t, post_mean_t[0, :] - post_mean_t[1, :])
            post_var_w_x_tilde = np.dot(x_tilde_t, np.matmul(post_var_t[0, :, :] + post_var_t[1, :, :], x_tilde_t))
            prob_0 = 1 - stats.norm.cdf(-post_mean_w_x_tilde / np.sqrt(post_var_w_x_tilde))  # the probability of sampling 0 without clipping constraint
            if decay_cut > 0 and t > int(decay_cut * T):
                p_0 = 0
            pi_0 = prob_clip(prob_0, p_0) # cli
            # pi_0 = 0.528
            pi_list[t, 0] = pi_0
            pi_list[t, 1] = 1 - pi_0
            
            ## sample from pi_t to obtain a_t, observe reward
            if t < int(warmup * T):
                at = np.random.binomial(1, 0.5)
            else:
                at = np.random.binomial(1, 1 - pi_0) 
            at_list[t] = at
            rt_0, rt_1 = self.get_cur_reward(t, burden)
            rt = [rt_0, rt_1][at] + self.potential_list['residual'][t]
            
            regret_t = self.get_cur_regret(t, burden, pi_0, p_0)
            regret = regret + regret_t  # update cumulative regret
            regret_list[t] = regret
            
            burden = self.get_next_burden(t, burden, at) # update burden
            
            if decay_cut > 0 and t > int(decay_cut * T):
                t = t+1
                continue
            
            ## update Vt, bt for all actions (only posterior w. action a is changed)
            Vt[at, :, :] = Vt[at, :, :] + np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d)))
            bt[at, :] = bt[at, :] + x_tilde_t * rt

            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "estimation_err_list_1": estimation_err_list_1,
            "regret_list": regret_list
        }
        return(history)    

    def Adv_w_predicted_state(self, C, gamma = 0.1, l = 1., p_0 = 0.1, decay_cut = 0., warmup = 0.1):
        ## runs the UCB algorithm on the bandit instance with hyperparameters C, l, p_0, ignoring meas. err.
        ##          p_0: minimum selection probability
        ##          C: parameter for UCB - coefficient before confidence width in UCB
        ##          l: parameter for UCB - parameter in ridge regression
        ##          x_tilde_list: list of predicted contexts
        ##          potential_reward_list: list of potential reward (to compare with other algorithms)
        ## returns: estimation_err_list, cumulative_regret
        ##          history also contains: x_tilde_list, potential_reward_list, at_dag_list,
        ##          theta_est_list: all estimated theta (posterior mean), n_action * T * d
        ##          pi_list: policy, T * n_action
        ##          at_list: action, T
        ##          cumulative_regret = \sum_t [E_{pi_t_dag}mu(x_t, a) - E_{pi_t}mu(x_t, a)]
        d = self.potential_list['x_list'].shape[1]
        T = self.potential_list['x_list'].shape[0]
        theta = self.theta # a d by 2 matrix
        n_action = self.n_action # always set to be 2

        # Candidate attack budget
        num = int(np.ceil(np.log(4* T)/np.log(2)))
        J_list = [2**j for j in range(num+1)]
        J = len(J_list)
        weights = [1]*J

        H = np.sqrt(T)
        alpha = min(1, np.sqrt(J * np.log(J) / (np.e-1) /(T / H)))
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, n_action))
        estimation_err_list_1 = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
        ## algorithm initialization
        regret = 0. # cumulative regret up to current time
        Vt = np.zeros((n_action, d, d))  # Vt[a, :, :] = l * I + sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau \tilde X_\tau^\top
        for a in range(n_action):
            Vt[a, :, :] = l * np.eye(d)
        bt = np.zeros((n_action, d)) # bt[a, :] = sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau r_\tau
        t = 0 # current time (from 0 to T-1)
        burden = 0.0

        ji = 0
        cur_sum = 0
        p = [1.0/J for _ in range(J)]
        
        ## algorithm iterations
        while t < T:
            if t % H == 0 and t > 0:
                # reset when an epoch ends
                Vt = np.zeros((n_action, d, d))  # Vt[a, :, :] = l * I + sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau \tilde X_\tau^\top
                for a in range(n_action):
                    Vt[a, :, :] = l * np.eye(d)
                bt = np.zeros((n_action, d)) # bt[a, :] = sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau r_\tau

                hat_y = [0 for _ in range(J)]
                hat_y[ji] = cur_sum / p[ji] / H
                
                weights = [weights[j] * np.exp(alpha / J * hat_y[j]) for j in range(J)]

                p = [alpha / J + (1-alpha) * (weights[j] / np.sum(weights)) for j in range(J)]
                ji = np.random.choice(range(J), p = p)

                cur_sum = 0
            
            x_tilde_t = self.get_cur_x_tilde(t, burden)
            
            ## compute best action using Vt, bt, update estimation error
            UCB_list = np.zeros(n_action)
            for i_action in range(n_action):
                theta_a_hat = np.matmul(np.linalg.inv(Vt[i_action, :, :]), bt[i_action, :].reshape((d, 1))).reshape(d)
                mu_a = np.dot(theta_a_hat, x_tilde_t)
                sigma_a2 = np.dot(x_tilde_t, np.matmul(np.linalg.inv(Vt[i_action, :, :]), x_tilde_t))
                UCB_list[i_action] = mu_a + (C + gamma * J_list[ji]) * np.sqrt(sigma_a2)
                theta_est_list[i_action, t, :] = theta_a_hat
                estimation_err_list[t, i_action] = (theta_a_hat[6] - theta[6, i_action])**2
                estimation_err_list_1[t, i_action] = np.linalg.norm(theta_a_hat - theta[:, i_action])
            at_ucb = np.argmax(UCB_list)  # best action from UCB
            
            ## compute current policy pi_t, adjust with clipping, compute regret
            if decay_cut > 0 and t > int(decay_cut * T):
                p_0 = 0
            for i_action in range(n_action):
                if (i_action == at_ucb):
                    pi_list[t, i_action] = 1 - (n_action - 1) * p_0
                else:
                    pi_list[t, i_action] = p_0*1.0
            pi_0 = pi_list[t, 0]
            if t < int(warmup * T):
                at = np.random.binomial(1, 0.5)
            else:
                at = np.random.binomial(1, 1 - pi_0) 
            at_list[t] = at
            rt_0, rt_1 = self.get_cur_reward(t, burden)
            rt = [rt_0, rt_1][at] + self.potential_list['residual'][t]
            
            cur_sum += rt
            
            burden = self.get_next_burden(t, burden, at) # update burden
            
            regret_t = self.get_cur_regret(t, burden, pi_0, p_0)
            regret = regret + regret_t  # update cumulative regret
            regret_list[t] = regret
            if decay_cut > 0 and t > int(decay_cut * T):
                t = t+1
                continue
            
            ## update Vt, bt for all actions (only data w. action a is changed)
            Vt[at, :, :] = Vt[at, :, :] + np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d)))
            bt[at, :] = bt[at, :] + x_tilde_t * rt

            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "estimation_err_list_1": estimation_err_list_1,
            "regret_list": regret_list
        }
        return(history) 

    def UCB_w_predicted_state(self, C, l = 1., p_0 = 0.1, decay_cut = 0., warmup = 0.1):
        ## runs the UCB algorithm on the bandit instance with hyperparameters C, l, p_0, ignoring meas. err.
        ##          p_0: minimum selection probability
        ##          C: parameter for UCB - coefficient before confidence width in UCB
        ##          l: parameter for UCB - parameter in ridge regression
        ##          x_tilde_list: list of predicted contexts
        ##          potential_reward_list: list of potential reward (to compare with other algorithms)
        ## returns: estimation_err_list, cumulative_regret
        ##          history also contains: x_tilde_list, potential_reward_list, at_dag_list,
        ##          theta_est_list: all estimated theta (posterior mean), n_action * T * d
        ##          pi_list: policy, T * n_action
        ##          at_list: action, T
        ##          cumulative_regret = \sum_t [E_{pi_t_dag}mu(x_t, a) - E_{pi_t}mu(x_t, a)]
        d = self.potential_list['x_list'].shape[1]
        T = self.potential_list['x_list'].shape[0]
        theta = self.theta # a d by 2 matrix
        n_action = self.n_action # always set to be 2
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, n_action))
        estimation_err_list_1 = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
        ## algorithm initialization
        regret = 0. # cumulative regret up to current time
        Vt = np.zeros((n_action, d, d))  # Vt[a, :, :] = l * I + sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau \tilde X_\tau^\top
        for a in range(n_action):
            Vt[a, :, :] = l * np.eye(d)
        bt = np.zeros((n_action, d)) # bt[a, :] = sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau r_\tau
        t = 0 # current time (from 0 to T-1)
        burden = 0.0
        
        ## algorithm iterations
        while t < T:
            x_tilde_t = self.get_cur_x_tilde(t, burden)
            
            ## compute best action using Vt, bt, update estimation error
            UCB_list = np.zeros(n_action)
            for i_action in range(n_action):
                theta_a_hat = np.matmul(np.linalg.inv(Vt[i_action, :, :]), bt[i_action, :].reshape((d, 1))).reshape(d)
                mu_a = np.dot(theta_a_hat, x_tilde_t)
                sigma_a2 = np.dot(x_tilde_t, np.matmul(np.linalg.inv(Vt[i_action, :, :]), x_tilde_t))
                UCB_list[i_action] = mu_a + C * np.sqrt(sigma_a2)
                theta_est_list[i_action, t, :] = theta_a_hat
                estimation_err_list[t, i_action] = (theta_a_hat[6] - theta[6, i_action])**2
                estimation_err_list_1[t, i_action] = np.linalg.norm(theta_a_hat - theta[:, i_action])
            at_ucb = np.argmax(UCB_list)  # best action from UCB
            
            ## compute current policy pi_t, adjust with clipping, compute regret
            if decay_cut > 0 and t > int(decay_cut * T):
                p_0 = 0
            for i_action in range(n_action):
                if (i_action == at_ucb):
                    pi_list[t, i_action] = 1 - (n_action - 1) * p_0
                else:
                    pi_list[t, i_action] = p_0*1.0
            pi_0 = pi_list[t, 0]
            if t < int(warmup * T):
                at = np.random.binomial(1, 0.5)
            else:
                at = np.random.binomial(1, 1 - pi_0)
            at_list[t] = at
            rt_0, rt_1 = self.get_cur_reward(t, burden)
            rt = [rt_0, rt_1][at] + self.potential_list['residual'][t]
            
            burden = self.get_next_burden(t, burden, at) # update burden
            
            regret_t = self.get_cur_regret(t, burden, pi_0, p_0)
            regret = regret + regret_t  # update cumulative regret
            regret_list[t] = regret
            if decay_cut > 0 and t > int(decay_cut * T):
                t = t+1
                continue
            
            ## update Vt, bt for all actions (only data w. action a is changed)
            Vt[at, :, :] = Vt[at, :, :] + np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d)))
            bt[at, :] = bt[at, :] + x_tilde_t * rt

            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "estimation_err_list_1": estimation_err_list_1,
            "regret_list": regret_list
        }
        return(history) 
    
    def online_me_adjust_w_predicted_state(self, ind_S, pi_nd_list = None, l = 1.0, p_0 = 0.1,
                                        naive = False, decay = False, decay_tmp = -1./3, 
                                        decay_coef = 1, decay_cut = 0., alpha = 0.2, warmup = 0.1):
        ## runs the online meas. adjust algorithm on the bandit instance with hyperparameters Sigma_e_hat_list, p_0, ind_S, pi_nd_list
        ##          p_0: minimum selection probability
        ##          ind_S: binary vector indicating if algorithm update model estimate

        ## returns: estimation_err_list, cumulative_regret
        ##          history also contains: x_tilde_list, potential_reward_list, at_dag_list,
        ##          theta_est_list: all estimated theta (posterior mean), n_action * T * d
        ##          pi_list: policy, T * n_action
        ##          at_list: action, T
        ##          cumulative_regret = \sum_t [E_{pi_t_dag}mu(x_t, a) - E_{pi_t}mu(x_t, a)]
        
        d = self.potential_list['x_list'].shape[1]
        T = self.potential_list['x_list'].shape[0]
        n_action = self.n_action
        theta = self.theta
        
        if type(ind_S) is int:
            ind_S = (np.arange(T) > ind_S)
        if pi_nd_list is None:
            pi_nd_list = 0.5 * np.ones((T, n_action))
        
        
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, n_action))
        estimation_err_list_1 = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
        # generate sigma_e list
        Sigma_e_hat_list = np.zeros((T, d, d))
        sigma_e = self.potential_list['Sigma_e_list']
        for t in range(T):
            Sigma_e_hat_list[t, 6, 6] = sigma_e
            # Sigma_e_hat_list[t, :, :] = np.diag([sigma_e]*self.d)
        ## algorithm initialization
        t = 0
        # s_t = -1  # s_t: the most recent update index before time t
        V_t = np.zeros((n_action, d, d))  # Vt = \sum_{\tau<t} iw_\tau * 1_{a_\tau = a} * x_tilde_\tau x_tilde_\tau^\top, iw_t = \pi^nd_t(a_t)/\pi_t(a_t)
        for a in range(n_action):
            V_t[a, :, :] = l * np.eye(d)
        W_t = np.zeros((n_action, d, d))  # Wt = \sum_{\tau<t} \pi_\tau^nd(a) * Sigma_e_hat[\tau, :, :]
        b_t = np.zeros((n_action, d))  # bt = \sum_{\tau<t} iw_\tau * 1_{a_\tau = a} * x_tilde_\tau r_\tau
        # V_st = np.zeros((n_action, d, d))  # most recent recorded V_t
        # W_st = np.zeros((n_action, d, d))  # most recent recorded W_t
        # b_st = np.zeros((n_action, d))  # most recent recorded b_t
        theta_st = np.zeros((n_action, d))  # most recent estimated theta
        
        burden = 0.0
        regret = 0.  # cumulative regret
        
        ## algorithm iterations
        while t<T:
            x_tilde_t = self.get_cur_x_tilde(t, burden)
            
            ## get most recent theta estimation, compute policy and regret
            if (t == 0): # no theta available, take theta=0
                pi_t_0 = .5
                pi_t_1 = .5
                pi_list[t, 0] = pi_t_0
                pi_list[t, 1] = pi_t_1
            else:
                ## compute best action using Vt, bt, update estimation error
                UCB_list = np.zeros(n_action)
                for i_action in range(n_action):
                    theta_a_hat = theta_st[i_action, :]
                    mu_a = np.dot(theta_a_hat, x_tilde_t)
                    UCB_list[i_action] = mu_a #+ 1.0 * np.sqrt(sigma_a2)
                    # theta_est_list[i_action, t, :] = theta_a_hat
                    # estimation_err_list[t, i_action] = np.linalg.norm(theta_a_hat - theta[:, i_action])
                at_tilde = np.argmax(UCB_list)  # best action from UCB
                # at_tilde = np.argmax(np.matmul(theta_st, x_tilde_t))
                if not decay:
                    if decay_cut > 0 and t > int(T * decay_cut):
                        p_0 = 0
                    pi_list[t, at_tilde] = 1 - p_0
                    pi_list[t, 1-at_tilde] = p_0
                else:
                    pp = min(t**(-decay_tmp)*decay_coef, 0.5)
                    pi_list[t, at_tilde] = 1-pp
                    pi_list[t, 1-at_tilde] = pp
            
            ## sample from pi_t to obtain a_t, observe reward
            pi_0 = pi_list[t, 0]
            if t < int(warmup * T):
                at = np.random.binomial(1, 0.5)
            else:
                at = np.random.binomial(1, 1 - pi_0) 
            at_list[t] = at
            rt_0, rt_1 = self.get_cur_reward(t, burden)
            rt = [rt_0, rt_1][at] + self.potential_list['residual'][t]
            
            regret_t = self.get_cur_regret(t, burden, pi_0, p_0)
            regret = regret + regret_t  # update cumulative regret
            regret_list[t] = regret
            
            burden = self.get_next_burden(t, burden, at) # update burden
            
            ## record theta
            for a in range(n_action):
                theta_est_list[a, t, :] = theta_st[a, :]
                estimation_err_list[t, a] = (theta_st[a, 6] - theta[6, a])**2
                estimation_err_list_1[t, a] = np.linalg.norm(theta_st[a, :] - theta[:, a])
            ## update V, W, b; most recent V, W, b; s_t, theta_st
            if naive == False:
                imp_weight_at = pi_nd_list[t, at] / pi_list[t, at]
            else:
                imp_weight_at = 1.
            for a in range(n_action):
                # update V and b
                if (a == at):
                    V_t[a, :, :] = V_t[a, :, :] + imp_weight_at * np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d)))
                    b_t[a, :] = b_t[a, :] + imp_weight_at * x_tilde_t * rt
                # update W
                if naive == False:
                    W_t[a, :, :] = W_t[a, :, :] + pi_nd_list[t, a] * Sigma_e_hat_list[t, :, :]
                elif (a == at):
                    W_t[a, :, :] = W_t[a, :, :] + Sigma_e_hat_list[t, :, :]
            if (ind_S[t] == 1):
                # if decay_cut == 0 or t < int(T * decay_cut):                    
                for a in range(n_action):
                    theta_st[a, :] = np.matmul(np.linalg.inv(V_t[a, :, :] - W_t[a, :, :]), b_t[a, :].reshape((d, 1))).reshape(d)
            else:
                for a in range(n_action):
                    theta_st[a, :] = np.matmul(np.linalg.inv(V_t[a, :, :]), b_t[a, :].reshape((d, 1))).reshape(d)
            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "estimation_err_list_1": estimation_err_list_1,
            "regret_list": regret_list
        }
        return(history)    
    