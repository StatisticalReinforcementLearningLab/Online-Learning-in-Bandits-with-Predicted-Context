import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

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
    def __init__(self, theta = None, sigma = None, n_action = None):
        ## initialize a bandit model with parameters theta and phi
        ## r = <theta_a, s> + eta, eta~N(0, sigma^2)
        ## theta: array, each column is a theta_a value
        ## n_action: number of actions of the bandit
        self.theta = theta
        self.sigma = sigma
        self.n_action = n_action
        if theta is None:
            self.dim = None
        else:
            self.dim = theta.shape[0]
            self.n_action = theta.shape[1]
      
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
        
    def generate_potential_reward_history_list(self, x_list, Sigma_e_list, dist_ops = 0.):
        ## generate potential reward history (with oracle at) for given context with meas. err. w. known variance
        ## input: bandit instance, 
        ##        x_list: (context list), T*d
        ##        Sigma_e_list: (list of meas. err.), T * d * d
        ##        dist_ops: distribution of meas. err. (0: mv normal, >2: t distribution with ? parameter)
        ## output: x_list, Sigma_e_list,
        ##         x_tilde_list: revealed context list, T*d, gaussian meas. err.!
        ##         potential_reward_list: potential observed reward given each action, T * n_action
        ##         at_dag_list: oracle action 
        theta = self.theta
        d = self.dim
        n_action = self.n_action
        T = x_list.shape[0]
        if (theta is None):
            return(None)
        x_tilde_list = np.zeros((T, d))
        potential_reward_list = np.zeros((T, n_action))
        at_dag_list = np.zeros(T)
        for t in range(T):
            Sigma_e_t = Sigma_e_list[t, :, :] # context noise at the step t
            if (dist_ops == 0):
                # Gaussian distribution
                x_tilde_list[t, :] = x_list[t, :] + np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma_e_t)
            else:
                # t distribution
                nu = dist_ops
                Y = np.random.multivariate_normal(mean=np.zeros(d), cov=(nu-2.0)/nu*Sigma_e_t)
                U = np.random.chisquare(df = nu)
                x_tilde_list[t, :] = x_list[t, :] + np.sqrt(nu / U) * Y
            for a in range(n_action):
                potential_reward_list[t, a] = self.realized_reward(x_list[t, :], a)
            mean_reward_dag = np.matmul(x_list[t, :].reshape((1, d)), theta).reshape(n_action)
            at_dag_list[t] = np.argmax(mean_reward_dag)
        return_list = {
            "x_list": x_list,
            "Sigma_e_list": Sigma_e_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list
        }
        self.potential_list = return_list
        # return(return_list)
    def compute_regret(self, pi_0, p_0, t):
        x_t = self.potential_list['x_list'][t, :]
        theta = self.theta
        at_dag_list_t = np.argmax(np.matmul(x_t.reshape((1, 5)), theta).reshape(2))
        # if (self.potential_list.at_dag_list[t] == 0):
        #     regret_t = np.dot(x_tilde_t, theta[:, 0] - theta[:, 1]) * (1 - p_0 - pi_0) # instantaneous regret
        # else:
        #     regret_t = np.dot(x_tilde_t, theta[:, 1] - theta[:, 0]) * (pi_0 - p_0)
        
        if (at_dag_list_t == 0): # reward of oracle policy
            reward_oracle_t = np.dot(x_t, theta[:, 0]) * (1 - p_0) \
                                  + np.dot(x_t, theta[:, 1]) * p_0        # oracle instantaneous reward
        else:
            reward_oracle_t = np.dot(x_t, theta[:, 1]) * (1 - p_0) \
                                  + np.dot(x_t, theta[:, 0]) * p_0
        reward_policy = np.dot(x_t, theta[:, 0]) * pi_0 \
                            + np.dot(x_t, theta[:, 1]) * (1-pi_0)    # reward of current policy
        regret_t = reward_oracle_t - reward_policy   # instantaneous regret
        return(regret_t)  # update cumulative regret
    def compute_regret_2(self, pi_0, p_0, t):
        x_t = self.potential_list['x_list'][t, :].reshape((1, 5))
        theta = self.theta
        reward_policy = np.dot(x_t, theta[:, 0]) * pi_0 \
                            + np.dot(x_t, theta[:, 1]) * (1-pi_0)    # reward of current policy
        regret_t = reward_policy   # instantaneous regret
        return(regret_t)  # update cumulative regret
    def compute_regret_false(self, pi_0, p_0, t):
        x_tilde_t = self.potential_list['x_tilde_list'][t, :]
        theta = self.theta
        at_dag_list_t = np.argmax(np.matmul(x_tilde_t.reshape((1, 5)), theta).reshape(2))
        # if (self.potential_list.at_dag_list[t] == 0):
        #     regret_t = np.dot(x_tilde_t, theta[:, 0] - theta[:, 1]) * (1 - p_0 - pi_0) # instantaneous regret
        # else:
        #     regret_t = np.dot(x_tilde_t, theta[:, 1] - theta[:, 0]) * (pi_0 - p_0)
        
        if (at_dag_list_t == 0): # reward of oracle policy
            reward_oracle_t = np.dot(x_tilde_t, theta[:, 0]) * (1 - p_0) \
                                  + np.dot(x_tilde_t, theta[:, 1]) * p_0        # oracle instantaneous reward
        else:
            reward_oracle_t = np.dot(x_tilde_t, theta[:, 1]) * (1 - p_0) \
                                  + np.dot(x_tilde_t, theta[:, 0]) * p_0
        reward_policy = np.dot(x_tilde_t, theta[:, 0]) * pi_0 \
                            + np.dot(x_tilde_t, theta[:, 1]) * (1-pi_0)    # reward of current policy
        regret_t = reward_oracle_t - reward_policy   # instantaneous regret
        return(regret_t)  # update cumulative regret

    def TS_w_predicted_state(self, rho2, l = 1., p_0 = 0.1, decay_cut = 0.0):
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
        x_tilde_list, potential_reward_list, at_dag_list = self.potential_list['x_tilde_list'],\
                                                        self.potential_list['potential_reward_list'],\
                                                        self.potential_list['at_dag_list']
        
        d = self.dim
        T = x_tilde_list.shape[0]
        theta = self.theta
        n_action = self.n_action
        if (n_action != 2):
            return(None)
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
        ## algorithm initialization
        regret = 0. # cumulative regret up to current time
        Vt = np.zeros((n_action, d, d))  # Vt[a, :, :] = l * I + sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau \tilde X_\tau^\top
        for a in range(n_action):
            Vt[a, :, :] = l * np.eye(d)
        bt = np.zeros((n_action, d)) # bt[a, :] = sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau r_\tau
        t = 0 # current time (from 0 to T-1)
        
        ## algorithm iterations
        while t < T:
            x_tilde_t = x_tilde_list[t, :]
            
            ## compute posterior mean/var for all actions using Vt, bt
            post_mean_t = np.zeros((n_action, d))
            post_var_t = np.zeros((n_action, d, d))
            for a in range(n_action):
                post_mean = np.matmul(np.linalg.inv(Vt[a, :, :]), bt[a, :])
                post_mean_t[a, :] = post_mean
                post_var_t[a, :, :] = rho2 * np.linalg.inv(Vt[a, :, :])
                theta_est_list[a, t, :] = post_mean
                estimation_err_list[t, a] = np.linalg.norm(post_mean - theta[:, a])
            
            ## compute the probability theta_0 drawn is more optimal, adjust with clipping probability to form pi_t_0
            # compute mean and var of <x_tilde_t, theta_0-theta_1> when thetas drawn from the posteriors
            post_mean_w_x_tilde = np.dot(x_tilde_t, post_mean_t[0, :] - post_mean_t[1, :])
            post_var_w_x_tilde = np.dot(x_tilde_t, np.matmul(post_var_t[0, :, :] + post_var_t[1, :, :], x_tilde_t))
            prob_0 = 1 - stats.norm.cdf(-post_mean_w_x_tilde / np.sqrt(post_var_w_x_tilde))  # the probability of sampling 0 without clipping constraint
            if decay_cut > 0 and t > int(decay_cut * T):
                p_0 = 0
            pi_0 = prob_clip(prob_0, p_0) # clipped probability of sampling 0
            pi_list[t, 0] = pi_0
            pi_list[t, 1] = 1 - pi_0
            regret_t = self.compute_regret(pi_0, p_0, t)
            regret = regret + regret_t  # update cumulative regret
            regret_list[t] = regret
            
            ## sample from pi_t to obtain a_t, observe reward
            at = np.random.binomial(1, 1 - pi_0) 
            at_list[t] = at
            rt = potential_reward_list[t, at]
            
            ## update Vt, bt for all actions (only posterior w. action a is changed)
            Vt[at, :, :] = Vt[at, :, :] + np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d)))
            bt[at, :] = bt[at, :] + x_tilde_t * rt

            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "regret_list": regret_list
        }
        return(history)    
    def oracle(self, p_0, decay_cut = 0.0):
        x_tilde_list, potential_reward_list, at_dag_list = self.potential_list['x_tilde_list'],\
                                                        self.potential_list['potential_reward_list'],\
                                                        self.potential_list['at_dag_list']
        
        d = self.dim
        T = x_tilde_list.shape[0]
        theta = self.theta
        n_action = self.n_action
        if (n_action != 2):
            return(None)
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
        ## algorithm initialization
        regret = 0. # cumulative regret up to current time
        t = 0 # current time (from 0 to T-1)
        
        ## algorithm iterations
        while t < T:
            x_tilde_t = x_tilde_list[t, :]
            prob_0 = 1-np.argmax(np.matmul(x_tilde_t.reshape((1, 5)), theta).reshape(2))
            if decay_cut > 0 and t > int(decay_cut * T):
                p_0 = 0
            pi_0 = prob_clip(prob_0, p_0) # clipped probability of sampling 0
            pi_list[t, 0] = pi_0
            pi_list[t, 1] = 1 - pi_0
            regret_t = self.compute_regret(pi_0, p_0, t)
            regret = regret + regret_t  # update cumulative regret
            regret_list[t] = regret
            
            ## sample from pi_t to obtain a_t, observe reward
            at = np.random.binomial(1, 1 - pi_0) 
            at_list[t] = at
            rt = potential_reward_list[t, at]

            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "regret_list": regret_list
        }
        return(history) 
    
    def UCB_w_predicted_state(self, C, l = 1., p_0 = 0.1, decay_cut = 0.):
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
        x_tilde_list, potential_reward_list, at_dag_list = self.potential_list['x_tilde_list'],\
                                                        self.potential_list['potential_reward_list'],\
                                                        self.potential_list['at_dag_list']
        d = self.dim
        T = x_tilde_list.shape[0]
        theta = self.theta
        n_action = self.n_action
        if (n_action != 2):
            return(None)
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
        ## algorithm initialization
        regret = 0. # cumulative regret up to current time
        Vt = np.zeros((n_action, d, d))  # Vt[a, :, :] = l * I + sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau \tilde X_\tau^\top
        for a in range(n_action):
            Vt[a, :, :] = l * np.eye(d)
        bt = np.zeros((n_action, d)) # bt[a, :] = sum_{\tau<t} 1_{A_\tau = a}\tilde X_\tau r_\tau
        t = 0 # current time (from 0 to T-1)
        
        ## algorithm iterations
        while t < T:
            x_tilde_t = x_tilde_list[t, :]
            
            ## compute best action using Vt, bt, update estimation error
            UCB_list = np.zeros(n_action)
            for i_action in range(n_action):
                theta_a_hat = np.matmul(np.linalg.inv(Vt[i_action, :, :]), bt[i_action, :].reshape((d, 1))).reshape(d)
                mu_a = np.dot(theta_a_hat, x_tilde_t)
                sigma_a2 = np.dot(x_tilde_t, np.matmul(np.linalg.inv(Vt[i_action, :, :]), x_tilde_t))
                UCB_list[i_action] = mu_a + C * np.sqrt(sigma_a2)
                theta_est_list[i_action, t, :] = theta_a_hat
                estimation_err_list[t, i_action] = np.linalg.norm(theta_a_hat - theta[:, i_action])
            at_ucb = np.argmax(UCB_list)  # best action from UCB
            
            ## compute current policy pi_t, adjust with clipping, compute regret
            if decay_cut > 0 and t > int(decay_cut * T):
                p_0 = 0
            for i_action in range(n_action):
                if (i_action == at_ucb):
                    pi_list[t, i_action] = 1 - (n_action - 1) * p_0
                else:
                    pi_list[t, i_action] = p_0*1.0
            regret_t = self.compute_regret(pi_list[t, 0], p_0, t)
            regret = regret + regret_t
            regret_list[t] = regret
            
            ## sample from pi_t to obtain a_t, observe reward
            at = np.random.binomial(1, pi_list[t, 1]) 
            at_list[t] = at
            rt = potential_reward_list[t, at]  
            
            ## update Vt, bt for all actions (only data w. action a is changed)
            Vt[at, :, :] = Vt[at, :, :] + np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d)))
            bt[at, :] = bt[at, :] + x_tilde_t * rt

            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "regret_list": regret_list
        }
        return(history) 
    
    def online_me_adjust_w_predicted_state(self, ind_S, pi_nd_list, l = 1.0, p_0 = 0.1,
                                        naive = False, decay = False, decay_tmp = -1./3, 
                                        decay_coef = 1, decay_cut = 0.):
        ## runs the online meas. adjust algorithm on the bandit instance with hyperparameters Sigma_e_hat_list, p_0, ind_S, pi_nd_list
        ##          p_0: minimum selection probability
        ##          ind_S: binary vector indicating if algorithm update model estimate

        ## returns: estimation_err_list, cumulative_regret
        ##          history also contains: x_tilde_list, potential_reward_list, at_dag_list,
        ##          theta_est_list: all estimated theta (posterior mean), n_action * T * d
        ##          pi_list: policy, T * n_action
        ##          at_list: action, T
        ##          cumulative_regret = \sum_t [E_{pi_t_dag}mu(x_t, a) - E_{pi_t}mu(x_t, a)]
        x_tilde_list, potential_reward_list, at_dag_list, Sigma_e_hat_list =\
                                                        self.potential_list['x_tilde_list'],\
                                                        self.potential_list['potential_reward_list'],\
                                                        self.potential_list['at_dag_list'],\
                                                        self.potential_list['Sigma_e_list']
        d = self.dim
        T = x_tilde_list.shape[0]
        theta = self.theta
        n_action = self.n_action
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
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
        regret = 0.  # cumulative regret
        
        ## algorithm iterations
        while t<T:
            x_tilde_t = x_tilde_list[t, :]
            
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
                    # theta_a_hat = np.matmul(np.linalg.inv(V_t[i_action, :, :]), b_t[i_action, :].reshape((d, 1))).reshape(d)
                    mu_a = np.dot(theta_a_hat, x_tilde_t)
                    sigma_a2 = np.dot(x_tilde_t, np.matmul(np.linalg.inv(V_t[i_action, :, :]), x_tilde_t))
                    UCB_list[i_action] = mu_a + 1.0 * np.sqrt(sigma_a2)
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
            # compute regret
            regret_t = self.compute_regret(pi_list[t, 0], p_0, t)
            regret = regret + regret_t
            regret_list[t] = regret
            
            ## sample action, observe reward
            at = np.random.binomial(1, pi_list[t, 1]) 
            at_list[t] = at
            rt = potential_reward_list[t, at]
            
            ## record theta
            for a in range(n_action):
                theta_est_list[a, t, :] = theta_st[a, :]
                estimation_err_list[t, a] = np.linalg.norm(theta_st[a, :] - theta[:, a])
            
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
                if decay_cut == 0 or t < int(T * decay_cut):                    
                    for a in range(n_action):
                        theta_st[a, :] = np.matmul(np.linalg.inv(V_t[a, :, :] - W_t[a, :, :]), b_t[a, :].reshape((d, 1))).reshape(d)
            else:
                for a in range(n_action):
                    theta_st[a, :] = np.matmul(np.linalg.inv(V_t[a, :, :]), b_t[a, :].reshape((d, 1))).reshape(d)
            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "regret_list": regret_list
        }
        return(history)    
    
    def online_me_adjust_w_predicted_state_mixed(self, x_tilde_list, potential_reward_list, at_dag_list, \
                                           Sigma_e_hat_list, ind_S, pi_nd_list, naive_end_time, p_0 = 0.1):
        ## runs the online meas. adjust algorithm on the bandit instance with hyperparameters Sigma_e_hat_list, p_0, ind_S, pi_nd_list
        ##          perform naive estimation for the first naive_end_time times, then switch to proposed estimator
        ##          p_0: minimum selection probability
        ##          ind_S: binary vector indicating if algorithm update model estimate
        ##          pi_nd_list: stablizing policy, T * n_action
        ##          Sigma_e_hat_list: estimated conditional var of meas. err
        ##          x_tilde_list: list of predicted contexts
        ##          potential_reward_list: list of potential reward (to compare with other algorithms)
        ## returns: estimation_err_list, cumulative_regret
        ##          history also contains: x_tilde_list, potential_reward_list, at_dag_list,
        ##          theta_est_list: all estimated theta (posterior mean), n_action * T * d
        ##          pi_list: policy, T * n_action
        ##          at_list: action, T
        ##          cumulative_regret = \sum_t [E_{pi_t_dag}mu(x_t, a) - E_{pi_t}mu(x_t, a)]
        d = self.dim
        T = x_tilde_list.shape[0]
        theta = self.theta
        n_action = self.n_action
        
        ## initialization of returned values
        theta_est_list = np.zeros((n_action, T, d)) 
        pi_list = np.zeros((T, n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, n_action))
        regret_list = np.zeros(T)
        
        ## algorithm initialization
        t = 0
        s_t = -1  # s_t: the most recent update index before time t
        V_t = np.zeros((n_action, d, d))  # Vt = \sum_{\tau<t} iw_\tau * 1_{a_\tau = a} * x_tilde_\tau x_tilde_\tau^\top, iw_t = \pi^nd_t(a_t)/\pi_t(a_t)
        W_t = np.zeros((n_action, d, d))  # Wt = \sum_{\tau<t} \pi_\tau^nd(a) * Sigma_e_hat[\tau, :, :]
        b_t = np.zeros((n_action, d))  # bt = \sum_{\tau<t} iw_\tau * 1_{a_\tau = a} * x_tilde_\tau r_\tau
        V_st = np.zeros((n_action, d, d))  # most recent recorded V_t
        W_st = np.zeros((n_action, d, d))  # most recent recorded W_t
        b_st = np.zeros((n_action, d))  # most recent recorded b_t
        theta_st = np.zeros((n_action, d))  # most recent estimated theta
        V_t_n = np.zeros((n_action, d, d))  # Vt_n = \sum_{\tau<t} 1_{a_\tau = a} * x_tilde_\tau x_tilde_\tau^\top
        W_t_n = np.zeros((n_action, d, d))  # Wt_n = \sum_{\tau<t} 1_{a_\tau = a} * Sigma_e_hat[\tau, :, :]
        b_t_n = np.zeros((n_action, d))  # bt_n = \sum_{\tau<t} 1_{a_\tau = a} * x_tilde_\tau r_\tau
        V_st_n = np.zeros((n_action, d, d))  # most recent recorded V_t_n
        W_st_n = np.zeros((n_action, d, d))  # most recent recorded W_t_n
        b_st_n = np.zeros((n_action, d))  # most recent recorded b_t_n
        theta_st_n = np.zeros((n_action, d))  # most recent estimated theta (naive)
        regret = 0.  # cumulative regret
        
        ## algorithm iterations
        while t<T:
            x_tilde_t = x_tilde_list[t, :]
            
            ## get most recent theta estimation, compute policy and regret
            if (s_t < 0): # no theta available, take theta=0
                pi_t_0 = .5
                pi_t_1 = .5
                pi_list[t, 0] = pi_t_0
                pi_list[t, 1] = pi_t_1
            else:
                # for first naive_end_time points, make decision according to theta_st_n, then to theta_st
                if (t<naive_end_time):
                    at_tilde = np.argmax(np.matmul(theta_st_n, x_tilde_t))
                else:
                    at_tilde = np.argmax(np.matmul(theta_st, x_tilde_t))
                pi_list[t, at_tilde] = 1 - p_0
                pi_list[t, 1-at_tilde] = p_0
            # compute regret
            regret_t = self.compute_regret(pi_list[t, 0], p_0, t)
            regret = regret + regret_t
            regret_list[t] = regret
            
            ## sample action, observe reward
            at = np.random.binomial(1, pi_list[t, 1]) 
            at_list[t] = at
            rt = potential_reward_list[t, at]          
            
            ## if t<naive_end_time, record theta_n, else record theta
            if (t < naive_end_time):
                for a in range(n_action):
                    theta_est_list[a, t, :] = theta_st_n[a, :]
                    estimation_err_list[t, a] = np.linalg.norm(theta_st_n[a, :] - theta[:, a])
            else:
                for a in range(n_action):
                    theta_est_list[a, t, :] = theta_st[a, :]
                    estimation_err_list[t, a] = np.linalg.norm(theta_st[a, :] - theta[:, a])
            
            ## update V, W, b; most recent V, W, b; s_t, theta_st
            ## if t<naive_end_time, update naive versions, of V, W, b; most recent V, W, b; s_t, theta_st
            imp_weight_at = pi_nd_list[t, at] / pi_list[t, at]
            for a in range(n_action):
                # update V and b, W
                if (a == at):
                    V_t[a, :, :] = V_t[a, :, :] + imp_weight_at * np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d)))
                    b_t[a, :] = b_t[a, :] + imp_weight_at * x_tilde_t * rt
                W_t[a, :, :] = W_t[a, :, :] + pi_nd_list[t, a] * Sigma_e_hat_list[t, :, :]    
            if (ind_S[t] == 1):
                # update most recent theta
                s_t = t
                V_st = V_t
                b_st = b_t
                W_st = W_t
                for a in range(n_action):
                    theta_st[a, :] = np.matmul(np.linalg.inv(V_st[a, :, :] - W_st[a, :, :]), b_st[a, :])
            if (t<naive_end_time):
                for a in range(n_action):
                # update V and b, W
                    if (a == at):
                        V_t_n[a, :, :] = V_t_n[a, :, :] + np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d)))
                        b_t_n[a, :] = b_t_n[a, :] + x_tilde_t * rt
                        W_t_n[a, :, :] = W_t_n[a, :, :] + Sigma_e_hat_list[t, :, :]    
            if (ind_S[t] == 1):
                # update most recent theta
                V_st_n = V_t_n
                b_st_n = b_t_n
                W_st_n = W_t_n
                for a in range(n_action):
                    theta_st_n[a, :] = np.matmul(np.linalg.inv(V_st_n[a, :, :] - W_st_n[a, :, :]), b_st_n[a, :])
                
            ## update t
            t = t + 1
        
        ## history construction
        history = {
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "estimation_err_list": estimation_err_list,
            "regret_list": regret_list
        }
        return(history)