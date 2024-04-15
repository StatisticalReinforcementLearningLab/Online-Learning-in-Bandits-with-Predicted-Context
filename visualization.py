import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
import io
import base64
## open from saved data
import pickle
from scipy import stats

class FlowLayout(object):
    ''' A class / object to display plots in a horizontal / flow layout below a cell '''
    def __init__(self):
        # string buffer for the HTML: initially some CSS; images to be appended
        self.sHtml =  """
        <style>
        .floating-box {
        display: inline-block;
        margin: 10px;
        border: 3px solid #888888;  
        }
        </style>
        """

    def add_plot(self, oAxes):
        ''' Saves a PNG representation of a Matplotlib Axes object '''
        Bio=io.BytesIO() # bytes buffer for the plot
        fig = oAxes.get_figure()
        fig.canvas.print_png(Bio) # make a png of the plot in the buffer

        # encode the bytes as string using base 64 
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        self.sHtml+= (
            '<div class="floating-box">'+ 
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')

    def PassHtmlToCell(self):
        ''' Final step - display the accumulated HTML '''
        display(HTML(self.sHtml))




# algs = ['TS', 'UCB', 'MEB', 'MEB_naive']
colors = {
    'MEB': 'blue',
    'MEB_naive': 'green',
    'TS': 'red',
    'UCB': 'violet',
    'oracle': 'yellow'
}

## ----estimation error----
# plot estimation error (theta_1)
def plot_error(results, algs, oPlot, i = 1, save = False, sub_sampling = 100, savename = None, warmup=0.):
    n_experiment = 10#results[algs[0]]['regret_err_sum'].shape[1]
    T = results[algs[0]]['regret_err_sum'].shape[0]
    fig, ax = plt.subplots(figsize=(4, 3))
    sub = np.arange(0, int((1-warmup) * T), sub_sampling)
    for alg in algs:
        if alg == 'oracle':
            continue
        mean = results[alg]['estimation_err_sum'][:, i] / n_experiment
        sd = (results[alg]['estimation_err_sum2'][:, i] - mean ** 2) ** 0.5 / (n_experiment)
        
        mean = mean[:int((1-warmup) * T)+1]
        sd = mean[:int((1-warmup) * T)+1]
        ax.plot(np.arange(T)[sub], np.log(mean)[sub], color = colors[alg], markersize=0.2, label = alg)
        ax.fill_between(np.arange(T)[sub], np.log(mean - sd)[sub], \
                        np.log(mean + sd)[sub], color = colors[alg], alpha=0.1)
    ax.legend(loc='upper right')
    oPlot.add_plot(ax)
    if save:
        if savename is None:
            plt.savefig('Figures/tmp_est.pdf')
        else:
            plt.savefig('Figures/%s_est.pdf'%(savename))
    plt.close()
    # fig.show()

## ----regret plot----
def plot_regret(results, algs, oPlot, log = False, upper = 6, warmup = 0.0, save = False, sub_sampling = 100, savename = None):
    n_experiment = 10
    T = results[algs[0]]['regret_err_sum'].shape[0]
    
    warmup_T = int(T * warmup)
    fig, ax = plt.subplots(figsize=(4, 3))
    # plot regret
    T = T - warmup_T
    sub = np.arange(0, T, sub_sampling)
    m, s = [], []
    for alg in algs:
        if alg == 'oracle':
            continue
        sum1 = results[alg]['regret_err_sum'][warmup_T:] 
        sum2 = results[alg]['regret_err_sum2'][warmup_T:]
        
        if 'oracle' in algs:
            sum1 -= results['oracle']['regret_err_sum'][warmup_T:]
            sum2 -= results['oracle']['regret_err_sum2'][warmup_T:]
        
        sum1 = sum1 - sum1[0]
        sum2 = sum2 - sum2[0]
        
        mean = (sum1 / n_experiment)
        sd = ((sum2 - mean ** 2) ** 0.5 / (n_experiment))
        print('Algorithm %s: %f, %f'%(alg, (mean[-1]-mean[0]) / mean.shape[0], sd[-1]/ mean.shape[0]))
        m.append((mean[-1]-mean[0]) / mean.shape[0])
        s.append(sd[-1]/ mean.shape[0])
        if not log:
            ax.plot(np.arange(T)[sub], mean[sub] , color = colors[alg], markersize=0.2, label = alg)
            ax.fill_between(np.arange(T)[sub], (mean - sd)[sub], \
                    (mean + sd)[sub], color = colors[alg], alpha=0.1)
        else:
            ax.plot(np.arange(T)[sub], np.log(mean)[sub] , color = colors[alg], markersize=0.2, label = alg)
            ax.fill_between(np.arange(T)[sub], np.log(mean - sd)[sub], \
                    np.log(mean + sd)[sub], color = colors[alg], alpha=0.1)
    ax.legend(loc='lower right')
    ax.set_ylim([0, upper])
    oPlot.add_plot(ax)
    if save:
        if savename is None:
            plt.savefig('Figures/tmp_regret.pdf')
        else:
            plt.savefig('Figures/%s_regret.pdf'%(savename))
    plt.close()
    return m, s
    # fig.show()