import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

font_size=8

sns.set()
sns.set_style('ticks')
sns.set_palette('colorblind')
sns.set_context("paper", rc={'font.size': font_size,
                             'axes.linewidth': font_size/12,
                             'patch.linewidth': font_size/12,
                             'lines.linewidth': font_size/10,
                             'xtick.major.width': font_size/12,
                             'xtick.minor.width': font_size/16,
                             'ytick.major.width': font_size/12,
                             'ytick.minor.width': font_size/16,
                             'legend.fontsize': 0.8*font_size,
                             'legend.title_fontsize': font_size,
                             #'legend.handlelength': 1,
                             'axes.titlesize': font_size,
                             'axes.labelsize': font_size,
                             'xtick.labelsize': 0.8*font_size,
                             'ytick.labelsize': 0.8*font_size})
matplotlib.rcParams.update({'font.family': 'Liberation Sans',
                            'legend.columnspacing': 1.0,
                            'legend.handletextpad': 0.6})


def plot_convergence(df, figsize=(2.760312431, 2.5), err_res: int = 500, res: int = 100, **kwargs):
    """Plots convergence of the KL-divergence and MSE
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the following columns: 
            'method': name of the evaluation method, 
            'query': cumulative number of queries,
            'sq_err': sum-squared-error of the estimate,
            'kl_div': kl-divergence from optimal proposal to approximation.
    
    figsize : tuple
        Figure size
    
    err_res : int
        Frequency at which to display error bars
        
    res : int
        Frequency at which to plot the line
        
    **kwargs : dict
        May include 'mse_ylim' key, 'ci' key (passed to seaborn)
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
    ci = kwargs.get('ci', 95)
    
    # Plot KL divergence
    sns.lineplot(data=df.loc[df['query'] % err_res == 0, ], y='kl_div', x='query', hue='method', 
                 style='method', ci=ci, err_style='bars', legend=False, ax=ax[0])
    for line in ax[0].lines:
        line.set_linestyle("None")
    sns.lineplot(data=df.loc[df['query'] % res == 0, ], y='kl_div', x='query', hue='method',
                 style='method', ci=None, ax=ax[0])
    ax[0].set_ylabel('KL div.', fontweight='bold')
    ax[0].set_yscale('log')
    ax[0].xaxis.set_ticks_position('none')
    
    ax[0].legend(title='Method', bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=3, borderaxespad=0.)
    
    plt.subplots_adjust(hspace=0.07)
    
    # Plot MSE
    sns.lineplot(data=df.loc[df['query'] % err_res == 0, ], y='sq_err', x='query', hue='method',
                 style='method', ci=ci, err_style='bars', ax=ax[1], legend=False)
    for line in ax[1].lines:
        line.set_linestyle("None")
    sns.lineplot(data=df.loc[df['query'] % res == 0, ], y='sq_err', x='query', hue='method',
                 style='method', ci=None, legend=False, ax=ax[1])
    ax[1].set_ylabel('MSE', fontweight='bold')
    ax[1].set_xlabel('Label budget', fontweight='bold')
    ax[1].set_yscale('log')
    mse_ylim = kwargs.get('mse_ylim')
    if mse_ylim:
        ax[1].set_ylim(mse_ylim)
    plt.close()
    
    return fig


def plot_results(df, label_budget: int = 1000, figsize=(2.760312431, 2.5)):
    """Plots MSE for each data set and method after a given label budget
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the following columns: 
            'dataname': data set name,
            'method': name of the evaluation method, 
            'query': cumulative number of queries,
            'sq_err': sum-squared-error of the estimate.
    
    label_budget : int
        Plot MSE after this many labels are consumed.
    
    figsize : tuple
        Figure size
    """    
    df = df.loc[df['query'] == label_budget,]
    fig = plt.figure(figsize=figsize)
    sns.barplot(data=df, x='dataname', y='sq_err', hue='method', errcolor='0')
    plt.xlabel('MSE', fontweight='bold')
    plt.ylabel('Data set', fontweight='bold')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.legend(title='Method', bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=3, borderaxespad=0.)
    return fig
