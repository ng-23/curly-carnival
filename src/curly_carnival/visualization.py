import optuna as op
from optuna.visualization import plot_optimization_history, plot_timeline, plot_param_importances

def plot_optim_hist(study:op.study.Study, model:str, obj_metric:str):
    fig = plot_optimization_history(study)
    fig.update_layout(title=f"Hyperparameter Optimization for {model}", yaxis_title=obj_metric)
    return fig

def plot_optim_timeline(study:op.study.Study, model:str):
    fig = plot_timeline(study)
    fig.update_layout(title=f'Optimization Timeline for {model}')
    return fig

def plot_hyperparam_importances(study:op.study.Study, model:str):
    fig = plot_param_importances(study)
    fig.update_layout(title=f'Hyperparameter Importances for {model}')
    return fig
