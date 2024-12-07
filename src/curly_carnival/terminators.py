from optuna import terminator

class PatientTerminator(terminator.BaseTerminator):
    '''
    Terminates a study if the objective function doesn't improve by at least a specified amount
    for a certain number of trials after running some initial warmup trials.
    '''

    def __init__(self, n_warmup_trials:int=10, patience:int=10, min_delta:float=0.0):
        super().__init__()
        self.n_warmup_trials = n_warmup_trials
        self.patience = patience
        self.min_delta = min_delta
        self.curr_patience = patience

    def should_terminate(self, study):
        if len(study.trials) > self.n_warmup_trials:
            curr_obj_val = study.trials[-1].value
            if abs(curr_obj_val-study.best_value) < self.min_delta:
                self.curr_patience -= 1
            else:
                self.curr_patience = self.patience
            if self.curr_patience == 0:
                return True
            return False
        else:
            return False
        