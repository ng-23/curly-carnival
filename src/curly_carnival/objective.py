import curly_carnival.schemas
import curly_carnival.utils
import optuna as op
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torchvision
import curly_carnival
import time
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class Objective():
    METRICS = {'acc','recall','precision','f1'}
    METRIC_FUNCS = {'acc':accuracy_score, 'recall':recall_score, 'precision':precision_score, 'f1':f1_score}

    def __init__(
            self, 
            model_name:str, 
            optimizer_name:str, 
            epochs:int,
            device:torch.device,
            dataset:torchvision.datasets.ImageFolder,
            search_space:dict,
            train_val_test_ratios=(0.6,0.2,0.2),
            train_transfms_conf={},
            val_transfms_conf={},
            test_transfms_conf={},
            eval_batch_size=128,
            num_workers=4,
            seed=42,
            lr_sched_name:str='',
            avg_method='micro',
            obj_metric:str='acc',
            output_dir='',
            ):
        
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.lr_sched_name = lr_sched_name
        self.search_space = curly_carnival.schemas.validate_config(search_space, 'SearchSpace')
        self.num_classes = len(dataset.classes)
        self.seed = seed
        self.avg_method = avg_method

        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        (self.train_ratio, self.val_ratio, self.test_ratio) = train_val_test_ratios
        self.train_ds, self.val_ds, self.test_ds = curly_carnival.utils.split_dataset(
            dataset, 
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
            )
        self.train_ds_transfms = curly_carnival.utils.make_transforms_pipeline(
        curly_carnival.utils.gen_default_transforms_config() if len(train_transfms_conf) == 0 else train_transfms_conf,
        )
        self.val_ds_transfms = curly_carnival.utils.make_transforms_pipeline(
        curly_carnival.utils.gen_default_transforms_config() if len(val_transfms_conf) == 0 else val_transfms_conf,
        )
        self.test_ds_transfms = curly_carnival.utils.make_transforms_pipeline(
        curly_carnival.utils.gen_default_transforms_config() if len(test_transfms_conf) == 0 else test_transfms_conf,
        )
        self.train_ds.dataset.transform = self.train_ds_transfms
        self.val_ds.dataset.transform = self.val_ds_transfms
        self.test_ds.dataset.transform = self.test_ds_transfms

        if obj_metric not in self.METRICS:
            raise ValueError(f'Invalid objective metric {obj_metric}. Valid options are {self.METRICS}')
        self.obj_metric = obj_metric

    def _load_hyperparams(self, trial:op.Trial):
        hyperparams = {'general_conf':{}, 'model_conf':{}, 'optim_conf':{}, 'lr_sched_conf':{}}

        for param_group in self.search_space:
            params = self.search_space[param_group]
            for param_name in params:
                val = None
                val_search_space = params[param_name]

                type = val_search_space['type']
                if type == 'float':
                    val = trial.suggest_float(
                        param_name, 
                        val_search_space['low'], 
                        val_search_space['high'], 
                        log=val_search_space['log'],
                        )
                elif type == 'int':
                    val = trial.suggest_int(
                        param_name, 
                        val_search_space['low'], 
                        val_search_space['high'], 
                        step=val_search_space['step'] if val_search_space['step'] is not None else 1, # special case since step cannot be None
                        log=val_search_space['log'],
                        )

                hyperparams[param_group][param_name] = val

        return hyperparams
    
    def _gen_dataloader_conf(self, batch_size:int, for_training=False):
        dl_conf = curly_carnival.utils.gen_default_dataloader_config()
        dl_conf['batch_size'] = batch_size
        dl_conf['num_workers'] = self.num_workers
        if for_training:
            dl_conf['shuffle'] = True
        return dl_conf

    def _calc_metrics(self, preds:np.ndarray, targets:np.ndarray):
        metrics = {metric:0.0 for metric in self.METRICS}

        for metric in metrics:
            if metric == 'acc':
                # special case since accuracy_score doesn't have average kwarg
                metrics[metric] = self.METRIC_FUNCS[metric](preds, targets)
            else:
                metrics[metric] = self.METRIC_FUNCS[metric](preds, targets, average=self.avg_method, zero_division=0.0)

        return metrics
    
    def _save_trial(self, trial:op.Trial):
        output_path = os.path.join(self.output_dir, f'trial{trial.number}')
        os.makedirs(output_path, exist_ok=True)

        # save train/val/test metrics as CSVs
        trial.user_attrs['train_metrics'].to_csv(os.path.join(output_path, 'train_metrics.csv'), index=False)
        trial.user_attrs['val_metrics'].to_csv(os.path.join(output_path, 'val_metrics.csv'), index=False)
        trial.user_attrs['test_metrics'].to_csv(os.path.join(output_path, 'test_metrics.csv'), index=False)

        # save hyperparams as JSON
        with open(os.path.join(output_path, 'hyperparams.json'), 'w') as f:
            json.dump(trial.params, f, indent=4)

    def _train_step(self, model:nn.Module, optimizer:optim.Optimizer, train_dl:torch.utils.data.DataLoader):
        model.train()

        step_metrics = {metric:[] for metric in self.METRICS}
        step_metrics['loss'] = []

        for i, (samples, targets) in tqdm(enumerate(train_dl), desc='Train Step'):
            samples, targets = samples.to(self.device), targets.to(self.device)

            output = model(samples)

            if isinstance(model, torchvision.models.GoogLeNet):
                logits = output[0]
                aux_logits2 = output[1]
                aux_logits1 = output[2]

                logits_loss = self.criterion(logits, targets)
                aux_logits1_loss = self.criterion(aux_logits1, targets)
                aux_logits2_loss = self.criterion(aux_logits2, targets)
                loss = logits_loss + (0.3 * aux_logits1_loss) + (0.3 * aux_logits2_loss)
            else:
                loss = self.criterion(output, targets)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if isinstance(model, torchvision.models.GoogLeNet):
                _, preds = torch.max(output[0], 1)
            else:
                _, preds = torch.max(output, 1)
            batch_metrics = self._calc_metrics(preds.cpu().numpy(), targets.cpu().numpy())
            batch_metrics['loss'] = loss.item()
            for metric in batch_metrics:
                step_metrics[metric].append(batch_metrics[metric])

        for metric in step_metrics:
            step_metrics[metric] = [np.mean(step_metrics[metric])]

        return pd.DataFrame.from_dict(step_metrics)
    
    def _val_step(self, model:nn.Module, val_dl:torch.utils.data.DataLoader):
        model.eval()

        step_metrics = {metric:[] for metric in self.METRICS}
        step_metrics['loss'] = []

        with torch.inference_mode():
            for i, (samples, targets) in tqdm(enumerate(val_dl), desc='Validation Step'):
                samples, targets = samples.to(self.device), targets.to(self.device)

                output = model(samples)

                loss = self.criterion(output, targets)

                _, preds = torch.max(output, 1)
                batch_metrics = self._calc_metrics(preds.cpu().numpy(), targets.cpu().numpy())
                batch_metrics['loss'] = loss.item()
                for metric in batch_metrics:
                    step_metrics[metric].append(batch_metrics[metric])

        for metric in step_metrics:
            step_metrics[metric] = [np.mean(step_metrics[metric])]

        return pd.DataFrame.from_dict(step_metrics)
    
    def _test(self, model:nn.Module, test_dl:torch.utils.data.DataLoader):
        model.eval()

        test_metrics = {metric:[] for metric in self.METRICS}

        with torch.inference_mode():
            for i, (samples, targets) in enumerate(test_dl):
                samples, targets = samples.to(self.device), targets.to(self.device)

                output = model(samples)
                
                _, preds = torch.max(output, 1)
                batch_metrics = self._calc_metrics(preds.cpu().numpy(), targets.cpu().numpy())
                for metric in batch_metrics:
                    test_metrics[metric].append(batch_metrics[metric])

        for metric in test_metrics:
            test_metrics[metric] = [np.mean(test_metrics[metric])]

        return pd.DataFrame.from_dict(test_metrics)

    def __call__(self, trial:op.Trial):
        epochs_train_metrics = pd.DataFrame.from_dict({metric:[] for metric in self.METRICS})
        epochs_val_metrics = pd.DataFrame.from_dict({metric:[] for metric in self.METRICS})

        hyperparams = self._load_hyperparams((trial))

        train_dl = curly_carnival.utils.make_dataloader(
            self.train_ds,
            self._gen_dataloader_conf(hyperparams['general_conf']['batch_size'], for_training=True),
            )
        
        val_dl = curly_carnival.utils.make_dataloader(
            self.val_ds,
            self._gen_dataloader_conf(self.eval_batch_size, for_training=False),
            )
        
        test_dl = curly_carnival.utils.make_dataloader(
            self.val_ds,
            self._gen_dataloader_conf(self.eval_batch_size, for_training=False),
            )
        
        model = torchvision.models.get_model(self.model_name, num_classes=self.num_classes, **hyperparams['model_conf'])
        model.to(self.device)

        hyperparams['optim_conf']['lr'] = hyperparams['general_conf']['lr']
        optimizer = curly_carnival.utils.make_optimizer(
            model.parameters(), 
            self.optimizer_name, 
            optimizer_config=hyperparams['optim_conf'],
            )
        
        lr_scheduler = None
        if self.lr_sched_name:
            if 'lr_sched_conf' not in hyperparams:
                raise Exception('Must supply a learning rate scheduler config if a learning rate scheduler is specified')
            lr_scheduler = curly_carnival.utils.make_lr_scheduler(
                optimizer, 
                self.lr_sched_name, 
                hyperparams['lr_sched_conf'],
                )
        
        print('-'*50)
        print(f'Hyperparameters for trial {trial.number}: \n{hyperparams}')
        for epoch in range(self.epochs):
            print('-'*50)
            print(f'Epoch: {epoch+1}/{self.epochs} for trial {trial.number}')

            # perform a train step
            start_time = time.time()
            train_metrics = self._train_step(model, optimizer, train_dl)
            print(f'Total train step time: {time.time()-start_time} seconds')
            print(f'Training metrics: \n{train_metrics.to_string(index=False)}')
            print('+'*50)

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

            # perform a val step
            start_time = time.time()
            val_metrics = self._val_step(model, val_dl)
            print(f'Total val step time: {time.time()-start_time} seconds')
            print(f'Validation metrics: \n{val_metrics.to_string(index=False)}')  

            # report intermediate value
            trial.report(val_metrics[self.obj_metric].iloc[0], epoch)     
            
            # aggregate metrics
            epochs_train_metrics = pd.concat([epochs_train_metrics, train_metrics])
            epochs_val_metrics = pd.concat([epochs_val_metrics, val_metrics])

            # early stop if using a pruner
            if trial.should_prune():
                print(f'Stopping trial {trial.number} early at epoch {epoch+1}/{self.epochs}')
                trial.set_user_attr('train_metrics', epochs_train_metrics)
                trial.set_user_attr('val_metrics', epochs_val_metrics)
                trial.set_user_attr('test_metrics', self._test(model, test_dl))
                self._save_trial(trial)
                raise op.TrialPruned()
        
        trial.set_user_attr('train_metrics', epochs_train_metrics)
        trial.set_user_attr('val_metrics', epochs_val_metrics)
        trial.set_user_attr('test_metrics', self._test(model, test_dl))

        test_obj_metric = epochs_val_metrics[self.obj_metric].iloc[0]
        
        print('-'*50)
        print(f'Finished trial {trial.number} with final test {self.obj_metric} of {test_obj_metric}')
        self._save_trial(trial)

        return test_obj_metric
    