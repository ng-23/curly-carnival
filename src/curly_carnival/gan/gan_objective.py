import curly_carnival.gan.models
import curly_carnival.gan.utils
import curly_carnival.schemas
import curly_carnival.utils
import optuna as op
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import json
import torchvision
import curly_carnival
import time
from tqdm import tqdm
import curly_carnival.gan
import os

class GANObjective():
    '''
    Objective function to optimize when optimizing hyperparameters for a GAN 
    
    Code largely taken/adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    '''

    METRICS = {'G_loss','D_loss','D_x','D_G_z1','D_G_z2','G_lr','D_lr'}
    REAL_LABEL = 1.0
    FAKE_LABEL = 0.0
    IMG_SIZE = 64
    NUM_CHANNELS = 3

    def __init__(
            self, 
            model_name:str,
            optimizer_name:str, 
            epochs:int,
            device:torch.device,
            dataset:torchvision.datasets.ImageFolder,
            search_space:dict,
            num_workers=4,
            seed=42,
            lr_sched_name:str='',
            output_dir='',
            ):
        
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.criterion = nn.BCELoss()
        self.epochs = epochs
        self.device = device
        self.num_workers = num_workers
        self.lr_sched_name = lr_sched_name
        self.search_space = curly_carnival.schemas.validate_config(search_space, 'SearchSpace')
        self.seed = seed
        try:
            self.weights_init_func = curly_carnival.gan.utils.get_weights_init_func(model_name)
        except Exception as e:
            self.weights_init_func = None
        self.ds = dataset

        transfms_conf = curly_carnival.utils.gen_default_transforms_config()
        transfms_conf['Resize']['size'] = (self.IMG_SIZE,self.IMG_SIZE)
        self.ds_transfms = curly_carnival.utils.make_transforms_pipeline(transfms_conf)
        self.ds.transform = self.ds_transfms

        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _save_trial(self, trial:op.Trial):
        output_path = os.path.join(self.output_dir, f'trial{trial.number}')
        os.makedirs(output_path, exist_ok=True)

        # save train/val/test metrics as CSVs
        trial.user_attrs['train_metrics'].to_csv(os.path.join(output_path, 'train_metrics.csv'), index=False)

        # save hyperparams as JSON
        with open(os.path.join(output_path, 'hyperparams.json'), 'w') as f:
            json.dump(trial.params, f, indent=4)

    def _load_hyperparams(self, trial:op.Trial):
        hyperparams = {'general_conf':{}, 'model_conf': {}, 'optim_conf':{}, 'lr_sched_conf':{}}

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

    def _train_step(self, generator:nn.Module, discriminator:nn.Module, G_optimizer:optim.Optimizer, D_optimizer:optim.Optimizer, train_dl:torch.utils.data.DataLoader):
        generator.train(); discriminator.train()

        step_metrics = {metric:[] for metric in self.METRICS if metric != 'G_lr' and metric != 'D_lr'}

        for i, (real_samples, _) in tqdm(enumerate(train_dl), desc='Train Step'):
            batch_metrics = {metric:[] for metric in self.METRICS if metric != 'G_lr' and metric != 'D_lr'}

            real_samples, _ = real_samples.to(self.device), _.to(self.device)

            batch_size = real_samples.size(0)

            # 1. Update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()

            # 1a. Train discriminator with batch of entirely real data taken from dataset
            sample_labels = torch.full(
                (batch_size,), 
                fill_value=self.REAL_LABEL, 
                dtype=torch.float, 
                device=self.device,
                )
            
            D_real_preds = discriminator(real_samples).flatten() # discriminator classifies the real images as being either real (1) or fake (0)
            D_loss_real = self.criterion(D_real_preds, sample_labels)
            D_loss_real.backward()

            D_real_confidence = D_real_preds.mean().item() # average of discriminators predictions on batch of real data as either real or fake
            batch_metrics['D_x'].append(D_real_confidence)

            # 1b. Train discriminator with batch of entirely fake data generated from latent vector of generator
            rand_noise = torch.randn(batch_size, generator.latent_size, 1, 1, device=self.device) # tensor with batch_size tensors of dimensions latent_size * 1 * 1

            fake_samples = generator(rand_noise) # generate fake batch of samples using generator
            #fid.update(fake_samples, is_real=False)
            sample_labels = torch.full(
                (batch_size,), 
                fill_value=self.FAKE_LABEL, 
                dtype=torch.float, 
                device=self.device,
                )

            D_fake_preds = discriminator(fake_samples.detach()).flatten()

            D_loss_fake = self.criterion(D_fake_preds, sample_labels)
            D_loss_fake.backward()

            D_fake_confidence = D_fake_preds.mean().item()
            batch_metrics['D_G_z1'].append(D_fake_confidence)

            D_loss_total = D_loss_real + D_loss_real
            batch_metrics['D_loss'].append(D_loss_total.item())

            D_optimizer.step()

            # 2. Update Generator: maximize log(D(G(z)))
            generator.zero_grad()

            # 2a.
            D_fake_preds = discriminator(fake_samples).flatten()

            sample_labels = torch.full(
                (batch_size,), 
                fill_value=self.REAL_LABEL, # goal of generator is to produce real images, so we pretend the fake images generated earlier are real
                dtype=torch.float, 
                device=self.device,
                )

            G_loss = self.criterion(D_fake_preds, sample_labels)
            G_loss.backward()

            # 2b.
            D_fake_confidence = D_fake_preds.mean().item()
            batch_metrics['D_G_z2'].append(D_fake_confidence)

            G_optimizer.step()

            batch_metrics['G_loss'].append(G_loss.item())

            for metric in batch_metrics:
                step_metrics[metric].append(batch_metrics[metric])

        for metric in step_metrics:
            step_metrics[metric] = [np.mean(step_metrics[metric])]

        return pd.DataFrame.from_dict(step_metrics)
    
    def __call__(self, trial:op.Trial):
        epochs_train_metrics = pd.DataFrame.from_dict({metric:[] for metric in self.METRICS})

        hyperparams = self._load_hyperparams((trial))

        train_dl = curly_carnival.utils.make_dataloader(
            self.ds,
            self._gen_dataloader_conf(hyperparams['general_conf']['batch_size']),
            )
        
        generator = curly_carnival.gan.models.get_generator(
            self.model_name, 
            **hyperparams['model_conf'],
            )
        generator.to(self.device)

        discriminator = curly_carnival.gan.models.get_discriminator(
            self.model_name, 
            **hyperparams['model_conf'],
            )
        discriminator.to(self.device)

        if self.weights_init_func is not None:
            generator.apply(self.weights_init_func)
            discriminator.apply(self.weights_init_func)

        hyperparams['optim_conf']['lr'] = hyperparams['general_conf']['lr']
        G_optimizer = curly_carnival.utils.make_optimizer(
            generator.parameters(), 
            self.optimizer_name, 
            optimizer_config=hyperparams['optim_conf'],
            )
        D_optimizer = curly_carnival.utils.make_optimizer(
            discriminator.parameters(), 
            self.optimizer_name, 
            optimizer_config=hyperparams['optim_conf'],
            )
        
        G_lr_scheduler, D_lr_scheduler = None, None
        if self.lr_sched_name:
            if 'lr_sched_conf' not in hyperparams:
                raise Exception('Must supply a learning rate scheduler config if a learning rate scheduler is specified')
            G_lr_scheduler = curly_carnival.utils.make_lr_scheduler(
                G_optimizer, 
                self.lr_sched_name, 
                hyperparams['lr_sched_conf'],
                )
            D_lr_scheduler = curly_carnival.utils.make_lr_scheduler(
                G_optimizer, 
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
            train_metrics = self._train_step(generator, discriminator, G_optimizer, D_optimizer, train_dl)
            print(f'Total train step time: {time.time()-start_time} seconds')

            # adjust learning rate
            if G_lr_scheduler is not None:
                G_lr_scheduler.step()
                train_metrics['G_lr'] = G_lr_scheduler.get_last_lr()
            else:
                train_metrics['G_lr'] = G_optimizer.param_groups[0]['lr']
            if D_lr_scheduler is not None:
                D_lr_scheduler.step()   
                train_metrics['D_lr'] = D_lr_scheduler.get_last_lr()
            else:
                train_metrics['D_lr'] = D_optimizer.param_groups[0]['lr']

            print(f'Training metrics: \n{train_metrics.to_string(index=False)}')
            print('+'*50)
            
            # aggregate metrics
            epochs_train_metrics = pd.concat([epochs_train_metrics, train_metrics])

        trial.set_user_attr('train_metrics', epochs_train_metrics)

        print('-'*50)
        print(f'Finished trial {trial.number} with final generator loss (G_loss) of {trial.user_attrs['train_metrics']['G_loss'].iloc[-1]} and discriminator loss (D_loss) of {trial.user_attrs['train_metrics']['D_loss'].iloc[-1]}')
        self._save_trial(trial)
        print(epochs_train_metrics)
        
        return epochs_train_metrics['G_loss'].iloc[-1], epochs_train_metrics['D_loss'].iloc[-1]
    