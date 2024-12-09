import os
import json
import torch
import argparse
import json
import optuna as op
from plotly.io import write_image
import curly_carnival.utils as cc_utils
import curly_carnival.visualization as cc_viz
import curly_carnival.gan.gan_objective as gan_obj
import curly_carnival.gan.models as gan_models

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='-= Hyperparameter Optimization of GANs via Optuna -=', 
        add_help=True,
        )
    
    parser.add_argument(
        'dataset_path',
        metavar='dataset-path',
        type=str,
        help='Path to image dataset, structured like ImageNet',
        )
    
    parser.add_argument(
        'gan',
        type=str, 
        choices=list(gan_models.registered_generators.keys()),
        help='GAN to train',
        )
    
    parser.add_argument(
        'search_space', 
        metavar='search-space', 
        type=str, 
        help='Path to JSON file defining hyperparameter search space',
        )
    
    parser.add_argument(
        '--trials', 
        type=int, 
        default=25, 
        help='Number of trials',
        )

    parser.add_argument(
        '--epochs', 
        type=int, 
        default=90,
        help='Maximum number of training iterations for each model',
        )
    
    parser.add_argument(
        '--optimizer', 
        type=str, 
        choices=cc_utils.SUPPORTED_OPTIMIZERS, 
        default='adam', 
        help='Optimizer to use for each model',
        )
    
    parser.add_argument(
        '--lr-scheduler', 
        type=str, 
        choices=cc_utils.SUPPORTED_LR_SCHEDULERS, 
        default=None, 
        help='Learning rate scheduler to use for each model. If unspecified, no learning rate scheduler is used.',
        )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda', 
        help='Hardware device to use for training/inference',
        )
    
    parser.add_argument(
        '--num-workers', 
        type=int, 
        default=4, 
        help='Number of processes to use for loading data',
        )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Seed to use for controlling randomness throughout training/testing',
        )
            
    parser.add_argument(
        '--plot-results', 
        action='store_true', 
        help='If specified, generate and save various results plots to disk',
        )
            
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='', 
        help='Filepath of directory to save model checkpoints, metrics, and related outputs to',
        )
    
    return parser

def main(args:argparse.Namespace):
    # save command line args to disk
    output_dir = os.getcwd()
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'cmd_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device(args.device)

    print("Defining objective...")
    obj = gan_obj.GANObjective(
        args.gan, 
        args.optimizer, 
        args.epochs, 
        device, 
        cc_utils.load_dataset(args.dataset_path),
        json.load(open(args.search_space)),
        num_workers=args.num_workers,
        seed=args.seed,
        lr_sched_name=args.lr_scheduler,
        output_dir=output_dir
        )

    print("Defining study...")
    study = op.create_study(
        sampler=op.samplers.TPESampler(seed=args.seed),
        study_name=f"{args.gan}-hyperparam-optimization", 
        directions=['minimize','minimize'],
        )
    
    print("Beginning study...")
    study.optimize(
        obj, 
        n_trials=args.trials,
        ) 

    if args.plot_results:
        print("Plotting results...")

        # see https://github.com/optuna/optuna/discussions/3825 for multi-objective plotting

        plt = cc_viz.plot_optim_hist(study, args.gan.capitalize(), f'Generator Loss', target=lambda trial: trial.values[0])
        write_image(plt, os.path.join(output_dir, f'{args.gan}_genLoss_optim_hist.png'))

        plt = cc_viz.plot_optim_hist(study, args.gan.capitalize(), f'Discriminator Loss', target=lambda trial: trial.values[1])
        write_image(plt, os.path.join(output_dir, f'{args.gan}_discLoss_optim_hist.png'))

        plt = cc_viz.plot_optim_timeline(study, args.gan.capitalize())
        write_image(plt, os.path.join(output_dir, f'{args.gan}_optim_timeline.png'))

        plt = cc_viz.plot_hyperparam_importances(study, args.gan.capitalize(), target=lambda trial: trial.values[0])
        write_image(plt, os.path.join(output_dir, f'{args.gan}_gen_hyperparam_importances.png'))

        plt = cc_viz.plot_hyperparam_importances(study, args.gan.capitalize(), target=lambda trial: trial.values[1])
        write_image(plt, os.path.join(output_dir, f'{args.gan}_disc_hyperparam_importances.png'))

    print("Saving best trial data...")
    best_trial = study.best_trial
    trial_output_dir = os.path.join(output_dir, f"best_trial{best_trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)
    best_trial.user_attrs['train_metrics'].to_csv(os.path.join(trial_output_dir, 'train_metrics.csv'), index=False)
    with open(os.path.join(trial_output_dir, 'hyperparams.json'), 'w') as f:
        json.dump(best_trial.params, f, indent=4)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)