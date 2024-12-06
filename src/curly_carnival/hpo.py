import os
import json
import torch
import argparse
import json
import optuna as op
from plotly.io import write_image
from curly_carnival import utils, objective, visualization

SUPPORTED_MODELS = {
    'vgg16','vgg19','alexnet','googlenet',
    'resnet18','efficientnet_b0','efficientnet_b1',
    'efficientnet_b2','efficientnet_b3','efficientnet_b4',
    'maxvit_t','vit_b_16','resnet34','resnet50',
    }

SUPPORTED_PRUNERS = {'median'}

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='-= Hyperparameter Optimization via Optuna -=', 
        add_help=True,
        )
    
    parser.add_argument(
        'dataset_path',
        metavar='dataset-path',
        type=str,
        help='Path to image dataset, structured like ImageNet',
        )
    
    parser.add_argument(
        'model',
        type=str, 
        choices=SUPPORTED_MODELS,
        help='Model to train',
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
        '--pruner', 
        type=str, 
        choices=SUPPORTED_PRUNERS, 
        default=None, 
        help='Pruning (early stopping) algorithm to use',
        )
    
    parser.add_argument(
        '--pruner-conf', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for pruner',
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
        choices=utils.SUPPORTED_OPTIMIZERS, 
        default='adam', 
        help='Optimizer to use for each model',
        )
    
    parser.add_argument(
        '--lr-scheduler', 
        type=str, 
        choices=utils.SUPPORTED_LR_SCHEDULERS, 
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
        '--val-ratio', 
        type=float, 
        default=0.2, 
        help='Ratio of data to use for model validation',
        )
    
    parser.add_argument(
        '--test-ratio', 
        type=float, 
        default=0.2, 
        help='Ratio of data to use for model testing',
        )
    
    parser.add_argument(
        '--train-transfms-conf', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for train dataset transforms',
        )

    parser.add_argument(
        '--val-transfms-conf', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for validation dataset transforms',
        )
    
    parser.add_argument(
        '--test-transfms-conf', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for test dataset transforms',
        )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Seed to use for controlling randomness throughout training/testing',
        )
    
    parser.add_argument(
        '--eval-batch-size', 
        type=int, 
        default=128, 
        help='Batch size to use when model is in eval mode. Does not affect training results.',
        )
    
    parser.add_argument(
        '--avg-method', 
        type=str, 
        choices=['micro','macro',], 
        default='micro', 
        help='Averaging method to use when calculating metrics',
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

    train_ds_transfms = utils.gen_default_transforms_config() if args.train_transfms_conf is None else json.load(open(args.train_transfms_conf))
    val_ds_transfms = utils.gen_default_transforms_config() if args.val_transfms_conf is None else json.load(open(args.val_transfms_conf))
    test_ds_transfms = utils.gen_default_transforms_config() if args.test_transfms_conf is None else json.load(open(args.test_transfms_conf))

    train_ratio = 1.0 - (args.val_ratio + args.test_ratio)

    print("Defining objective...")
    obj = objective.Objective(
        args.model, 
        args.optimizer, 
        args.epochs, 
        device, 
        utils.load_dataset(args.dataset_path),
        json.load(open(args.search_space)),
        (train_ratio, args.val_ratio, args.test_ratio),
        train_transfms_conf=train_ds_transfms,
        val_transfms_conf=val_ds_transfms,
        test_transfms_conf=test_ds_transfms,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        lr_sched_name=args.lr_scheduler,
        avg_method=args.avg_method,
        )
    
    pruner = None
    if args.pruner is not None:
        pruner = utils.make_pruner(args.pruner, pruner_config=None if args.pruner_conf is None else json.load(open(args.pruner_conf)))

    print("Defining study...")
    study = op.create_study(
        study_name=f"{args.model}-hyperparam-optimization", 
        direction='maximize', 
        pruner=pruner,
        )
    
    print("Beginning study...")
    study.optimize(obj, n_trials=args.trials)

    if args.plot_results:
        print("Plotting results...")

        plt = visualization.plot_optim_hist(study, args.model.capitalize(), f'Test {obj.obj_metric.capitalize()}')
        write_image(plt, os.path.join(output_dir, f'{args.model}_optim_hist.png'))

        plt = visualization.plot_optim_timeline(study, args.model.capitalize())
        write_image(plt, os.path.join(output_dir, f'{args.model}_optim_timeline.png'))

        plt = visualization.plot_hyperparam_importances(study, args.model.capitalize())
        write_image(plt, os.path.join(output_dir, f'{args.model}_hyperparam_importances.png'))

    print("Saving best trial data...")
    best_trial = study.best_trial
    trial_output_dir = os.path.join(output_dir, f"trial{best_trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)
    best_trial.user_attrs['train_metrics'].to_csv(os.path.join(trial_output_dir, 'train_metrics.csv'))
    best_trial.user_attrs['val_metrics'].to_csv(os.path.join(trial_output_dir, 'val_metrics.csv'))
    best_trial.user_attrs['test_metrics'].to_csv(os.path.join(trial_output_dir, 'test_metrics.csv'))
    with open(os.path.join(trial_output_dir, 'hyperparams.json'), 'w') as f:
        json.dump(best_trial.params, f)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)