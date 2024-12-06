import os
import torch
import torch.utils
import torchvision
import torch.utils.data
import optuna as op
from optuna import terminator
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2transforms
from curly_carnival.schemas import validate_config
from curly_carnival import terminators
from torch.utils.data import default_collate
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

SUPPORTED_OPTIMIZERS = {
    'adam':optim.Adam, 
    'adamw':optim.AdamW, 
    'sgd':optim.SGD, 
    'rmsprop':optim.RMSprop,
    }

SUPPORTED_LR_SCHEDULERS = {
    'step_lr':optim.lr_scheduler.StepLR, 
    'exponential_lr':optim.lr_scheduler.ExponentialLR, 
    'cosineannealing_lr':optim.lr_scheduler.CosineAnnealingLR,
    }

SUPPORTED_PRUNERS = {
    'median':op.pruners.MedianPruner,
    'patient':op.pruners.PatientPruner,
    }

SUPPORTED_TERMINATORS = {
    'patient':terminators.PatientTerminator,
}

SUPPORTED_COLLATE_FUNCS = {
    'cutmix':v2transforms.CutMix,
    'mixup':v2transforms.MixUp,
    }
        
def encode_str_list(str_list:list[str]):
    return [s.encode('utf-8') for s in str_list]

def gen_default_transforms_config():
    return {
        'Resize':{'size':(224,224),},
        'ToTensor':{},
        'Normalize':{'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225],},
        }

def gen_default_dataloader_config():
    return {
        'batch_size':32,
        'shuffle':False,
        'num_workers':0,
        'pin_memory':True,
        }

def make_collate_func(func_name:str, num_classes:int):
    if func_name not in SUPPORTED_COLLATE_FUNCS:
        raise Exception(f'Unknown/unsupported collate function {func_name}')
    
    func = SUPPORTED_COLLATE_FUNCS[func_name](num_classes=num_classes)
    
    def collate(batch):
        return func(*default_collate(batch))
    
    return collate

def make_transforms_pipeline(transforms_config:dict):
    pipeline = []
        
    for transform_name in transforms_config:
        if not hasattr(transforms, transform_name):
            raise Exception(f'No such transform {transform_name}')
        
        transform_config = validate_config(transforms_config[transform_name], transform_name)

        transform_func = getattr(transforms, transform_name)
        pipeline.append(transform_func(**transform_config))
            
    return transforms.Compose(pipeline)

def load_dataset(filepath:str):
    return torchvision.datasets.ImageFolder(root=filepath)

def split_dataset(dataset:torchvision.datasets.ImageFolder, train_ratio:float=0.6, val_ratio:float=0.2, test_ratio:float=0.2, seed=42):
    if train_ratio + val_ratio + test_ratio > 1.0:
        raise Exception('Train/val/test ratios must be <= 1.0')
    
    generator = torch.Generator().manual_seed(seed)
    
    return torch.utils.data.random_split(dataset, [train_ratio, val_ratio, test_ratio], generator=generator)

def make_dataloader(dataset, dataloader_config:dict|None=None, collate_func=None):
    if dataloader_config is None:
        dataloader_config = {} # use dataloader's defaults

    dataloader_config = validate_config(dataloader_config, 'DataLoader')

    return torch.utils.data.DataLoader(dataset, **dataloader_config, collate_fn=collate_func)

def make_optimizer(model_params, optimizer:str, optimizer_config:dict|None=None) -> optim.Optimizer:
    optimizer = optimizer.lower()

    if optimizer_config is None:
        optimizer_config = {} # use optimizer's defaults

    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise Exception(f'Unknown/unsupported optimizer {optimizer}. Supported ones are currently {list(SUPPORTED_OPTIMIZERS.keys())}')
    optim_class = SUPPORTED_OPTIMIZERS[optimizer]

    optimizer_config = validate_config(optimizer_config, optim_class.__name__)

    if optim_class == optim.Adam or optim_class == optim.AdamW:
        optimizer_config['betas'] = (optimizer_config['beta1'],optimizer_config['beta2'])
        del optimizer_config['beta1']; del optimizer_config['beta2']

    return optim_class(model_params, **optimizer_config)

def make_lr_scheduler(optimizer, lr_scheduler:str, lr_scheduler_config:dict) -> optim.lr_scheduler.LRScheduler:
    lr_scheduler = lr_scheduler.lower()

    if lr_scheduler not in SUPPORTED_LR_SCHEDULERS:
        raise Exception(f'Unknown/unsupported learning rate scheduler {lr_scheduler}. Supported learning rate schedulers are {list(SUPPORTED_LR_SCHEDULERS.keys())}')
    lr_scheduler_class = SUPPORTED_LR_SCHEDULERS[lr_scheduler]

    lr_scheduler_config = validate_config(lr_scheduler_config, lr_scheduler_class.__name__)

    return lr_scheduler_class(optimizer, **lr_scheduler_config)

def make_model_ema(model, model_ema_config:dict):
    model_ema_config = validate_config(model_ema_config, 'ModelEMA')

    return AveragedModel(
        model, 
        multi_avg_fn=get_ema_multi_avg_fn(decay=model_ema_config['decay']),
        )

def make_pruner(pruner:str, pruner_config:dict|None=None) -> op.pruners.BasePruner:
    pruner = pruner.lower()

    if pruner_config is None:
        pruner_config = {}

    if pruner not in SUPPORTED_PRUNERS:
        raise Exception(f'Unknown/unsupported pruner {pruner}. Supported ones are currently {list(SUPPORTED_PRUNERS.keys())}')
    pruner_class = SUPPORTED_PRUNERS[pruner]

    pruner_config = validate_config(pruner_config, pruner_class.__name__)

    if pruner_class == op.pruners.PatientPruner:
        pruner_config['wrapped_pruner'] = None # special case

    return pruner_class(**pruner_config)

def make_terminator(terminator:str, terminator_config:dict|None=None) -> terminator.BaseTerminator:
    terminator = terminator.lower()

    if terminator_config is None:
        terminator_config = {}

    if terminator not in SUPPORTED_TERMINATORS:
        raise Exception(f'Unknown/unsupported terminator {terminator}. Supported ones are currently {list(SUPPORTED_TERMINATORS.keys())}')
    
    terminator_class = SUPPORTED_TERMINATORS[terminator]

    return terminator_class(**terminator_config)

def load_search_space(config:dict):
    return validate_config(config, 'SearchSpace')

def save_checkpoint(model, optimizer, epoch, model_ema=None, lr_scheduler=None, output_dir=''):
    checkpoint = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':epoch,
    }
    
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

    if model_ema is not None:
        checkpoint['model_ema'] = model_ema.state_dict()

    torch.save(
        checkpoint, 
        os.path.join(output_dir, f'checkpoint_{epoch}.pth'),
        )
