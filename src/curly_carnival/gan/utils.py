from torch import nn

registered_weights_init_funcs = {}

def register_weights_init_func(name:str):
    def wrapper(func):
        registered_weights_init_funcs[name] = func
        return func
    return wrapper

def get_weights_init_func(gan_name:str):
    if gan_name not in registered_weights_init_funcs:
        raise ValueError(f'GAN {gan_name} has no registered weights init function')
    return registered_weights_init_funcs[gan_name]

@register_weights_init_func('DCGAN')
def dcgan_weights_init(m):
    '''
    Taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    '''
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
