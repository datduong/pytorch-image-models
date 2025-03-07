""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim
from timm.optim import Nadam, RMSpropTF, AdamW, RAdam, NovoGrad, NvNovoGrad, Lookahead, AdamP, SGDP
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

import re

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [    {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay} ]


def set_lr_per_params(args, model, last_layer_list, weight_decay=1e-5, skip_list=()): 
    # return {param1: lr, params2}
    # separate params of base model from last classification layer
    # follow @add_weight_decay
    
    base_params_decay = []
    base_params_no_decay = []
    
    last_layer_params = []
    last_layer_param_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        for layer in last_layer_list: 
            if re.match(layer,name): # ! last classification layer
                last_layer_params.append(param)
                last_layer_param_name.append(name)
            else: # ! base layer
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    base_params_no_decay.append(param)
                else:
                    base_params_decay.append(param)
    # 
    print ('last_layer_param_name {}'.format(last_layer_param_name))
    if args.lr_base_params is None: 
        args.lr_base_params = args.lr/10.0
    
    return [    {'params': last_layer_params, 'weight_decay': args.last_layer_weight_decay }, 
                {'params': base_params_no_decay, 'weight_decay': 0., 'lr': args.lr_base_params }, # lower base param lr by 10x based on their paper
                {'params': base_params_decay, 'weight_decay': weight_decay, 'lr': args.lr_base_params } ]
     

def create_optimizer(args, model, filter_bias_and_bn=True, classification_layer_name=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        # Compensate for the way current AdamW and RAdam optimizers apply LR to the weight-decay
        # I don't believe they follow the paper or original Torch7 impl which schedules weight
        # decay based on the ratio of current_lr/initial_lr
        weight_decay /= args.lr

    if weight_decay and filter_bias_and_bn: # batch norm and bias params
        if classification_layer_name is not None : 
            parameters = set_lr_per_params (args, model, classification_layer_name, weight_decay)
        else: 
            parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0. # reset to 0
    else:
        if classification_layer_name is not None: 
            parameters = set_lr_per_params (args, model, classification_layer_name, weight_decay=0)
        else: 
            parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'momentum':
        optimizer = optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'adamw':
        optimizer = AdamW(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'nadam':
        optimizer = Nadam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'radam':
        optimizer = RAdam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps,
            delta=0.1, wd_ratio=0.01, nesterov=True)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, 
            eps=args.opt_eps, nesterov=True)        
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=weight_decay)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=weight_decay)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedsgd':
        optimizer = FusedSGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'fusedmomentum':
        optimizer = FusedSGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(
            parameters, lr=args.lr, adam_w_mode=False, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(
            parameters, lr=args.lr, adam_w_mode=True, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusednovograd':
        optimizer = FusedNovoGrad(
            parameters, lr=args.lr, betas=(0.95, 0.98), weight_decay=weight_decay, eps=args.opt_eps)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
