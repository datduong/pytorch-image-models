#!/usr/bin/env python
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import sys
import argparse
import os
import csv
import glob
import time
import yaml
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict

import numpy as np
import re, pickle, random

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.models.layers.classifier import create_classifier_layerfc

from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config, we will use them for evaluation', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')

# ! my own args
parser.add_argument("--has_eval_label", action='store_true', default=False,
                    help='on-the-fly aug of eval data')
parser.add_argument('--ave_precompute_aug', action='store_true', default=False,
                    help='average augmentation of each test sample')
parser.add_argument("--aug_eval_data_num", type=int, default=50, # ! aug_eval_data should be "on" in training
                    help='how many data aug, and average them')
parser.add_argument("--aug_eval_data", action='store_true', default=False,
                    help='on-the-fly aug of eval data')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')


def set_jit_legacy():
    """ Set JIT executor to legacy w/ support for op fusion
    This is hopefully a temporary need in 1.5/1.5.1/1.6 to restore performance due to changes
    in the JIT exectutor. These API are not supported so could change.
    """
    #
    assert hasattr(torch._C, '_jit_set_profiling_executor'), "Old JIT behavior doesn't exist!"
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_gpu(True)
    #torch._C._jit_set_texpr_fuser_enabled(True)



def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    if args.legacy_jit:
        set_jit_legacy()

    # create model
    if 'inception' in args.model: 
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            aux_logits=True, # ! add aux loss
            in_chans=3,
            scriptable=args.torchscript)
    else: 
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            in_chans=3,
            scriptable=args.torchscript)

    # ! add more layer to classifier layer
    if args.create_classifier_layerfc: 
        model.global_pool, model.classifier = create_classifier_layerfc(model.num_features, model.num_classes)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = apply_test_time_pool(model, data_config, args)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    if args.amp:
        model = amp.initialize(model.cuda(), opt_level='O1')
    else:
        model = model.cuda()

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    if args.has_eval_label: 
        criterion = nn.CrossEntropyLoss().cuda() # ! don't have gold label

    if os.path.splitext(args.data)[1] == '.tar' and os.path.isfile(args.data):
        dataset = DatasetTar(args.data, load_bytes=args.tf_preprocessing, class_map=args.class_map)
    else:
        dataset = Dataset(args.data, load_bytes=args.tf_preprocessing, class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f: # @valid_labels is index numbering
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']

    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'], # 'blank' is default Image.BILINEAR https://github.com/rwightman/pytorch-image-models/blob/470220b1f4c61ad7deb16dbfb8917089e842cd2a/timm/data/transforms.py#L43
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing,
        auto_augment=args.aa,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        aug_eval_data=args.aug_eval_data, # ! do same data augmentation as train data, so we can eval on many test aug. images
        shuffle=False ) 

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    prediction = None # ! need to save output
    true_label = None 
    
    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + data_config['input_size']).cuda()
        model(input)
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader): # ! not have real label  

            if args.has_eval_label: # ! just save true labels anyway... why not
                if true_label is None: true_label = target.cpu().data.numpy() 
                else: true_label = np.concatenate ( ( true_label,target.cpu().data.numpy() ) , axis=0 )
                
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
                if args.fp16:
                    input = input.half()

            # compute output
            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0] # ! some model returns both loss + aux loss
            
            if valid_labels is not None:
                output = output[:, valid_labels] # ! keep only valid labels ? good to eval by class.

            # ! save prediction, don't append too slow ... whatever ?  
            # ! are names of files also sorted ? 
            if prediction is None: 
                prediction = output.cpu().data.numpy() # batchsize x label 
            else: # stack
                prediction = np.concatenate ( (prediction, output.cpu().data.numpy() ) , axis=0 )
                
            
            if real_labels is not None:
                real_labels.add_result(output)

            if args.has_eval_label: 
                # measure accuracy and record loss
                loss = criterion(output, target) # ! don't have gold standard on testset
                acc1, acc5 = accuracy(output.data, target, topk=(1, args.topk))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1.item(), input.size(0))
                topk.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.has_eval_label and (batch_idx % args.log_freq == 0):
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@topk: {topk.val:>7.3f} ({topk.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, topk=topk))

    if not args.has_eval_label: 
        top1a, topka = 0, 0 # just dummy, because we don't know ground labels
    else:
        if real_labels is not None:
            # real labels mode replaces topk values at the end
            top1a, topka = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=args.topk)
        else:
            top1a, topka = top1.avg, topk.avg

    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        topk=round(topka, 4), topk_err=round(100 - topka, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@topk {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['topk'], results['topk_err']))

    return results, prediction, true_label


def main():
    setup_default_logging()

    args, args_text = _parse_args()
    # args = parser.parse_args()

    if args.ave_precompute_aug and args.aug_eval_data: 
        sys.exit ('check args.ave_precompute_aug and args.aug_eval_data, only 1 can be true ')

    torch.manual_seed(args.seed) # ! mostly for aug on test
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True)
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

    results_file = args.results_file or './results-all.csv'
    if len(model_cfgs):
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r, prediction, true_label = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:  
          
        from HAM10000 import helper
        
        if args.ave_precompute_aug: # ! we pre-compute data aug on test set
            results_file = re.sub (r'\.csv', '', results_file ) + '-ave-aug-offline.csv'
            r, prediction, true_label = validate(args)
        elif args.aug_eval_data: # ! do augmentation on-the-fly
            results_file = re.sub (r'\.csv', '', results_file ) + '-ave-aug-online.csv'
            prediction = None
            for i in range(args.aug_eval_data_num) : # ! slow
                r, prediction_i, true_label = validate(args)
                prediction_i = helper.softmax(prediction_i,theta=1) # convert to range 0-1
                if prediction is None: 
                    prediction = prediction_i
                else: 
                    prediction = prediction + prediction_i
            # average
            prediction = prediction / args.aug_eval_data_num
        else: 
            results_file = re.sub (r'\.csv', '', results_file ) + '-standard.csv'
            r, prediction, true_label = validate(args)

        
        if not args.aug_eval_data: # otherwise we already convert to 0-1 range ? but now a row will not add to exactly 1 ?
            prediction = helper.softmax(prediction,theta=1) # softmax convert to range 0-1, sum to 1

        if args.has_eval_label: 
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            true_label = np.identity(args.num_classes)[true_label] # array into 1 hot
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
            _logger.info ( 'sklearn accuracy_score {} '.format ( accuracy_score(true_label, np.round(prediction)) ) ) 

        # output csv, need to reorder columns
        helper.save_output_csv(prediction, [], results_file, average_augment=args.ave_precompute_aug)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()
        


if __name__ == '__main__':
    main()
