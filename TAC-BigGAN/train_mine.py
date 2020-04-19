""" BigGAN: The Authorized Unofficial PyTorch release
        Code by A. Brock and A. Andonian
        This code is an unofficial reimplementation of
        "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
        by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

        Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
from utils import toggle_grad

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
from torch.utils.tensorboard import SummaryWriter

from Resnet import multi_resolution_resnet, Resnet_mi

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'
    
    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    D = model.Discriminator(**config).to(device)

    ema = None
    
    # FP16?
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?

    print(D)

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'config': config}

    # If parallel, parallelize the GD module
    if config['parallel']:
        D = nn.DataParallel(D)

        if config['cross_replica']:
            patch_replication_callback(D)

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname, 
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # set tensorboard logger
    tb_logdir = '%s/%s/tblogs' % (config['logs_root'], experiment_name)
    if os.path.exists(tb_logdir):
        for filename in os.listdir(tb_logdir):
            if filename.startswith('events'):
                os.remove(os.path.join(tb_logdir, filename))  # remove previous event logs
    tb_writer = SummaryWriter(log_dir=tb_logdir)
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])
    loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size, 'start_itr': state_dict['itr']})

    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])

    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'MINE':
        train = train_fns.MINE_training_function(D, ema, state_dict, config)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()

    print('Beginning training at epoch %d...' % state_dict['epoch'])

    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):    
        # Which progressbar to use? TQDM or my own (mine, ok)?
        if config['pbar'] == 'mine':
            pbar = utils.progress(loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])
        for i, (x, y) in enumerate(pbar):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            D.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            metrics = train(x, y)
            train_log.log(itr=int(state_dict['itr']), **metrics)
            for metric_name in metrics:
                tb_writer.add_scalar('Train/%s' % metric_name, metrics[metric_name], state_dict['itr'])

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                    print(', '.join(['itr: %d' % state_dict['itr']] 
                                     + ['%s : %+4.3f' % (key, metrics[key])
                                     for key in metrics]), end=' ')

        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()