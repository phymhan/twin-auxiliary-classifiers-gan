''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os
import torch.nn.functional as F

import utils
import losses
import numpy as np
import pdb


# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def hinge_multi(prob, y, hinge=True):

    len = prob.size()[0]

    index_list = [[], []]

    for i in range(len):
        index_list[0].append(i)
        index_list[1].append(np.asscalar(y[i].cpu().detach().numpy()))

    prob_choose = prob[index_list]
    prob_choose = (prob_choose.squeeze()).unsqueeze(dim=1)

    if hinge == True:
        loss = ((1-prob_choose+prob).clamp(min=0)).mean()
    else:
        loss = (1-prob_choose+prob).mean()

    return loss

# def hinge_multi_gen(prob,y):
#
#     len = prob.size()[0]
#
#     index_list = [[],[]]
#
#     for i in range(len):
#         index_list[0].append(i)
#         index_list[1].append(np.asscalar(y[i].cpu().detach().numpy()))
#
#     prob_choose = prob[index_list]
#     prob_choose = (prob_choose.squeeze()).unsqueeze(dim=1)
#
#     loss = ((-1-prob_choose+prob).clamp(max=0)).mean()
#
#     return loss


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  else:
    raise NotImplementedError

  def train(x, y):
    # train one iteration
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    half_size = config['batch_size']
    counter = 0
    MINE_weight = config['MINE_weight'] if config['weighted_MINE_loss'] else 1.0
    
    # Optionally toggle D and G's "require_grad"

    utils.toggle_grad(D, True)
    utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        gy_bar = y_[torch.randperm(half_size), ...] if D.TQ or D.TP else None
        dy_bar = y[counter][torch.randperm(half_size), ...] if D.TP or D.TQ else None
        D_fake, D_real, mi, c_cls, tP, tP_bar, tQ, tQ_bar = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                                               x[counter], y[counter], gy_bar, dy_bar,
                                                               train_G=False, split_D=config['split_D'], add_bias=True)
        # Compute components of D's loss, average them, and divide by the number of gradient accumulations
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        C_loss = 0.
        MI_P = 0.
        MI_Q = 0.
        if config['loss_type'] == 'fCGAN':
          # MINE-P on real
          etP_bar = torch.mean(torch.exp(tP_bar[half_size:]))
          if D.ma_etP_bar is None:
            D.ma_etP_bar = etP_bar.detach().item()
          D.ma_etP_bar += config['ma_rate'] * (etP_bar.detach().item() - D.ma_etP_bar)
          MI_P = torch.mean(tP[half_size:]) - torch.log(etP_bar) * etP_bar.detach() / D.ma_etP_bar
          # MINE-Q on fake
          etQ_bar = torch.mean(torch.exp(tQ_bar[:half_size]))
          if D.ma_etQ_bar is None:
            D.ma_etQ_bar = etQ_bar.detach().item()
          D.ma_etQ_bar += config['ma_rate'] * (etQ_bar.detach().item() - D.ma_etQ_bar)
          MI_Q = torch.mean(tQ[:half_size]) - torch.log(etQ_bar) * etQ_bar.detach() / D.ma_etQ_bar
        if config['loss_type'] == 'MINE':
          # AC
          C_loss += F.cross_entropy(c_cls[half_size:], y[counter])
          if config['train_AC_on_fake']:
            C_loss += F.cross_entropy(c_cls[:half_size], y_)
          # MINE-Q on fake
          etQ_bar = torch.mean(torch.exp(tQ_bar[:half_size]))
          if D.ma_etQ_bar is None:
            D.ma_etQ_bar = etQ_bar.detach().item()
          D.ma_etQ_bar += config['ma_rate'] * (etQ_bar.detach().item() - D.ma_etQ_bar)
          MI_Q = torch.mean(tQ[:half_size]) - torch.log(etQ_bar) * etQ_bar.detach() / D.ma_etQ_bar
        if config['loss_type'] == 'Twin_AC':
          C_loss += F.cross_entropy(c_cls[half_size:], y[counter]) + F.cross_entropy(mi[:half_size], y_)
          if config['train_AC_on_fake']:
            C_loss += F.cross_entropy(c_cls[:half_size], y_)
        if config['loss_type'] == 'AC':
          C_loss += F.cross_entropy(c_cls[half_size:], y[counter])  # AC should be trained on fake also
          if config['train_AC_on_fake']:
            C_loss += F.cross_entropy(c_cls[:half_size], y_)
        D_loss = (D_loss_real + D_loss_fake + C_loss*config['AC_weight'] - (MI_P + MI_Q)*MINE_weight
                  ) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()

    # Optionally toggle "requires_grad"
    utils.toggle_grad(D, False)
    utils.toggle_grad(G, True)

    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    for accumulation_index in range(config['num_G_accumulations']):
      z_.sample_()
      y_.sample_()
      gy_bar = y_[torch.randperm(half_size), ...] if D.TQ else None
      D_fake, G_z, mi, c_cls, tP, tP_bar, tQ, tQ_bar = GD(z_, y_, gy_bar=gy_bar,
                                                          train_G=True, split_D=config['split_D'],
                                                          return_G_z=True, add_bias=config['loss_type'] != 'fCGAN')
      C_loss = 0.
      MI_loss = 0.
      MI_Q_loss = 0.
      f_div = 0.
      if config['loss_type'] == 'fCGAN':
        # f-div
        f_div = (tQ - tP).mean()  # rev-kl
      if config['loss_type'] == 'MINE':
        # AC
        C_loss += F.cross_entropy(c_cls, y_)
        # MINE-Q
        MI_Q_loss = torch.mean(tQ) - torch.log(torch.mean(torch.exp(tQ_bar)))
      if config['loss_type'] == 'AC' or config['loss_type'] == 'Twin_AC':
        C_loss += F.cross_entropy(c_cls, y_)
        if config['loss_type'] == 'Twin_AC':
          MI_loss = F.cross_entropy(mi, y_)

      G_loss = generator_loss(D_fake) / float(config['num_G_accumulations'])
      C_loss = C_loss / float(config['num_G_accumulations'])
      MI_loss = MI_loss / float(config['num_G_accumulations'])
      MI_Q_loss = MI_Q_loss / float(config['num_G_accumulations'])
      f_div = f_div / float(config['num_G_accumulations'])
      (G_loss + (C_loss - MI_loss)*config['AC_weight'] +
       MI_Q_loss*config['MINE_weight'] + f_div*config['fCGAN_weight']).backward()

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])

    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item()),
           'C_loss': utils.get_tensor_item(C_loss),
           'MI_loss': utils.get_tensor_item(MI_loss),
           'f_div': utils.get_tensor_item(f_div),
           'MI_P': utils.get_tensor_item(MI_P),
           'MI_Q': utils.get_tensor_item(MI_Q)}
    # Return G's loss and the components of D's loss.
    return out
  return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name):
  utils.save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  # Save an additional copy to mitigate accidental corruption if process
  # is killed during a save (it's happened to me before -.-)
  if config['num_save_copies'] > 0:
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' % state_dict['save_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']
    
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
    else:
      fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                  experiment_name,
                                                  state_dict['itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  # For now, every time we save, also save sample sheets
  utils.sample_sheet(which_G,
                     classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                     num_classes=config['n_classes'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_)
  # Also save interp sheets
  for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    utils.interp_sheet(which_G,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=config['n_classes'],
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')


''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log, tb_writer=None):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                               config['num_inception_images'],
                                               num_splits=10)
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
  # If improved over previous best metric, save approrpiate copy
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean), IS_std=float(IS_std), FID=float(FID))
  if tb_writer is not None:
    tb_writer.add_scalar('Test/IS_mean', float(IS_mean), int(state_dict['itr']))
    tb_writer.add_scalar('Test/IS_std', float(IS_std), int(state_dict['itr']))
    tb_writer.add_scalar('Test/FID', float(FID), int(state_dict['itr']))
