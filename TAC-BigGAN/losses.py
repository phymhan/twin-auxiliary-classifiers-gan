import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss


def loss_bce_dis(dis_fake, dis_real):
  target_fake = torch.tensor(0.).cuda().expand_as(dis_fake)
  target_real = torch.tensor(1.).cuda().expand_as(dis_real)
  loss_fake = F.binary_cross_entropy_with_logits(dis_fake, target_fake)
  loss_real = F.binary_cross_entropy_with_logits(dis_real, target_real)
  return loss_real, loss_fake


def loss_bce_gen(dis_fake):
  target_real = torch.tensor(1.).cuda().expand_as(dis_fake)
  loss = F.binary_cross_entropy_with_logits(dis_fake, target_real)
  return loss


def loss_lsgan_dis(dis_fake, dis_real):
  target_fake = torch.tensor(0.).cuda().expand_as(dis_fake)
  target_real = torch.tensor(1.).cuda().expand_as(dis_real)
  loss_fake = F.mse_loss(dis_fake, target_fake)
  loss_real = F.mse_loss(dis_real, target_real)
  return loss_real, loss_fake


def loss_lsgan_gen(dis_fake):
  target_real = torch.tensor(1.).cuda().expand_as(dis_fake)
  loss = F.mse_loss(dis_fake, target_real)
  return loss


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis