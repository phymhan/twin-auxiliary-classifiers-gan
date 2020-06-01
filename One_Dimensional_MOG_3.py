
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from mmd_metric import polynomial_mmd
import argparse

# Hinge Loss
def loss_hinge_dis_real(dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    return loss_real


def loss_hinge_dis_fake(dis_fake):
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


# Vanilla
def loss_bce_dis_real(dis_output):
    target = torch.tensor(1.).cuda().expand_as(dis_output)
    loss = F.binary_cross_entropy_with_logits(dis_output, target)
    return loss


def loss_bce_dis_fake(dis_output):
    target = torch.tensor(0.).cuda().expand_as(dis_output)
    loss = F.binary_cross_entropy_with_logits(dis_output, target)
    return loss


def loss_bce_gen(dis_fake):
    target_real = torch.tensor(1.).cuda().expand_as(dis_fake)
    loss = F.binary_cross_entropy_with_logits(dis_fake, target_real)
    return loss


def plot_density(flights,binwidth=0.1):
    ax = plt.subplot(1,1,1)

    # Draw the plot
    ax.hist(flights, bins=int(180 / binwidth),
            color='blue', edgecolor='black')

    # Title and labels
    ax.set_title('Histogram with Binwidth = %d' % binwidth, size=30)
    ax.set_xlabel('Delay (min)', size=22)
    ax.set_ylabel('Flights', size=22)

    plt.tight_layout()
    plt.show()


class G_guassian(nn.Module):

    def __init__(self, nz, num_classes=3):
        super(G_guassian, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=nz)
        self.decode = nn.Sequential(
            nn.Linear(nz*2, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            # nn.Tanh(),
            nn.Linear(10, 1),
        )
        self.__initialize_weights()

    def forward(self, z, label, output=None):
        input = torch.cat([z, self.embed(label)], dim=1)
        x = input.view(input.size(0), -1)
        output = self.decode(x)
        return output

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


class D_guassian(nn.Module):

    def __init__(self, num_classes=3, AC=True, TAC=True, Proj=False,
                 dis_mlp=False):
        super(D_guassian, self).__init__()
        self.AC = AC
        self.TAC = TAC
        self.Proj = Proj

        self.encode = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            # nn.Tanh(),
        )
        if dis_mlp:
            self.gan_linear = nn.Sequential(
                nn.Linear(10, 10),
                nn.Tanh(),
                nn.Linear(10, 1)
            )
        else:
            self.gan_linear = nn.Linear(10, 1)
        if self.AC:
            self.ac_linear = nn.Linear(10, num_classes)
        if self.TAC:
            self.tac_linear = nn.Linear(10, num_classes)

        if self.Proj:
            self.embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=10)

        self.__initialize_weights()

    def forward(self, input, y=None):
        x = self.encode(input)
        x = x.view(-1, 10)
        out = self.gan_linear(x)
        if self.Proj:
            out += torch.sum(self.embed(y) * x, dim=1, keepdim=True)

        ac = None
        tac = None
        if self.AC:
            ac = self.ac_linear(x)
        if self.TAC:
            tac = self.tac_linear(x)

        return out, ac, tac

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def train(data1, data2, data3, nz, G, D, optd, optg, loss_type='fc', gan_loss='bce'):
    if gan_loss == 'hinge':
        loss_dis_real, loss_dis_fake, loss_gen = loss_hinge_dis_real, loss_hinge_dis_fake, loss_hinge_gen
    elif gan_loss == 'bce':
        loss_dis_real, loss_dis_fake, loss_gen = loss_bce_dis_real, loss_bce_dis_fake, loss_bce_gen
    else:
        raise NotImplementedError
    bs = 384
    for _ in range(20):
        for i in range(1000):

            #####D step
            for _ in range(1):
                # prepare real data
                data = torch.cat(
                    [data1[128 * i:128 * i + 128], data2[128 * i:128 * i + 128], data3[128 * i:128 * i + 128]],
                    dim=0).unsqueeze(dim=1)
                label = torch.cat([torch.ones(128).cuda().long()*0, torch.ones(128).cuda().long()*1, torch.ones(128).cuda().long()*2], dim=0)

                ###D on real
                d_real, ac_real, tac_real = D(data, label)

                # prepare fake data
                z = torch.randn(bs, nz).cuda()
                fake_label = torch.LongTensor(bs).random_(3).cuda()
                fake_data = G(z, label=fake_label)

                ###D on fake
                d_fake, ac_fake, tac_fake = D(fake_data, fake_label)

                if loss_type == 'hybrid':
                    d_real = d_real + ac_real[range(bs), label] - tac_real[range(bs), label]
                    d_fake = d_fake + ac_fake[range(bs), fake_label] - tac_fake[range(bs), fake_label]

                D_loss = loss_dis_real(d_real) + loss_dis_fake(d_fake)
                if loss_type in ['ac', 'tac', 'fc', 'hybrid']:
                    D_loss += F.cross_entropy(ac_real, label)
                if loss_type in ['ac', 'tac']:
                    D_loss += F.cross_entropy(ac_fake, fake_label)
                if loss_type in ['tac', 'fc', 'hybrid']:
                    D_loss += F.cross_entropy(tac_fake, fake_label)

                optd.zero_grad()
                D_loss.backward()
                optd.step()

            #####G step
            if i % 10 == 0:
                z = torch.randn(bs, nz).cuda()
                fake_label = torch.LongTensor(bs).random_(3).cuda()
                fake_data = G(z, label=fake_label)
                d_fake, ac_fake, tac_fake = D(fake_data, fake_label)

                # G_loss = F.binary_cross_entropy(d_fake, torch.ones(bs).cuda())
                if loss_type == 'hybrid':
                    d_fake = d_fake + ac_fake[range(bs), fake_label] - tac_fake[range(bs), fake_label]
                G_loss = loss_gen(d_fake)

                if loss_type in ['ac', 'tac', 'fc']:
                    G_loss += F.cross_entropy(ac_fake, fake_label)
                if loss_type in ['tac', 'fc']:
                    G_loss -= F.cross_entropy(tac_fake, fake_label)

                optg.zero_grad()
                G_loss.backward()
                optg.step()


def get_start_id(args):
    if args.resume:
        return 0
    else:
        cnt = 0
        suffix = '_mlp' if not args.suffix and args.dis_mlp else args.suffix
        for i in os.listdir(os.path.join('MOG', '1D')):
            if i.startswith(f'{args.distance}_{args.gan_loss}{suffix}_'):
                cnt += 1
        return max(0, cnt - 1)


def evel_model(G, save_path, name, data1, data2, data3, r_data, distance):
    no_graph = False
    nz = 2
    z = torch.randn(10000, nz).cuda()
    label = torch.zeros(10000).long().cuda()  # torch.LongTensor(10000).random_(2).cuda()#
    data1_g = G(z=z, label=label).squeeze().cpu().detach()
    z = torch.randn(10000, nz).cuda()
    label = torch.ones(10000).long().cuda()  # torch.LongTensor(10000).random_(2).cuda()#
    data2_g = G(z=z, label=label).squeeze().cpu().detach()
    z = torch.randn(10000, nz).cuda()
    label = torch.ones(10000).long().cuda() + 1  # torch.LongTensor(10000).random_(2).cuda()#
    data3_g = G(z=z, label=label).squeeze().cpu().detach()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1['score_{0}'.format(0)] = data1_g.numpy()
    df1['score_{0}'.format(1)] = data2_g.numpy()
    df1['score_{0}'.format(2)] = data3_g.numpy()
    g_data = torch.cat([data1_g, data2_g, data3_g], dim=0).numpy()
    np.save(save_path + '/%s_data'%name, g_data)
    df2['score_{0}'.format(2)] = g_data
    if not no_graph:
        fig, ax = plt.subplots(1, 1)
        for s in df1.columns:
            df1[s].plot(kind='kde')
        for s in df2.columns:
            df2[s].plot(style='--', kind='kde')
        plt.xlim((-4, 9 + distance * 2))
        ax.legend(["Class_0", "Class_1", "Class_2", "Marginal"])
        # plt.title(name)
        fig.savefig(save_path + '/%s.eps'%name)
    mean0_0, var0_0 = polynomial_mmd(np.expand_dims(data1_g.numpy(), axis=1), np.expand_dims(data1.cpu().numpy(),axis=1))
    mean0_1, var0_1 = polynomial_mmd(np.expand_dims(data2_g.numpy(), axis=1),
                                     np.expand_dims(data2.cpu().numpy(), axis=1))
    mean0_2, var0_2 = polynomial_mmd(np.expand_dims(data3_g.numpy(), axis=1),
                                     np.expand_dims(data3.cpu().numpy(), axis=1))
    mean0, var0 = polynomial_mmd(np.expand_dims(g_data, axis=1), np.expand_dims(r_data, axis=1))
    with open(save_path + f'/results.txt', 'a+') as f:
        f.write(f'{name}:\n')
        f.write(f'{mean0_0}, {var0_0}\n')
        f.write(f'{mean0_1}, {var0_1}\n')
        f.write(f'{mean0_2}, {var0_2}\n')
        f.write(f'{mean0}, {var0}\n')
    return (mean0_0, var0_0), (mean0_1, var0_1), (mean0_2, var0_2), (mean0, var0)


def multi_results(distance, gan_loss='bce', dis_mlp=False, run_id=0, suffix='', no_graph=False):
    if not suffix and dis_mlp:
        suffix = '_mlp'
    # time.sleep(distance*3)
    nz = 2
    
    distance = (distance + 2) / 2
    if os.path.exists(os.path.join('MOG', '1D', f'{distance}_{gan_loss}{suffix}_{run_id}')):
        pass
    else:
        os.makedirs(os.path.join('MOG', '1D', f'{distance}_{gan_loss}{suffix}_{run_id}'))
    save_path = os.path.join('MOG', '1D', f'{distance}_{gan_loss}{suffix}_{run_id}')

    data1 = torch.randn(128000).cuda()
    data2 = torch.randn(128000).cuda() * 2 + distance
    data3 = torch.randn(128000).cuda() * 3 + distance * 2

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    df1['score_{0}'.format(0)] = data1.cpu().numpy()
    df1['score_{0}'.format(1)] = data2.cpu().numpy()
    df1['score_{0}'.format(2)] = data3.cpu().numpy()
    r_data = torch.cat([data1, data2, data3], dim=0).cpu().numpy()
    df2['score_{0}'.format(2)] = r_data
    np.save(save_path+'/o_data', r_data)

    if not no_graph:
        fig, ax = plt.subplots(1, 1)
        for s in df1.columns:
            df1[s].plot(kind='kde')
        for s in df2.columns:
            df2[s].plot(style='--', kind='kde')
        plt.xlim((-4, 9 + distance * 2))
        ax.legend(["Class_0", "Class_1", "Class_2", "Marginal"])
        # plt.title('Original')
        fig.savefig(save_path + '/original.eps')

    ### fc, tac, ac, hy, proj
    ## train fc
    G = G_guassian(nz=nz, num_classes=3).cuda()
    D = D_guassian(num_classes=3, AC=True, TAC=True, Proj=False).cuda()
    optg = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
    train(data1, data2, data3, nz, G, D, optd, optg, loss_type='fc', gan_loss=gan_loss)
    print('fc training done.')
    res0_fc, res1_fc, res2_fc, resm_fc = evel_model(G, save_path, 'fc', data1, data2, data3, r_data, distance)

    ## train tac
    G = G_guassian(nz=nz, num_classes=3).cuda()
    D = D_guassian(num_classes=3, AC=True, TAC=True, Proj=False).cuda()
    optg = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
    train(data1, data2, data3, nz, G, D, optd, optg, loss_type='tac', gan_loss=gan_loss)
    print('tac training done.')
    res0_tac, res1_tac, res2_tac, resm_tac = evel_model(G, save_path, 'tac', data1, data2, data3, r_data, distance)

    ## train ac
    G = G_guassian(nz=nz, num_classes=3).cuda()
    D = D_guassian(num_classes=3, AC=True, TAC=False, Proj=False).cuda()
    optg = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
    train(data1, data2, data3, nz, G, D, optd, optg, loss_type='ac', gan_loss=gan_loss)
    print('ac training done.')
    res0_ac, res1_ac, res2_ac, resm_ac = evel_model(G, save_path, 'ac', data1, data2, data3, r_data, distance)

    ## train hybrid
    G = G_guassian(nz=nz, num_classes=3).cuda()
    D = D_guassian(num_classes=3, AC=True, TAC=True, Proj=False).cuda()
    optg = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
    train(data1, data2, data3, nz, G, D, optd, optg, loss_type='hybrid', gan_loss=gan_loss)
    print('hybrid training done.')
    res0_hy, res1_hy, res2_hy, resm_hy = evel_model(G, save_path, 'hybrid', data1, data2, data3, r_data, distance)

    ## train proj
    G = G_guassian(nz=nz, num_classes=3).cuda()
    D = D_guassian(num_classes=3, AC=False, TAC=False, Proj=True).cuda()
    optg = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
    train(data1, data2, data3, nz, G, D, optd, optg, loss_type='projection', gan_loss=gan_loss)
    print('fc training done.')
    res0_proj, res1_proj, res2_proj, resm_proj = evel_model(G, save_path, 'projection', data1, data2, data3, r_data, distance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', type=float, help='distance for 1D MoG exp', default=4)
    parser.add_argument('--num_runs', type=int, help='number of runs', default=1)
    parser.add_argument('--gan_loss', type=str, help='gan loss type', default='bce')
    parser.add_argument('--dis_mlp', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no_graph', action='store_true')
    parser.add_argument('--suffix', type=str, help='suffix', default='')
    args = parser.parse_args()
    for i in range(get_start_id(args), args.num_runs):
        multi_results(args.distance, args.gan_loss, args.dis_mlp, i, args.suffix, args.no_graph)
