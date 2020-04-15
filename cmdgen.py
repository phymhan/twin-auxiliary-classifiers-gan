s = 'share_proj'
no_share = 'false'
use_projection = 'true'
with open(f'{s}_cmd.sh', 'w') as f:
    for m in ['0.9', '0.95', '1.0', '1.05', '1.1']:
        for g in ['0.9', '0.95', '1.0', '1.05', '1.1']:
            name = f'{s}_{m}_{g}.txt'
            cmd = f'CUDA_VISIBLE_DEVICES=5 python my_One_Dimensional_MOG.py --no_share {no_share} --use_projection {use_projection} --lambda_m {m} --lambda_g {g} --name {name}\n'
            f.write(cmd)


s = 'noshare_proj'
no_share = 'true'
use_projection = 'true'
with open(f'{s}_cmd.sh', 'w') as f:
    for m in ['0.9', '0.95', '1.0', '1.05', '1.1']:
        for g in ['0.9', '0.95', '1.0', '1.05', '1.1']:
            name = f'{s}_{m}_{g}.txt'
            cmd = f'CUDA_VISIBLE_DEVICES=6 python my_One_Dimensional_MOG.py --no_share {no_share} --use_projection {use_projection} --lambda_m {m} --lambda_g {g} --name {name}\n'
            f.write(cmd)


s = 'noshare_concat'
no_share = 'true'
use_projection = 'false'
with open(f'{s}_cmd.sh', 'w') as f:
    for m in ['0.9', '0.95', '1.0', '1.05', '1.1']:
        for g in ['0.9', '0.95', '1.0', '1.05', '1.1']:
            name = f'{s}_{m}_{g}.txt'
            cmd = f'CUDA_VISIBLE_DEVICES=7 python my_One_Dimensional_MOG.py --no_share {no_share} --use_projection {use_projection} --lambda_m {m} --lambda_g {g} --name {name}\n'
            f.write(cmd)

