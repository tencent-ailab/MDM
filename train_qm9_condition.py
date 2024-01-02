import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader

from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
from utils.common import get_optimizer, get_scheduler
from utils.datasets import QM93D
from qm9.utils import prepare_context, compute_mean_mad
from configs.datasets_config import get_dataset_info


# ------------------------------------------------------------------------------
# Conditioned training file
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--resume_iter', type=int, default=None)
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--dataset', type=str, default='qm9',
                help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument("--context", nargs='+', default=[],
                help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument("--data_context", type=str, default=None,
                help='arguments : alpha | gap' )


def train(it):
    model.train()
    sum_loss, sum_n = 0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0, 0
    sum_loss_node_global, sum_loss_node_local = 0, 0
    with tqdm(total=len(train_loader), desc='Training') as pbar:
        for batch in train_loader:
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            batch = batch.to(device)
            if len(args.context) > 0:
                context = prepare_context(args.context, batch, property_norms)
            else:
                context = None
            loss_vae_kl = 0.00

            loss = model(
                batch,
                context=context,
                return_unreduced_loss=True
            )

            if config.model.vae_context:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, loss_vae_kl = loss
                loss_vae_kl = loss_vae_kl.mean().item()
            else:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local = loss
            loss = loss.mean()
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer_global.step()
            optimizer_local.step()
            sum_loss += loss.item()
            sum_n += 1
            sum_loss += loss.mean().item()
            sum_loss_pos_global += loss_pos_global.mean().item()
            sum_loss_node_global += loss_node_global.mean().item()
            sum_loss_pos_local += loss_pos_local.mean().item()
            sum_loss_node_local += loss_node_local.mean().item()
            pbar.set_postfix({'loss': '%.2f' % (loss.item())})
            pbar.update(1)

    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / sum_n
    avg_loss_node_global = sum_loss_node_global / sum_n
    avg_loss_pos_local = sum_loss_pos_local / sum_n
    avg_loss_node_local = sum_loss_node_local / sum_n


    logger.info(
        f'[Train] Iter {it:05d} | Loss {loss.item():.2f} | '
        f'Loss(pos_Global) {avg_loss_pos_global:.2f} | Loss(pos_Local) {avg_loss_pos_local:.2f} | '
        f'Loss(node_global) {avg_loss_node_global:.2f} | Loss(node_local) {avg_loss_node_local:.2f} | '
        f'Loss(vae_KL) {loss_vae_kl:.2f} |Grad {orig_grad_norm:.2f} | '
        f'LR {optimizer_global.param_groups[0]["lr"]:.6f}'
    )
    writer.add_scalar('train/loss', avg_loss, it)
    writer.add_scalar('train/loss_pos_global', avg_loss_pos_global, it)
    writer.add_scalar('train/loss_node_global', avg_loss_node_global, it)
    writer.add_scalar('train/loss_pos_local', avg_loss_pos_local, it)
    writer.add_scalar('train/loss_node_local', avg_loss_node_local, it)
    writer.add_scalar('train/loss_vae_KL', loss_vae_kl, it)
    writer.add_scalar('train/lr', optimizer_global.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()


def validate(it):
    sum_loss, sum_n = 0, 0
    sum_loss, sum_n = 0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0, 0
    sum_loss_node_global, sum_loss_node_local = 0, 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)
            if len(args.context) > 0:
                context = prepare_context(args.context, batch, property_norms_val)
            else:
                context = None
            loss = model(
                batch,
                context=context,
                return_unreduced_loss=True
            )

            if config.model.vae_context:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, loss_vae_kl = loss
                loss_vae_kl = loss_vae_kl.mean().item()
            else:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local = loss

            # print(loss)
            sum_loss += loss.sum().item()
            sum_n += loss.size(0)
            sum_loss_pos_global += loss_pos_global.mean().item()
            sum_loss_node_global += loss_node_global.mean().item()
            sum_loss_pos_local += loss_pos_local.mean().item()
            sum_loss_node_local += loss_node_local.mean().item()

    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / sum_n
    avg_loss_node_global = sum_loss_node_global / sum_n
    avg_loss_pos_local = sum_loss_pos_local / sum_n
    avg_loss_node_local = sum_loss_node_local / sum_n

    if config.train.scheduler.type == 'plateau':
        scheduler_global.step(avg_loss_pos_global + avg_loss_node_global)
        scheduler_local.step(avg_loss_pos_local + avg_loss_node_local)
    else:
        scheduler_global.step()
        if 'global' not in config.model.network:
            scheduler_local.step()


    logger.info('[Validate] Iter %05d | Loss %.6f ' % (
        it, avg_loss
    ))
    writer.add_scalar('val/loss', avg_loss, it)
    writer.flush()
    return avg_loss

# ------------------------------------------------------------------------------
# Training file  not in ddp mode
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()

    args.config = 'configs/qm9_full_epoch.yml'
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = 'qm9_full_temb_charge_norm_edmdataset_Gshnet_context_alpha_test'#'qm9_full_temb_charge_norm_edmdataset' 
    seed_all(config.train.seed)

    # if context
    args.context = ['alpha']
    # args.context = []
    # args.data_context = 'alpha'
    
    # Logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    shutil.copyfile('./train_qm9_condition.py', os.path.join(log_dir, 'train_full.py'))
    # Datasets and loaders
    logger.info('Loading datasets...')
    dataset_info = get_dataset_info(args.dataset, remove_h=False)
    transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])
    # train_set = ConformationDataset(config.dataset.train, transform=transforms)
    train_set = QM93D('train', condition=args.data_context, pre_transform=transforms)
    if len(args.context)>0:
        args.dataset = 'qm9_second_half'
        split_idx = train_set.get_half_split(train_set.data.y.size(0), 'qm9_second_half')
        train_set = train_set[split_idx] #half set, follow EDM

    val_set = QM93D('valid', condition=args.data_context, pre_transform=transforms)
    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)


    if len(args.context) > 0:
        print(f'Conditioning on {args.context}')
        property_norms = compute_mean_mad(train_loader, args.context, args.dataset)
        property_norms_val = compute_mean_mad(val_loader, args.context, args.dataset)
    else:
        property_norms = None
        context = None
    # Model
    logger.info('Building model...')
    config.model.context = args.context
    model = get_model(config.model).to(device)

    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    if 'simple' not  in config.model.network:
        optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
        scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)

    start_iter = 1

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        logger.info('Iteration: %d' % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer_global.load_state_dict(ckpt['optimizer_global'])
        optimizer_local.load_state_dict(ckpt['optimizer_local'])
        scheduler_global.load_state_dict(ckpt['scheduler_global'])
        scheduler_local.load_state_dict(ckpt['scheduler_local'])
    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'scheduler_local': scheduler_local.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

