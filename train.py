import argparse
import shutil

import torch.distributed as dist
import torch.utils.tensorboard
import yaml
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm

from configs.datasets_config import get_dataset_info
from models.epsnet import get_model
from qm9.utils import prepare_context, compute_mean_mad
from utils.common import get_optimizer, get_scheduler
from utils.datasets import QM93D, Geom
from utils.misc import *
from utils.transforms import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9, geom')
parser.add_argument('--config', type=str)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--use_mixed_precision', type=bool, default=False)
parser.add_argument('--dp', type=bool, default=True)
parser.add_argument('--resume_iter', type=int, default=None)
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument("--context", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')


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
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    args.dataset = 'qm9' if 'qm9' in args.config else 'geom'

    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    # config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    config_name = '%s_full_ddpm_2losses' % args.dataset  # 'qm9_full_temb_charge_norm_edmdataset' # log name
    seed_all(config.train.seed)

    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        if not os.path.exists(os.path.join(log_dir, 'models')):
            shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)

    # Logging
    
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    shutil.copyfile('./train.py', os.path.join(log_dir, 'train.py'))
    # Datasets and loaders
    logger.info('Loading %s datasets...' % (args.dataset))

    dataset_info = get_dataset_info(args.dataset, remove_h=False)
    transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])
    # train_set = ConformationDataset(config.dataset.train, transform=transforms)

    if args.dataset == 'qm9':
        train_set = QM93D('train', pre_transform=transforms)
        val_set = QM93D('valid', pre_transform=transforms)
        val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)
    elif args.dataset == 'geom':
        train_set = Geom(pre_transform=transforms)
    else:
        raise Exception('Wrong dataset name')
    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True)

    # if context
    # args.context = ['alpha']
    if len(args.context) > 0:
        print(f'Conditioning on {args.context}')
        property_norms = compute_mean_mad(train_set, args.context, args.dataset)
        property_norms_val = compute_mean_mad(val_set, args.context, args.dataset)
    else:
        property_norms = None
        context = None

    # Model
    logger.info('Building model...')
    config.model.context = args.context
    config.model.num_atom = len(dataset_info['atom_decoder']) + 1

    model = get_model(config.model).to(device)

    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)

    start_iter = 0

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

    best_val_loss = float('inf')
    for it in range(start_iter, config.train.max_iters + 1):
        start_time = time.time()
        train(it)
        end_time = (time.time() - start_time)
        print('each iteration requires {} s'.format(end_time))
        avg_val_loss = validate(it)
        if it % config.train.val_freq == 0:
            if avg_val_loss < best_val_loss:
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
                print('Successfully saved the model!')
                best_val_loss = avg_val_loss
