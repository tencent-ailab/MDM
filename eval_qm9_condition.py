import os
import argparse
import pickle
from random import sample
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict
from statistics import mean
from os.path import join

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.chem import *
import time
from eval import retrieve_qm9_smiles, analyze_stability_for_molecules, PropOptEvaluator, diversity, retrieve_geom_smiles
# from utils.reconstruct import *

from rdkit.Chem.rdmolfiles import  MolToPDBFile
from rdkit.Chem.AllChem import EmbedMolecule

from configs.datasets_config import get_dataset_info
from qm9.models import DistributionNodes
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction.main_qm9_prop import get_model as get_cls_model
from qm9.utils import prepare_context, compute_mean_mad
from utils.sample import *
from utils.reconstruct import *
from utils.datasets import QM93D

from collections import Counter
from torch_geometric.data import DataLoader


def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = get_cls_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier

def padding(data_list, max_nodes):
    padding_list = []
    for data in data_list:
        if data.size(0)<max_nodes:
            padding_data = torch.cat([data,torch.zeros(max_nodes-data.size(0), data.size(1)).to(data.device)],dim=0)
        padding_list.append(padding_data)
    return torch.stack(padding_list)
        
def sample(args, device, model, dataset_info, prop_dist, nodesxsample, context):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1
    
    datas = []
    num_atom = len(dataset_info['atom_decoder'])+1


    for i,n_particles in enumerate(nodesxsample):
        atom_type = torch.randn(n_particles, num_atom)
        pos = torch.randn(n_particles, 3)
        rows, cols = [], []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        rows = torch.LongTensor(rows).unsqueeze(0)
        cols = torch.LongTensor(cols).unsqueeze(0)
        adj = torch.cat([rows, cols], dim=0)
        data = Data(x=atom_type,edge_index=adj,pos=pos)
        datas.append(data)

    batch = Batch.from_data_list(datas).to(device)
    context = context.index_select(0, batch.batch)

    pos_gen, pos_gen_traj, atom_type_list, atom_traj = model.langevin_dynamics_sample(
            atom_type=batch.x,
            pos_init=batch.pos,
            bond_index=batch.edge_index,
            bond_type=None,
            batch=batch.batch,
            num_graphs=batch.num_graphs,
            context=context,
            extend_order=False, # Done in transforms.
            n_steps=args.n_steps,
            step_lr=1e-6, #1e-6
            w_global_pos=args.w_global_pos,
            w_global_node=args.w_global_node,
            w_local_pos=args.w_local_pos,
            w_local_node=args.w_local_node, 
            global_start_sigma=args.global_start_sigma,
            clip=args.clip,
            clip_local=clip_local,
            sampling_type=args.sampling_type,
            eta=args.eta,
            
        )
    pos_list = unbatch(pos_gen, batch.batch)
    atom_list = unbatch(F.one_hot(torch.argmax(atom_type_list[:,:-1], dim=1), 5), batch.batch)
    # for m in range(batch_size):
    #     pos = pos_list[m]
    #     atom_type = atom_list[m]
    #     atom_type = torch.argmax(atom_type, dim=1)
    #     mol = build_molecule(pos, atom_type, dataset_info)
    #     smile = mol2smiles(mol)
    #     print(smile)
    pos_list_pad = padding(pos_list, max_n_nodes)
    atom_list_pad = padding(atom_list, max_n_nodes)
    return pos_list_pad, atom_list_pad, node_mask


class DiffusionDataloader:
    def __init__(self, args, model, nodes_dist, prop_dist, device, dataset_info, unkown_labels=False,
                 batch_size=1, iterations=200):

        self.args = args
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = dataset_info
        self.i = 0

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
 
        pos, atom_type, node_mask = sample(self.args, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)
        
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1).to(self.device)

        context = context.squeeze(1)


        prop_key = self.prop_dist.properties[0]
        if self.unkown_labels:
            context[:] = self.prop_dist.normalizer[prop_key]['mean']
        else:
            context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
        data = {
            'positions': pos.detach(),
            'one_hot': atom_type.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            prop_key: context.detach()
        }
        return data

    def __next__(self):
        if self.i < self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9, geom')
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/outputs/exp_class_alpha_pretrained')
    parser.add_argument('--save_traj', action='store_true', default="/apdcephfs/private_layneyhuang/MDM/logs/model/checkpoint/drugs_default.pt",
                    help='whether store the whole trajectory for sampling')
    parser.add_argument('--debug_break', type=eval, default=False,
                        help='break point or not')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=float('inf'),
                    help='enable global gradients only when noise is low')
    parser.add_argument('--local_start_sigma', type=float, default=float('inf'),
                    help='enable local gradients only when noise is low')
    parser.add_argument('--w_global_pos', type=float, default=1.0,
                    help='weight for global gradients')
    parser.add_argument('--w_local_pos', type=float, default=1.0,
                    help='weight for local gradients')
    parser.add_argument('--w_global_node', type=float, default=1.0,
                    help='weight for global gradients')
    parser.add_argument('--w_local_node', type=float, default=1.0,
                    help='weight for local gradients')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='break point or not')
    parser.add_argument('--iterations', type=int, default=20,
                        help='break point or not')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ld',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    # Load checkpoint
    args.ckpt = 'logs/qm9_full_temb_charge_norm_edmdataset_Gshnet_context_alpha_test_2023_12_29__10_25_33/checkpoints/300.pt'
    args.classifiers_path = 'qm9/property_prediction/outputs/exp_class_alpha/'
    ckpt = torch.load(args.ckpt)
    args.dataset = 'qm9' if 'qm9' in args.ckpt else 'geom'
    config = ckpt['config']
    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    args.sampling_type = 'generalized'
    args.eta=1
    args.global_start_sigma = 0.5 # float('inf')
    # args.local_start_sigma = 1
    args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    args.w_global_pos = 1
    args.w_global_node = 0.5
    args.w_local_pos = 1
    args.w_local_node = 3
    # Logging
    output_dir = get_new_log_dir(log_dir, args.sampling_type+'_test', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    # Model (BDPM)
    logger.info('Building model...')
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])
    print(ckpt['config'].model)
    
    model.eval()

    sa_list = []

    valid = 0
    smile_list = []
    results = []

    clip_local = None
    stable = 0
    logger.info('dataset:%s'%args.dataset)
    # logger.info('sample num:%d'%num_samples)
    logger.info('sample method:%s'%args.sampling_type)
    logger.info('w_global_pos:%.1f'%args.w_global_pos)
    logger.info('w_global_node:%.1f'%args.w_global_node)
    logger.info('w_local_pos:%.1f'%args.w_local_pos)
    logger.info('w_local_node:%.1f'%args.w_local_node)
    show_detail = True
    
    dataset_info = get_dataset_info(args.dataset, False)
    histogram = dataset_info['n_nodes']
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None

    transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])
    train_set = QM93D('train', condition=None, pre_transform=transforms)
    split_idx = train_set.get_half_split(train_set.data.y.size(0), 'qm9_second_half')
    train_set = train_set[split_idx]
    dataloader_train = DataLoader(train_set, 256, shuffle=False)

    args.context = ['alpha']
    property_norms = compute_mean_mad(dataloader_train, args.context, args.dataset)
    mean, mad = property_norms[args.context[0]]['mean'], property_norms[args.context[0]]['mad']

    
    # print(mean)
    # print(mad)

    # exit()


    # conditioning = ['alpha']
    prop_dist = DistributionProperty(dataloader_train, args.context)
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    diffusion_dataloader = DiffusionDataloader(args, model, nodes_dist, prop_dist,
                                                args.device, dataset_info, batch_size=args.batch_size, iterations=args.iterations)
    # for batch in diffusion_dataloader:
    #     print(batch['positions'].size())
    #     print(batch['one_hot'].size())
    classifier = get_classifier(args.classifiers_path).to(args.device)
    print("MDM: We evaluate the classifier on our generated samples")
    loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.context[0], args.device, 1, args.debug_break)
    print("Loss classifier on Generated samples: %.4f" % loss)
    position_list = []
    atom_type_list = []
    mols_dict = []

    

       

        
 


    
    