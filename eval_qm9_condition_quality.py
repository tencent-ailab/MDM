import argparse
import math
from statistics import mean

from eval import retrieve_qm9_smiles, analyze_stability_for_molecules, retrieve_geom_smiles, compute_prop
from torch_geometric.data import DataLoader

from configs.datasets_config import get_dataset_info
from models.epsnet import *
from qm9.utils import compute_mean_mad
from utils.datasets import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample import *
from utils.transforms import *
import faulthandler
faulthandler.enable()

# from utils.reconstruct import *


def RMSD(probe, ref):
    rmsd = 0.0
    # print(amap)
    assert len(probe) == len(ref)
    atom_num = len(probe)
    for i in range(len(probe)):
        posp = probe[i]
        posf = ref[i]
        rmsd += dist_2(posp, posf)
    rmsd = math.sqrt(rmsd / atom_num)
    return rmsd


def dist_2(atoma_xyz, atomb_xyz):
    dis2 = 0.0
    for i, j in zip(atoma_xyz, atomb_xyz):
        dis2 += (i - j) ** 2
    return dis2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='qm9, geom')
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')

    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--context", nargs='+', default=[],
                        help='arguments : homo | lumo | alpha | gap | mu | Cv')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_sdf', type=bool, default=True)
    parser.add_argument('--quality_sampling', type=bool, default=True,
                        help='quality sampling for visualization in Figure 4, else validity sampling for Table 2')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=1000,
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
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='generalized',
                        help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    
    ckpt = torch.load(args.ckpt)
    args.dataset = 'qm9' if 'qm9' in args.ckpt else 'geom'
    config = ckpt['config']
    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    args.global_start_sigma = 0.5  # float('inf')
    # args.local_start_sigma = 1
    args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    args.w_global_pos = 1
    args.w_global_node = 0.5
    args.w_local_pos = 1
    args.w_local_node = 3

    # data_list
    dataset_info = get_dataset_info(args.dataset, False)
    num_samples = args.num_samples
    batch_size = args.batch_size
    data_list, nodesxsample_list = construct_dataset(num_samples, batch_size, dataset_info)
    transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])
    if args.dataset == 'qm9':
        train_set = QM93D('train', pre_transform=transforms)
        split_idx = train_set.get_half_split(train_set.data.y.size(0), 'qm9_second_half')
        train_set = train_set[split_idx]
        val_set = QM93D('valid', pre_transform=transforms)
        val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)
    elif args.dataset == 'geom':
        train_set = Geom(pre_transform=transforms)
    else:
        raise Exception('Wrong dataset name')
    train_loader = inf_iterator(DataLoader(train_set, batch_size, shuffle=True))

    # Logging
    TAG = 'result'
    if num_samples < 10000:
        tag = 'test'
    output_dir = get_new_log_dir(log_dir, args.sampling_type + "_vae_N(1)_" + tag, tag=args.tag)
    logger = get_logger('test')
    logger.info(args)

    # Model
    logger.info('Building model...')
    logger.info(ckpt['config'].model['network'])
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])
    print(ckpt['config'].model)

    model.eval()

    sa_list = []
    valid = 0
    smile_list = []
    results = []
    sum_rmsd = 0

    clip_local = None
    stable = 0
    logger.info('dataset:%s' % args.dataset)
    logger.info('sample num:%d' % num_samples)
    logger.info('sample method:%s' % args.sampling_type)
    logger.info('w_global_pos:%.1f' % args.w_global_pos)
    logger.info('w_global_node:%.1f' % args.w_global_node)
    logger.info('w_local_pos:%.1f' % args.w_local_pos)
    logger.info('w_local_node:%.1f' % args.w_local_node)
    show_detail = False

    position_list = []
    atom_type_list = []
    mols_dict = []

    args.context = ['alpha']
    context_value = ['validity sampling']
    if len(args.context) > 0:
        property_norms = compute_mean_mad(train_set, args.context, args.dataset)
        mean, mad, max_v, min_v = property_norms[args.context[0]]['mean'], property_norms[args.context[0]]['mad'], \
                                  property_norms[args.context[0]]['max'], property_norms[args.context[0]]['min']
        if args.quality_sampling:
            context_value = torch.tensor(np.arange(0.05, 0.38, 0.02))
        else:
            prop_dist = DistributionProperty(train_loader, args.context)
            if prop_dist is not None:
                prop_dist.set_normalizer(property_norms)

    for c in context_value:
        if args.quality_sampling:
            c_norm = (c - mean) / mad
            context = torch.tensor([c_norm] * batch_size).to(args.device)
        else:
            context = prop_dist.sample_batch(nodesxsample_list[n]).to(args.device)
        for n, datas in enumerate(tqdm(data_list)):
            with torch.no_grad():
                start_time = time.time()
                batch = Batch.from_data_list(datas).to(args.device)

                try:
                    pos_gen, pos_gen_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                        atom_type=batch.x,
                        # atom_type = batch.atom_feat_full.float(),
                        pos_init=batch.pos,
                        bond_index=batch.edge_index,
                        bond_type=None,
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                        extend_order=False,  # Done in transforms.
                        n_steps=args.n_steps,
                        step_lr=1e-6,  # 1e-6
                        w_global_pos=args.w_global_pos,
                        w_global_node=args.w_global_node,
                        w_local_pos=args.w_local_pos,
                        w_local_node=args.w_local_node,
                        global_start_sigma=args.global_start_sigma,
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta,
                        context=context

                    )

                    pos_list = unbatch(pos_gen, batch.batch)
                    atom_list = unbatch(atom_type, batch.batch)
                    current_num_samples = (n + 1) * batch_size
                    secs_per_sample = (time.time() - start_time) / current_num_samples
                    print('\t %d/%d Molecules generated at %.2f secs/sample' % (
                        current_num_samples, num_samples, secs_per_sample))
                    for m in range(batch_size):
                        pos = pos_list[m]
                        atom_type = atom_list[m]

                        # charge
                        atom_type = atom_type[:, :-1]
                        charge = atom_type[:, -1]

                        atom_type = torch.argmax(atom_type, dim=1)
                        position_list.append(pos.cpu().detach())
                        atom_type_list.append(atom_type.cpu().detach())

                        a = 0

                        mol = build_molecule(pos, atom_type, dataset_info)
                        smile = mol2smiles(mol)
                        ptable = Chem.GetPeriodicTable()
                        atom_decoder = dataset_info['atom_decoder']
                        atom_type = [atom_decoder[i] for i in atom_type]
                        atom_type = [ptable.GetAtomicNumber(i.capitalize()) for i in atom_type]
                        property = compute_prop(atom_type, pos.cpu().numpy(), args.context[0])
                        print(f'{args.context[0]: {property:.4f}}')         
                        if show_detail:
                            print("generated smile:", smile)
                        result = {'atom_type': atom_type, 'pos': pos, 'smile': smile}
                        results.append(result)
                        if smile is not None:
                            valid += 1
                            stable_flag = False
                            if "." not in smile:
                                stable += 1
                                stable_flag = True

                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                            smile = mol2smiles(largest_mol)
                            smile_list.append(smile)

                            if args.save_sdf:
                                conf = Chem.Conformer(mol.GetNumAtoms())
                                for i in range(mol.GetNumAtoms()):
                                    conf.SetAtomPosition(i, (float(pos[i][0]), float(pos[i][1]), float(pos[i][2])))
                                mol.AddConformer(conf)
                                sdf_dir = './results/conditioned/{}/molecule_{}'.format(args.context[0],
                                                                                        'with_value')
                                if not os.path.exists(sdf_dir):
                                    os.mkdir(sdf_dir)
                                # writer = Chem.SDWriter(os.path.join(sdf_dir, '%s.sdf' % 'full_{}_{}'.format(n,m)))
                                writer = Chem.SDWriter(os.path.join(sdf_dir, '%s.sdf' % 'full_{}_{}_{}'.format(
                                    args.context[0], c.item(), gap)))
                                # print('%s.sdf' % 'full_{}_{}'.format(n,m))
                                writer.write(mol, confId=0)
                                writer.close()
                    break

                except FloatingPointError:
                    clip_local = 10
                    logger.warning('Retrying with local clipping.')
                    raise Exception('Nan in position')

                print('----------------------------')
                # print('diversity:', diversity(smile_list))
                logger.info("The %dth validity:%.4f" % (n + 1, valid / ((n + 1) * batch_size)))
                logger.info("The %dth stable:%.4f" % (n + 1, stable / ((n + 1) * batch_size)))
                logger.info("The %dth Uniq:%.4f" % (n + 1, len(set(smile_list)) / ((n + 1) * batch_size)))

                print('----------------------------')

    validity_dict = analyze_stability_for_molecules(position_list, atom_type_list, dataset_info)
    print(validity_dict)
    print("Final validity:", valid / num_samples)
    print("Final stable:", stable / num_samples)
    print("Final unique:", len(set(smile_list)) / num_samples)
    print(len(set(smile_list)) / valid)
    logger.info("Final validity:%.4f" % (valid / num_samples))
    logger.info("Final stable:%.4f" % (stable / num_samples))
    logger.info("Final unique:%.4f" % (len(set(smile_list)) / num_samples))

    uniq = list(set(smile_list))

    if args.dataset == 'qm9':
        dataset_smile_list = retrieve_qm9_smiles()
    else:
        dataset_smile_list = retrieve_geom_smiles()

    novel = []
    for smile in uniq:
        if smile not in dataset_smile_list:
            novel.append(smile)
            # print(smile)
        # else:
        #     print()
    print(len(novel))
    novelty = len(novel) / len(uniq)
    logger.info("Final novelty:%.4f" % novelty)

    save = True
    if num_samples == 10000:
        save = True
    if save:
        save_path = os.path.join(output_dir, 'samples_all.pkl')
        logger.info('Saving samples to: %s' % save_path)

        save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        with open(save_smile_path, 'wb') as f:
            pickle.dump(smile_list, f)

    # diversity_score = diversity(list(set(smile_list)))
    # logger.info("Final similar score in uniqueness list:%.4f" % (diversity_score))
    # print(diversity_score)

    # import pandas as pd
    # name = ['smiles']
    # smiles = pd.DataFrame(columns=name,data=list(set(smile_list)))
    # smiles.to_csv('./MDM6x4_GEOM_100_smiles_list_bondTure2_531.csv',encoding='utf-8')
