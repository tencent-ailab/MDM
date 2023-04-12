import pickle
from torch_geometric.data import DataLoader
from utils.datasets import ConformationDataset
from utils.misc import *
from qm9 import bond_analyze
from utils.reconstruct import *
from pyscf import gto, dft
from scipy.constants import physical_constants
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from configs.datasets_config import get_dataset_info
EH2EV = physical_constants['Hartree energy in eV'][0]


atom_decoder = ['H', 'C', 'N', 'O', 'F']
ptable = Chem.GetPeriodicTable()

def compute_qm9_smiles(dataset_name, remove_h=False):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print("\tConverting QM9 dataset to SMILES ...")

    train_set = ConformationDataset(dataset_name)
    train_iterator = DataLoader(train_set, 1, shuffle=True)
    mols_smiles = []
    for batch in tqdm(train_iterator):
        # print(batch.smiles[0])
        smile = batch.smiles[0]
        mols_smiles.append(smile)

    return mols_smiles

def compute_geom_smiles(file, index, remove_h=False):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print("\tConverting geom dataset to SMILES ...")
    dataset_info = get_dataset_info('geom', remove_h)
    data = np.load(file)
    num_index = np.load(index)

    N= len(num_index)
    sum_nodes = 0

    R_list = data[:,2:]
    z_list = data[:,1]

    mols_smiles = []
    for i in tqdm(range(N)):
        R_i = torch.tensor(R_list[sum_nodes:sum_nodes+num_index[i],:],dtype=torch.float32)
        z_i = z_list[sum_nodes:sum_nodes+num_index[i]]
        atom_type = torch.tensor([dataset_info['atom_index'][int(m)] for m in z_i])
        mol = build_molecule(R_i, atom_type, dataset_info)
        smile = mol2smiles(mol)
        mols_smiles.append(smile)

    return mols_smiles

# data = compute_qm9_smiles('./data/GEOM/QM9/test_data_1k.pkl')
# print(data)
def retrieve_qm9_smiles():
    file_name = '../qm9/temp/qm9_smiles.pickle'
    try:
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        # try:
        #     os.makedirs('./data/GEOM')
        # except:
        #     pass
        qm9_smiles = compute_qm9_smiles('./data/GEOM/QM9/train_data_40k.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(qm9_smiles, f)
        return qm9_smiles

def retrieve_geom_smiles():
    file_name = './data/GEOM/Drugs/geom_smiles.pickle'
    try:
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        # try:
        #     os.makedirs('./data/GEOM')
        # except:
        #     pass
        file = './data/GEOM/Drugs/geom/geom_drugs_1.npy'
        index = './data/GEOM/Drugs/geom/geom_drugs_n_1.npy'
        qm9_smiles = compute_geom_smiles(file, index)
        with open(file_name, 'wb') as f:
            pickle.dump(qm9_smiles, f)
            print('Successfuly converted!')
        return qm9_smiles

# retrieve_qm9_smiles()
def analyze_stability_for_molecules(pos_list, atom_type_list, dataset_info):
    # one_hot = molecule_list['one_hot']
    # x = molecule_list['x']
    # node_mask = molecule_list['node_mask']

    # if isinstance(node_mask, torch.Tensor):
    #     atomsxmol = torch.sum(node_mask, dim=1)
    # else:
    #     atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(pos_list)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        # atom_type = atom_type_list[i].cpu().detach()
        # pos = pos_list[i].cpu().detach()
        atom_type = atom_type_list[i]
        pos = pos_list[i]

        # atom_type = atom_type[0:int(atomsxmol[i])]
        # pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
    }


    return validity_dict

def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                order = bond_analyze.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)

def geom2gap(geom):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = '6-31G(2df,p)' #QM9
    mol.nelectron += mol.nelectron % 2 # Make it full shell? Otherwise the density matrix will be 3D
    mol.build(0, 0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    nocc = mol.nelectron // 2
    homo = mf.mo_energy[nocc - 1] * EH2EV
    lumo = mf.mo_energy[nocc] * EH2EV
    gap = lumo - homo
    return gap


def geom2alpha(geom):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = '6-31G(2df,p)' #QM9
    # mol.basis = '6-31G*' # Kddcup
    mol.nelectron += mol.nelectron % 2 # Make it full shell? Otherwise the density matrix will be 3D
    mol.build(0, 0)

    mf = dft.UKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    # from pynao import tddft_iter

    # td = tddft_iter(mf=gto_mf, gto=mol)
    polar = mf.Polarizability().polarizability()
    xx, yy, zz = polar.diagonal()
    return (xx + yy + zz) / 3


def compute_prop(atomic_number, position, prop_name):
    ptb = Chem.GetPeriodicTable()
    geom = [[ptb.GetElementSymbol(int(z)), position[i]] for i, z in enumerate(atomic_number)]

    if prop_name == 'gap':
        prop = geom2gap(geom)
    elif prop_name == 'alpha':
        prop = geom2alpha(geom)
    
    return prop

class PropOptEvaluator:
    def __init__(self, prop_name='gap', good_threshold=4.5):
        assert prop_name in ['gap', 'alpha']
        self.prop_name = prop_name
        self.good_threshold = good_threshold
    
    def eval(self, mols_dict):
        results = {}
        prop_list = []
        for i in range(len(mols_dict)):
            atom_type, positions = mols_dict[i]['atom_type'], mols_dict[i]['positions']
            mol = build_molecule(positions, atom_type)
            smile = mol2smiles(mol)
            if smile is None:
                continue
            atom_type = [atom_decoder[i] for i in atom_type]
            atom_type = [ptable.GetAtomicNumber(i.capitalize()) for i in atom_type]
            prop_list.append(compute_prop(atom_type, positions.cpu().numpy(), self.prop_name))
        
        mean, median = np.mean(prop_list), np.median(prop_list)
        if self.prop_name == 'gap':
            best = np.min(prop_list)
            good_per = np.sum(np.array(prop_list) <= self.good_threshold) / len(prop_list)
        elif self.prop_name == 'alpha':
            best = np.max(prop_list)
            good_per = np.sum(np.array(prop_list) >= self.good_threshold) / len(prop_list)
        
        results['mean'] = mean
        results['median'] = median
        results['best'] = best
        results['good_per'] = good_per
        return results

def similarity_single(x, y):
    try:
        m_x = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(x.strip()), 2, useChirality=True)
        m_y = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(y.strip()), 2, useChirality=True)
        return DataStructs.TanimotoSimilarity(m_x, m_y)
    except Exception:
        return 0.0


def diversity(smile_list):
    smile_score_list = []
    for i in tqdm(range(len(smile_list)-1)):
        for j in range(i+1,len(smile_list)):
            similar_score = similarity_single(smile_list[i],smile_list[j])
            smile_score_list.append(similar_score)
    return np.mean(smile_score_list)

# print(diversity(['CCC','CCO']))
        
