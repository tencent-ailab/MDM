from .egnn import *
from .gin import *
from .schnet import *
from .edge import *
# from .spherenet import *


def get_encoder(config):
    if config.name == 'egnn':
        return EGNNSparseNetwork(
            n_layers=config.layer,
            # feats_dim = config.hidden_channels,
            feats_dim=config.feats_dim,
            edge_attr_dim=config.edge_attr_dim,
            m_dim=config.m_dim,
            soft_edge=config.soft_edges,
            norm_coors=config.norm_coors,
            # soft_edges = True
            aggr='sum'
            # cutoff = config.cutoff,
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)
