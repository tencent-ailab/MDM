from .dualenc import DualEncoderEpsNetwork
from .dualenc_full import DualEncoderEpsNetwork_full, DualEncoderEpsNetwork_full_split, \
    DualEncoderEpsNetwork_egnn_simple, DualEncoderEpsNetwork_egnn
from .mdm_ddpm import mdm_full_ddpm, mdm_full_ddpm_2loss, mdm_full_ddpm_dist
from .mdm_dp import MDMFullDP
from .mdm_global import mdm_global_ddpm, mdm_global_ddpm_dist
from .mdm_local import mdm_local_ddpm
from ..BDPM_eq import *


def get_model(config):
    if config.network == 'dualenc':
        return DualEncoderEpsNetwork(config)
    if config.network == 'dualenc_full':
        return DualEncoderEpsNetwork_full(config)
    if config.network == 'MDMFullDP':
        return MDMFullDP(config)
    if config.network == 'dualenc_full_split':
        return DualEncoderEpsNetwork_full_split(config)
    if config.network == 'dualenc_egnn_full':
        return DualEncoderEpsNetwork_egnn(config)
    if config.network == 'mdm_full':
        return mdm_full(config)
    if config.network == 'mdm_full_ddpm':
        return mdm_full_ddpm(config)
    if config.network == 'mdm_full_ddpm_dist':
        return mdm_full_ddpm_dist(config)
    if config.network == 'mdm_full_ddpm_2loss':
        return mdm_full_ddpm_2loss(config)
    if config.network == 'mdm_global':
        return mdm_global_ddpm(config)
    if config.network == 'mdm_global_dist':
        return mdm_global_ddpm_dist(config)
    if config.network == 'mdm_local':
        return mdm_local_ddpm(config)
    if config.network == 'dualenc_egnn_simple':
        return DualEncoderEpsNetwork_egnn_simple(config)

    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
