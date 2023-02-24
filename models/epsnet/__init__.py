from .MDM import MDMFullDP



def get_model(config):
    if config.network == 'MDMFullDP':
        return MDMFullDP(config)

    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
