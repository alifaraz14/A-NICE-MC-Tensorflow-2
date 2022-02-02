import os
import sys

import numpy as np
#sys.path.append("..")
sys.path.append("c:\\Users\\ali21\\Documents\\GitHub\\a-nice-mc")
#sys.path.append(os.getcwd())


def noise_sampler(bs):
    return np.random.normal(0.0, 1.0, [bs, 64])

if __name__ == '__main__':
    from a_nice_mc.objectives.expression.xy_exp_a_nice_mc import XYModel
    from a_nice_mc.models.discriminator import MLPDiscriminator
    from a_nice_mc.models.generator import create_nice_network
    from a_nice_mc.train.wgan_nll import Trainer

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    energy_fn = XYModel(lattice_shape = (8,8), beta = 20, display=True, J=1)
    discriminator = MLPDiscriminator([400, 400, 400])
    generator = create_nice_network(
        64, 64,
        [
            ([400], 'v1', False),
            ([400], 'x1', True),
            ([400], 'v2', False),
        ]
    )

    trainer = Trainer(generator, energy_fn, discriminator, noise_sampler, b=8, m=2)
    trainer.train()
