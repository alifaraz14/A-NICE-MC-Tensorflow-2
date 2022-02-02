import os
import time

import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from a_nice_mc.utils.bootstrap import Buffer
from a_nice_mc.utils.logger import create_logger
from a_nice_mc.utils.nice import TrainingOperator, InferenceOperator
from a_nice_mc.utils.hmc import HamiltonianMonteCarloSampler as HmcSampler


class Trainer(object):
    """
    Trainer for A-NICE-MC.
    - Wasserstein GAN loss with Gradient Penalty for x
    - Cross entropy loss for v

    Maybe for v we can use MMD loss, but in my experiments
    I didn't see too much of an improvement over cross entropy loss.
    """
    def __init__(self,
                 network, energy_fn, discriminator, noise_sampler,
                 b, m, eta=1.0, scale=10.0):
        self.energy_fn = energy_fn
        self.discriminator = discriminator
        self.ns = noise_sampler
        self.ds = None
        self.path = 'logs/' + energy_fn.name
        try:
            os.makedirs(self.path)
        except OSError:
            pass
        self.logger = create_logger(__name__)
        self.train_op = TrainingOperator(network)
        self.infer_op = InferenceOperator(network, energy_fn)
        logits_b = tf.math.log(tf.ones([1, b]))
        logits_m = tf.math.log(tf.ones([1, m]))
        self.b = tf.cast(tf.reshape(tf.random.categorical(logits_b, 1), []), tf.int32) + 1
        self.m = tf.cast(tf.reshape(tf.random.categorical(logits_m, 1), []), tf.int32) + 1
        self.network = network
        self.hmc_sampler = None
        self.x_dim, self.v_dim = network.x_dim, network.v_dim
        #self.x_dim, self.v_dim = 2, 2
        self.eta = eta
        self.scale = scale

    @tf.function
    def inference_op(self, bz, z):
        # Obtain values from inference ops
        # `infer_op` contains Metropolis step
        v = tf.random.normal(tf.stack([bz, self.v_dim]))
        z_, v_ = self.infer_op((z, v), self.steps, self.nice_steps)
        return z_, v_

    @tf.function
    def training_operations_graph(self, x, z):
        # Reshape for pairwise discriminator
        x_dash = tf.reshape(x, [-1, 2 * self.x_dim])
        bx, bz = tf.shape(x)[0], tf.shape(z)[0] 
        # Obtain values from train ops
        v1 = tf.random.normal(tf.stack([bz, self.v_dim]))
        x1_, v1_ = self.train_op((z, v1), self.b)
        x1_ = x1_[-1]
        x1_sg = tf.stop_gradient(x1_)
        v2 = tf.random.normal(tf.stack([bx, self.v_dim]))
        x2_, v2_ = self.train_op((x, v2), self.m)
        x2_ = x2_[-1]
        v3 = tf.random.normal(tf.stack([bx, self.v_dim]))
        x3_, v3_ = self.train_op((x1_sg, v3), self.m)
        x3_ = x3_[-1]

        # The pairwise discriminator has two components:
        # (x, x2) from x -> x2
        # (x1, x3) from z -> x1 -> x3
        #
        # The optimal case is achieved when x1, x2, x3
        # are all from the data distribution
        x_ = tf.concat([
                tf.concat([x2_, x], 1),
                tf.concat([x3_, x1_], 1)
        ], 0)

        # Concat all v values for log-likelihood training
        v1_ = v1_[-1]
        v2_ = v2_[-1]
        v3_ = v3_[-1]
        v_ = tf.concat([v1_, v2_, v3_], 0)
        v_ = tf.reshape(v_, [-1, self.v_dim])

        d = self.discriminator(x_dash)
        d_ = self.discriminator(x_)

        return d, d_, x_, v_

    @tf.function
    def another_graph(self, xl, x_):
        xl = tf.reshape(xl, [-1, 2 * self.x_dim])
        epsilon = tf.random.uniform([], 0.0, 1.0)
        x_hat = xl * epsilon + x_ * (1 - epsilon)
        with tf.GradientTape() as tape:
            tape.watch(x_hat)
            d_hat = self.discriminator(x_hat)
        ddx = tape.gradient(d_hat, x_hat)
        ddx = tf.norm(ddx, axis=1)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.scale)
        return ddx

    @tf.function
    def loss_calculator(self, x, z, xl):
        d, d_, x_, v_ = self.training_operations_graph(x, z)
        # generator loss
        # TODO: MMD loss (http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html)
        # it is easy to implement, but maybe we should wait after this codebase is settled.
        v_loss = tf.reduce_mean(0.5 * tf.multiply(v_, v_))
        g_loss = tf.reduce_mean(d_) + v_loss * self.eta

        # discriminator loss
        d_loss = tf.reduce_mean(d) - tf.reduce_mean(d_)
        ddx = self.another_graph(xl, x_)
        d_loss = d_loss + ddx
        return d_loss, g_loss, v_loss

        # gpu_options = tf.GPUOptions(allow_growth=True)
        # self.sess = tf.Session(config=tf.ConfigProto(
        #     inter_op_parallelism_threads=1,
        #     intra_op_parallelism_threads=1,
        #     gpu_options=gpu_options,
        # ))

    #@tf.function
    def sample(self, steps=2000, nice_steps=1, batch_size=32):
        z = self.ns(batch_size)
        self.steps = steps
        self.nice_steps = nice_steps
        bz = tf.shape(z)[0]

        start = time.time()
        z, v = self.inference_op(bz, z)
        end = time.time()

        self.logger.info('A-NICE-MC: batches [%d] steps [%d : %d] time [%5.4f] samples/s [%5.4f]' %
                         (batch_size, steps, nice_steps, end - start, (batch_size * steps) / (end - start)))
        z = np.transpose(z, [1, 0, 2])
        v = np.transpose(v, [1, 0, 2])
        return z, v

    def bootstrap(self, steps=5000, nice_steps=1, burn_in=1000, batch_size=32,
                  discard_ratio=0.5, use_hmc=False):
        # TODO: it might be better to implement bootstrap in a separate class
        if use_hmc:

            if not self.hmc_sampler:
                self.hmc_sampler = HmcSampler(self.energy_fn,
                                            lambda bs: tf.random.normal([bs, self.x_dim]))

            z = self.hmc_sampler.sample(steps, batch_size)
        else:
            z, _ = self.sample(steps + burn_in, nice_steps, batch_size)
        z = np.reshape(z[:, burn_in:], [-1, z.shape[-1]])
        if self.ds:
            self.ds.discard(ratio=discard_ratio)
            self.ds.insert(z)
        else:
            self.ds = Buffer(z)
    
    @tf.function
    def disc_training_step(self, x, z, xl):
        with tf.GradientTape() as tape:
            d_loss, g_loss, v_loss = self.loss_calculator(x, z, xl)
        #discriminator_variables = tape.watched_variables()[12:]
        #discriminator_variables = self.discriminator.nn.trainable_variables
        discriminator_gradients = tape.gradient(d_loss, self.discriminator.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.nn.trainable_variables))
    
    @tf.function
    def gen_training_step(self, x, z, xl):    
        with tf.GradientTape() as tape:
            d_loss, g_loss, v_loss = self.loss_calculator(x, z, xl)
        #generator_variables = tape.watched_variables()[:12]           
        # generator_gradients = tape.gradient(g_loss, {self.network.layers[0].nn.trainable_variables,
        #                                              self.network.layers[1].nn.trainable_variables,
        #                                              self.network.layers[2].nn.trainable_variables
        #                                             }
        #                                     )           
        # optimizer.apply_gradients(zip(generator_gradients, {self.network.layers[0].nn.trainable_variables,
        #                                                     self.network.layers[1].nn.trainable_variables,
        #                                                     self.network.layers[2].nn.trainable_variables
        #                                                    }  
        #                             )
        #                         )
        generator_gradients = tape.gradient(g_loss, tape.watched_variables()[:12])
        self.optimizer.apply_gradients(zip(generator_gradients, tape.watched_variables()[:12]))
    
    #@tf.function
    def train(self, d_iters = 5,
              epoch_size=500, log_freq=100, max_iters=100000,
              #epoch_size=500, log_freq=5, max_iters=100000,
              bootstrap_steps=5000, bootstrap_burn_in=1000,
              #bootstrap_steps=50, bootstrap_burn_in=10,
              bootstrap_batch_size=32, bootstrap_discard_ratio=0.5,
              evaluate_steps=5000, evaluate_burn_in=1000, evaluate_batch_size=32, nice_steps=1,
              #evaluate_steps=50, evaluate_burn_in=10, evaluate_batch_size=32, nice_steps=1,
              hmc_epochs=1):
        """
        Train the NICE proposal using adversarial training.
        :param d_iters: number of discriminator iterations for each generator iteration
        :param epoch_size: how many iteration for each bootstrap step
        :param log_freq: how many iterations for each log on screen
        :param max_iters: max number of iterations for training
        :param bootstrap_steps: how many steps for each bootstrap
        :param bootstrap_burn_in: how many burn in steps for each bootstrap
        :param bootstrap_batch_size: # of chains for each bootstrap
        :param bootstrap_discard_ratio: ratio for discarding previous samples
        :param evaluate_steps: how many steps to evaluate performance
        :param evaluate_burn_in: how many burn in steps to evaluate performance
        :param evaluate_batch_size: # of chains for evaluating performance
        :param nice_steps: Experimental.
            num of steps for running the nice proposal before MH. For now do not use larger than 1.
        :param hmc_epochs: number of epochs to bootstrap off HMC rather than NICE proposal
        :return:
        """
        # def _feed_dict(bs):
        #     return {self.z: self.ns(bs), self.x: self.ds(bs), self.xl: self.ds(4 * bs)}

        batch_size = 32
        train_time = 0
        num_epochs = 0
        use_hmc = True
        self.optimizer = optimizers.Adam(learning_rate = 5e-4, beta_1 = 0.5, beta_2 = 0.9)
        for t in range(0, max_iters):
            if t % epoch_size == 0:
                num_epochs += 1
                if num_epochs > hmc_epochs:
                    use_hmc = False
                self.bootstrap(
                    steps=bootstrap_steps, burn_in=bootstrap_burn_in,
                    batch_size=bootstrap_batch_size, discard_ratio=bootstrap_discard_ratio,
                    use_hmc=use_hmc
                )
                z, v = self.sample(evaluate_steps + evaluate_burn_in, nice_steps, evaluate_batch_size)
                z, v = z[:, evaluate_burn_in:], v[:, evaluate_burn_in:]
                self.energy_fn.evaluate([z, v], path=self.path)
                # TODO: save model
            if t % log_freq == 0:
                x, z, xl = self.ds(batch_size), self.ns(batch_size), self.ds(4*batch_size)
                d_loss, g_loss, v_loss = self.loss_calculator(x, z, xl)
                self.logger.info('Iter [%d] time [%5.4f] d_loss [%.4f] g_loss [%.4f] v_loss [%.4f]' %
                                 (t, train_time, d_loss, g_loss, v_loss))
            start = time.time()
            for _ in range(d_iters):
                x, z, xl = self.ds(batch_size), self.ns(batch_size), self.ds(4*batch_size)
                self.disc_training_step(x, z, xl)
            x, z, xl = self.ds(batch_size), self.ns(batch_size), self.ds(4*batch_size)
            self.gen_training_step(x, z, xl)
            end = time.time()
            train_time += end - start

    def load(self):
        # TODO: load model
        raise NotImplementedError(str(type(self)))

    def save(self):
        # TODO: save model
        raise NotImplementedError(str(type(self)))
