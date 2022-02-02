import tensorflow as tf
from tensorflow import keras
from keras import layers
from a_nice_mc.utils.hmc import hamiltonian, metropolis_hastings_accept


# class Layer(object):
#     """
#     Base method for implementing flow based models.
#     `forward` and `backward` methods return two values:
#      - the output of the layer
#      - the resulting change of log-determinant of the Jacobian.
#     """
#     def __init__(self):
#         pass

#     def forward(self, inputs):
#         raise NotImplementedError(str(type(self)))

#     def backward(self, inputs):
#         raise NotImplementedError(str(type(self)))


class NiceLayer(object):
    def __init__(self, input_shape, dim, name='nice', swap=False):
        """
        NICE Layer that takes in [x, v] as input and updates one of them.
        Note that for NICE, the Jacobian is always 1; but we keep it for
        possible extensions to real NVP based flow models.
        :param dims: structure of the nice network
        :param name: TensorFlow variable name scope for variable reuse.
        :param swap: Update x if True, or update v if False.
        """
        self.swap = swap
        self.name = 'generator/' + name
        # input = keras.Input(input_shape)
        # intermediate = layers.Dense(dim, activation = 'relu')(input)
        # output = layers.Dense(input_shape)(intermediate)
        # self.nn = keras.Model(input, output)
        self.nn = keras.Sequential()
        self.nn.add(layers.InputLayer(input_shape = input_shape))
        self.nn.add(layers.Dense(dim, activation = 'relu'))
        self.nn.add(layers.Dense(input_shape))

    def forward(self, inputs):
        x, v = inputs
        # x_dim, v_dim = (x.shape)[-1], (v.shape)[-1]
        if self.swap:
            t = self.nn(v)
            x = x + t
        else:
            t = self.nn(x)
            v = v + t
        return [x, v], 0.0

    def backward(self, inputs):
        x, v, = inputs
        # x_dim, v_dim = (x.shape)[-1], (v.shape)[-1]
        if self.swap:
            t = self.nn(v)
            x = x - t
        else:
            t = self.nn(x)
            v = v - t
        return [x, v], 0.0

    # def add(self, x, dx):
    #     for dim in self.dims:
    #         x = dense(x, dim, activation_fn=tf.nn.relu)
    #     x = dense(x, dx)
    #     return x

    # def create_variables(self, x_dim, v_dim):
    #     x = tf.zeros([1, x_dim])
    #     v = tf.zeros([1, v_dim])
    #     _ = self.forward([x, v])


class NiceNetwork(object):
    def __init__(self, x_dim, v_dim):
        self.layers = []
        self.x_dim, self.v_dim = x_dim, v_dim

    def append(self, layer):
        #layer.create_variables(self.x_dim, self.v_dim)
        self.layers.append(layer)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x, _ = layer.forward(x)
        return x
    
    def backward(self, inputs):
        x = inputs
        for layer in reversed(self.layers):
            x, _ = layer.backward(x)
        return x

    def __call__(self, x, is_backward):
        return tf.cond(
            is_backward,
            lambda: self.backward(x),
            lambda: self.forward(x)
        )


class TrainingOperator(object):
    def __init__(self, network):
        self.network = network

    def __call__(self, inputs, steps):

        def fn(zv, x):
            """
            Transition for training, without Metropolis-Hastings.
            `z` is the input state.
            `v` is created as a dummy variable to allow output of v_, for training p(v).
            :param x: variable only for specifying the number of steps
            :return: next state `z_`, and the corresponding auxiliary variable `v_`.
            """
            z, v = zv
            v = tf.random.normal(shape=tf.stack([tf.shape(z)[0], self.network.v_dim]))
            z_, v_ = self.network.forward([z, v])
            return z_, v_

        elems = tf.zeros([steps])
        return tf.scan(fn, elems, inputs)


class InferenceOperator(object):
    def __init__(self, network, energy_fn):
        self.network = network
        self.energy_fn = energy_fn

    def __call__(self, inputs, steps, nice_steps=1):

        def nice_proposal(zv, x):
            """
            Nice Proposal (without Metropolis-Hastings).
            `z` is the input state.
            `v` is created as a dummy variable to allow output of v_, for debugging purposes.
            :param zv:
            :param x:
            :return: next state `z_`, and the corresponding auxiliary variable `v_' (without MH).
            """
            z, v = zv
            z_, v_ = self.network([z, v], is_backward=(x < 0.5)) #(tf.random_uniform([]) < 0.5))
            return z_, v_

        def fn(zv, x):
            """
            Transition with Metropolis-Hastings.
            `z` is the input state.
            `v` is created as a dummy variable to allow output of v_, for debugging purposes.
            :param zv: [z, v]. It is written in this form merely to appeal to Python 3.
            :param x: variable only for specifying the number of steps
            :return: next state `z_`, and the corresponding auxiliary variable `v_`.
            """
            z, v = zv
            v = tf.random.normal(shape=tf.stack([tf.shape(z)[0], self.network.v_dim]))
            #v = tf.random.normal(shape=tf.stack([tf.shape(z)[0], 2]))
            # z_, v_ = self.network([z, v], is_backward=(tf.random_uniform([]) < 0.5))
            z_, v_ = tf.nest.map_structure(tf.stop_gradient, tf.scan(nice_proposal, x * tf.random.uniform([]), (z, v)))
            #print(tf.shape(z_),tf.shape(v_))
            z_, v_ = z_[-1], v_[-1]
            ep = hamiltonian(z, v, self.energy_fn)
            en = hamiltonian(z_, v_, self.energy_fn)
            accept = metropolis_hastings_accept(energy_prev=ep, energy_next=en)
            accept = tf.reshape(accept, [tf.shape(accept)[0], 1]) 
            z_ = tf.where(accept, z_, z)
            return z_, v_

        elems = tf.ones([steps, nice_steps])
        return tf.nest.map_structure(tf.stop_gradient, tf.scan(fn, elems, inputs))
