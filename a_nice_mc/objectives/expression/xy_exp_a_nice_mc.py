import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from a_nice_mc.objectives.expression import Expression
from a_nice_mc.utils.logger import create_logger


logger = create_logger(__name__)


class XYModel(Expression):
    def __init__(self, lattice_shape, beta, name='XY_model', display=True, J=1):
        super(XYModel, self).__init__(name=name, display=display)
        # self.z = tf.placeholder(tf.float32, [None, 64], name='z')
        self.beta = beta
        #self.rs = np.random.RandomState(seed=random_state)
        #self.L = self.rs.rand(*lattice_shape)
        self.lattice_shape = lattice_shape
        self.d = len(lattice_shape)
        #self.initial_L = self.L.copy()
        #self.t = 0
        self.J = J
        self.H_matrix = tf.Variable(tf.zeros([*lattice_shape]))

    def _calculate_H_matrix(self,z):
        z = tf.reshape(z,self.lattice_shape)
        for i in range(self.lattice_shape[0]):
            for j in range(self.lattice_shape[1]):
                self.H_matrix[i, j].assign(0)
                self.H_matrix[i, j].assign(self.H_matrix[i, j]-tf.math.cos(2 * np.pi * (z[i, j] - z[i, (j + 1) % tf.shape(z)[1]])))
                self.H_matrix[i, j].assign(self.H_matrix[i, j]-tf.math.cos(2 * np.pi * (z[i, j] - z[i, (j - 1) % tf.shape(z)[1]])))
                self.H_matrix[i, j].assign(self.H_matrix[i, j]-tf.math.cos(2 * np.pi * (z[i, j] - z[(i + 1) % tf.shape(z)[0], j])))
                self.H_matrix[i, j].assign(self.H_matrix[i, j]-tf.math.cos(2 * np.pi * (z[i, j] - z[(i - 1) % tf.shape(z)[0], j])))
        self.H_matrix.assign(self.H_matrix*self.J)

    def __call__(self, z):
        """
        with tf.variable_scope(self.name): #Did not understand this statement
            z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
            z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
            v1 = (tf.sqrt(z1 * z1 + z2 * z2) - 1) / 0.2
            v2 = (tf.sqrt(z1 * z1 + z2 * z2) - 2) / 0.2
            v3 = (tf.sqrt(z1 * z1 + z2 * z2) - 3) / 0.2
            v4 = (tf.sqrt(z1 * z1 + z2 * z2) - 4) / 0.2
            v5 = (tf.sqrt(z1 * z1 + z2 * z2) - 5) / 0.2
            p1, p2, p3, p4, p5 = v1 * v1, v2 * v2, v3 * v3, v4 * v4, v5 * v5
            return tf.minimum(tf.minimum(tf.minimum(tf.minimum(p1, p2), p3), p4), p5)
        """
        with tf.variable_scope(self.name):
            self._calculate_H_matrix(z)
            self.H = tf.math.reduce_sum(self.H_matrix) / 2
            return self.beta*self.H

    # @staticmethod
    # def mean():
    #     return np.array([3.6])

    # @staticmethod
    # def std():
    #     return np.array([1.24])

    @staticmethod
    def xlim():
        return [-6, 6]

    @staticmethod
    def ylim():
        return [-6, 6]

    @staticmethod
    def statistics(z):
        z_ = np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True))
        return z_
