from a_nice_mc.utils.nice import NiceLayer, NiceNetwork
# from tensorflow import keras
def create_nice_network(x_dim, v_dim, args):
    net = NiceNetwork(x_dim, v_dim)
    for dims, name, swap in args:
        net.append(NiceLayer(x_dim, dims, name, swap))
    # model = keras.Sequential()
    # for layer in net.layers:
    #     model.add(layer.nn)
    # net.model = model
    return net
