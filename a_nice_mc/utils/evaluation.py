import numpy as np


def auto_correlation_time(x, s, mu, var):
    b, t, d = x.shape
    act_ = np.zeros([d])
    for i in range(0, b):
        y = x[i] - mu
        p, n = y[:-s], y[s:]
        act_ += np.mean(p * n, axis=0) / var
    act_ = act_ / b
    return act_


def effective_sample_size(x, mu, var, logger):
    """
    Calculate the effective sample size of sequence generated by MCMC.
    :param x:
    :param mu: mean of the variable
    :param var: variance of the variable
    :param logger: logg
    :return: effective sample size of the sequence
    Make sure that `mu` and `var` are correct!
    """
    # batch size, time, dimension
    b, t, d = x.shape
    ess_ = np.ones([d])
    for s in range(1, t):
        p = auto_correlation_time(x, s, mu, var)
        if np.sum(p > 0.05) == 0:
            break
        else:
            for j in range(0, d):
                if p[j] > 0.05:
                    ess_[j] += 2.0 * p[j] * (1.0 - float(s) / t)

    logger.info('ESS: max [%f] min [%f] / [%d]' % (t / np.min(ess_), t / np.max(ess_), t))
    return t / ess_


def acceptance_rate(z):
    cnt = z.shape[0] * z.shape[1]
    for i in range(0, z.shape[0]):
        for j in range(1, z.shape[1]):
            if np.min(np.equal(z[i, j - 1], z[i, j])):
                cnt -= 1
    return cnt / float(z.shape[0] * z.shape[1])