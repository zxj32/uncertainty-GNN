import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions

def masked_DPN(preds, labels): # preds is the orginal model output, e.g., logit; label is hot value lable, like (0, 1, 0, 0, 0)

    preds_alpha = tf.exp(preds) + 1.0
    Dir_predict = tfd.Dirichlet(preds_alpha)
    prior_alpha = labels*50 + 1.1  # create the groundtruth based label
    Dir_ori = tfd.Dirichlet(prior_alpha)

    KL_term = Dir_predict.kl_divergence(Dir_ori)

    loss = KL_term

    return tf.reduce_mean(loss)


def dpn_epistemic(result): # result is the orginal model output, e.g., logit
    alpha = np.exp(result) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    p = alpha/S
    term1 = np.log(p) - digamma(alpha + 1) + digamma(S + 1)
    un = p * term1
    un = -np.sum(un, axis=1, keepdims=True)
    return un