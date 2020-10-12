import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    alpha = tf.pow(1.5, preds) + 1.0
    # alpha = tf.exp(preds) + 1.0
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    prob = tf.div(alpha, S)
    loss = -labels * tf.log(prob)
    loss = tf.reduce_sum(loss, axis=1)
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_cross_entropy_relu(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    alpha = preds + 1.0
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    prob = tf.div(alpha, S)
    loss = -labels * tf.log(prob)
    loss = tf.reduce_sum(loss, axis=1)
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_cross_entropy_dirichlet(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    alpha = tf.exp(preds) + 1.0
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    s_digmma = tf.digamma(S)
    loss = labels * (s_digmma - tf.digamma(alpha))
    loss = tf.reduce_sum(loss, axis=1)
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_square_error_edl(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    preds = preds + 1.0
    S = tf.reduce_sum(preds, axis=1, keepdims=True)
    prob = tf.div(preds, S)
    loss = tf.square(prob - labels) + prob * (1 - prob) / (S + 1.0)
    loss = tf.reduce_sum(loss, axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_square_error_dirichlet(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    alpha = tf.exp(preds) + 1.0
    # alpha = tf.pow(1.5, preds) + 1.0
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    prob = tf.div(alpha, S)
    loss = tf.square(prob - labels) + prob * (1 - prob) / (S + 1.0)
    loss = tf.reduce_sum(loss, axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_DPN(preds, labels, mask):

    preds_alpha = tf.exp(preds) + 1.0
    Dir_predict = tfd.Dirichlet(preds_alpha)
    prior_alpha = labels*50 + 1.1
    Dir_ori = tfd.Dirichlet(prior_alpha)

    KL_term = Dir_predict.kl_divergence(Dir_ori)
    # KL_term = Dir_ori.kl_divergence(Dir_predict)

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = KL_term * mask

    return tf.reduce_mean(loss)

def masked_square_error_dirichlet2(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    alpha = preds + 2.0
    # alpha = tf.pow(1.5, preds) + 1.0
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    prob = tf.div(alpha, S)
    loss = tf.square(prob - labels) + prob * (1 - prob) / (S + 1.0)
    loss = tf.reduce_sum(loss, axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_kl_edl(preds, label, labels_num, mask):
    K = labels_num
    alpha = preds * (1.0 - label) + 1.0
    beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha, axis=1, keep_dims=True)
    S_beta = tf.reduce_sum(beta, axis=1, keep_dims=True)
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keep_dims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta), axis=1, keep_dims=True) - tf.lgamma(S_beta)

    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alpha)
    kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keep_dims=True) + lnB + lnB_uni

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = kl * mask
    return tf.reduce_mean(loss)


def masked_kl_teacher(preds, gcn_pred):
    alpha = tf.exp(preds) + 1.0
    S = tf.reduce_sum(alpha, axis=1, keep_dims=True)
    prob = tf.div(alpha, S)
    kl = prob * (tf.log(tf.div(prob, gcn_pred)))
    return tf.reduce_mean(kl)


def masked_kl_prior(preds, prior_alpha, mask):
    preds_alpha = tf.exp(preds) + 1.0
    Dir_predict = tfd.Dirichlet(preds_alpha)
    Dir_ori = tfd.Dirichlet(prior_alpha)

    KL_term = Dir_predict.kl_divergence(Dir_ori)
    # KL_term = Dir_ori.kl_divergence(Dir_predict)

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = KL_term * mask

    return tf.reduce_mean(loss)

def kl_prior(preds, prior_alpha):
    preds_alpha = tf.exp(preds) + 1.0
    Dir_predict = tfd.Dirichlet(preds_alpha)
    Dir_ori = tfd.Dirichlet(prior_alpha)

    KL_term = Dir_predict.kl_divergence(Dir_ori)
    # KL_term = Dir_ori.kl_divergence(Dir_predict)

    loss = KL_term

    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_accuracy_numpy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = np.equal(np.argmax(preds, 1), np.argmax(labels, 1))
    accuracy_all = np.asarray(correct_prediction, np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    mask /= np.mean(mask)
    accuracy_all *= mask
    return np.mean(accuracy_all)


def masked_accuracy_co(correct_prediction, mask):
    """Accuracy with masking."""
    accuracy_all = np.array(correct_prediction, np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    mask /= np.mean(mask)
    accuracy_all *= mask
    return np.mean(accuracy_all)
