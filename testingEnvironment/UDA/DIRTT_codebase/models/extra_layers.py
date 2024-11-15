import tensorflow as tf
import tensorbayes as tb
import numpy as np
from DIRTT_codebase.args import args
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorflow.python.framework import ops


def normalize_perturbation(d, scope=None):
    with tf.name_scope(scope or 'norm_pert'):
        output = tf.nn.l2_normalize(d, axis=list(range(1, len(d.shape))))
    return output


def scale_gradient(x, scale, scope=None, reuse=None):
    with tf.name_scope('scale_grad'):
        output = (1 - scale) * tf.stop_gradient(x) + scale * x
    return output


def noise(x, std, phase, scope=None, reuse=None):
    with tf.name_scope(scope or 'noise'):
        eps = tf.random.normal(tf.shape(x), 0.0, std)
        output = tf.where(phase, x + eps, x)
    return output


def leaky_relu(x, a=0.2, name=None):
    with tf.name_scope(name or 'leaky_relu'):
        return tf.maximum(x, a * x)

def logistic(x, name=None):
    with tf.name_scope(name or 'logistic'):
        return tf.sigmoid(x)


def basic_accuracy(a, b, scope=None):
    with tf.name_scope(scope or 'basic_acc'):
        a = tf.argmax(a, axis=1)
        b = tf.argmax(b, axis=1)
        eq = tf.cast(tf.equal(a, b), dtype=tf.float32)
        output = tf.reduce_mean(eq)
    return output


def perturb_image(x, p, classifier, pert='vat', scope=None):
    with tf.name_scope(scope or 'perturb_image'):
        eps = 1e-6 * normalize_perturbation(tf.random.normal(shape=tf.shape(x)))

        # Predict on randomly perturbed image
        eps_p = classifier(x + eps, phase=True, reuse=True)
        loss = softmax_xent_two(labels=p, logits=eps_p)

        # Based on perturbed image, get direction of greatest error
        eps_adv = tf.gradients(loss, [eps], aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]

        # Use that direction as adversarial perturbation
        eps_adv = normalize_perturbation(eps_adv)
        x_adv = tf.stop_gradient(x + args.radius * eps_adv)

    return x_adv


def vat_loss(x, p, classifier, scope=None):
    with tf.name_scope(scope or 'smoothing_loss'):
        x_adv = perturb_image(x, p, classifier)
        p_adv = classifier(x_adv, phase=True, reuse=True)
        loss = tf.reduce_mean(softmax_xent_two(labels=tf.stop_gradient(p), logits=p_adv))

    return loss