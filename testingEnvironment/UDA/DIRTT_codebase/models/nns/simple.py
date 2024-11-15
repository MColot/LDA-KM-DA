import tensorflow as tf
from DIRTT_codebase.args import args
from DIRTT_codebase.models.extra_layers import leaky_relu, noise, logistic
from tensorbayes.layers import dense, conv2d, batch_norm
from contextlib import contextmanager

@contextmanager
def arg_scope(layer_funcs, **kwargs):
    original_attrs = {}
    for layer_func in layer_funcs:
        original_attrs[layer_func] = {}
        for key, value in kwargs.items():
            if hasattr(layer_func, key):
                original_attrs[layer_func][key] = getattr(layer_func, key)
                setattr(layer_func, key, value)
    yield
    for layer_func, attrs in original_attrs.items():
        for key, value in attrs.items():
            setattr(layer_func, key, value)


def classifier(x, phase, enc_phase=1, trim=0, scope='class', reuse=None, internal_update=False, getter=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            # complex layout
            layout = [
                (dense, (64,), dict(activation=leaky_relu)),
                (dense, (8,), dict(activation=None)),
                (dense, (args.Y,), dict(activation=None))
            ]

            # LDA-KM-DA-like layout
            """layout = [
                (dense, (3,), dict(activation=None)),
                (dense, (args.Y,), dict(activation=logistic))
            ]"""

            if enc_phase:
                start = 0
                end = len(layout) - trim
            else:
                start = len(layout) - trim
                end = len(layout)

            for i in range(start, end):
                with tf.compat.v1.variable_scope('l{:d}'.format(i)):
                    f, f_args, f_kwargs = layout[i]
                    x = f(x, *f_args, **f_kwargs)

    return x

def feature_discriminator(x, phase, C=1, reuse=None):
    with tf.compat.v1.variable_scope('disc/feat', reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu): # Switch to leaky?

            x = dense(x, 100)
            x = dense(x, C, activation=None)

    return x