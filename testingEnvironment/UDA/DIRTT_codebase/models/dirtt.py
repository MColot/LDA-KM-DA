import importlib
import tensorflow as tf
import tensorbayes as tb
import numpy as np
from .extra_layers import basic_accuracy, vat_loss
from DIRTT_codebase.args import args
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorbayes.layers import placeholder, constant
from tensorflow.nn import softmax_cross_entropy_with_logits as softmax_xent
from tensorflow.nn import sigmoid_cross_entropy_with_logits as sigmoid_xent

nn = importlib.import_module(f'DIRTT_codebase.models.nns.{args.nn}')


def dirtt(xShape):
    T = tb.utils.TensorDict(dict(
        sess=tf.compat.v1.Session(config=tb.growth_config()),
        src_x=placeholder((None, xShape)), #((None, 32, 32, 3)),
        src_y=placeholder((None, args.Y)),
        trg_x=placeholder((None, xShape)), #((None, 32, 32, 3)),
        trg_y=placeholder((None, args.Y)),
        test_x=placeholder((None, xShape)), #((None, 32, 32, 3)),
        test_y=placeholder((None, args.Y)),
    ))

    # Supervised and conditional entropy minimization
    src_e = nn.classifier(T.src_x, phase=True, enc_phase=1, trim=args.trim)
    trg_e = nn.classifier(T.trg_x, phase=True, enc_phase=1, trim=args.trim, reuse=True, internal_update=True)
    src_p = nn.classifier(src_e, phase=True, enc_phase=0, trim=args.trim)
    trg_p = nn.classifier(trg_e, phase=True, enc_phase=0, trim=args.trim, reuse=True, internal_update=True)

    loss_src_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_p))
    loss_trg_cent = tf.reduce_mean(softmax_xent_two(labels=trg_p, logits=trg_p))

    # Domain confusion
    if args.dw > 0 and args.dirt == 0:
        real_logit = nn.feature_discriminator(src_e, phase=True)
        fake_logit = nn.feature_discriminator(trg_e, phase=True, reuse=True)

        loss_disc = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit))
        loss_domain = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.zeros_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit))

    else:
        loss_disc = constant(0)
        loss_domain = constant(0)

    # Virtual adversarial training (turn off src in non-VADA phase)
    loss_src_vat = vat_loss(T.src_x, src_p, nn.classifier) if args.sw > 0 and args.dirt == 0 else constant(0)
    loss_trg_vat = vat_loss(T.trg_x, trg_p, nn.classifier) if args.tw > 0 else constant(0)

    # Evaluation (EMA)
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    var_class = tf.compat.v1.get_collection('trainable_variables', 'class/')
    ema_op = ema.apply(var_class)
    ema_p = nn.classifier(T.test_x, phase=False, reuse=True, getter=tb.tfutils.get_getter(ema))

    # Teacher model (a back-up of EMA model)
    teacher_p = nn.classifier(T.test_x, phase=False, scope='teacher')
    var_main = tf.compat.v1.get_collection('variables', 'class/(?!.*ExponentialMovingAverage:0)')
    var_teacher = tf.compat.v1.get_collection('variables', 'teacher/(?!.*ExponentialMovingAverage:0)')
    teacher_assign_ops = []
    for t, m in zip(var_teacher, var_main):
        ave = ema.average(m)
        ave = ave if ave is not None else m
        teacher_assign_ops += [tf.compat.v1.assign(t, ave)]
    update_teacher = tf.group(*teacher_assign_ops)
    teacher = tb.function(T.sess, [T.test_x], tf.nn.softmax(teacher_p))

    # Accuracies
    src_acc = basic_accuracy(T.src_y, src_p)
    trg_acc = basic_accuracy(T.trg_y, trg_p)
    ema_acc = basic_accuracy(T.test_y, ema_p)
    fn_ema_acc = tb.function(T.sess, [T.test_x, T.test_y], ema_acc)

    # Optimizer
    dw = constant(args.dw) if args.dirt == 0 else constant(0)
    cw = constant(1) if args.dirt == 0 else constant(args.bw)
    sw = constant(args.sw) if args.dirt == 0 else constant(0)
    tw = constant(args.tw)
    loss_main = (dw * loss_domain +
                 cw * loss_src_class +
                 sw * loss_src_vat +
                 tw * loss_trg_cent +
                 tw * loss_trg_vat)

    var_main = tf.compat.v1.get_collection('trainable_variables', 'class')
    train_main = tf.compat.v1.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)
    train_main = tf.group(train_main, ema_op)

    if args.dw > 0 and args.dirt == 0:
        var_disc = tf.compat.v1.get_collection('trainable_variables', 'disc')
        train_disc = tf.compat.v1.train.AdamOptimizer(args.lr, 0.5).minimize(loss_disc, var_list=var_disc)
    else:
        train_disc = constant(0)

    # Summarizations
    summary_disc = [tf.compat.v1.summary.scalar('domain/loss_disc', loss_disc), ]
    summary_main = [tf.compat.v1.summary.scalar('domain/loss_domain', loss_domain),
                    tf.compat.v1.summary.scalar('class/loss_src_class', loss_src_class),
                    tf.compat.v1.summary.scalar('class/loss_trg_cent', loss_trg_cent),
                    tf.compat.v1.summary.scalar('lipschitz/loss_trg_vat', loss_trg_vat),
                    tf.compat.v1.summary.scalar('lipschitz/loss_src_vat', loss_src_vat),
                    tf.compat.v1.summary.scalar('hyper/dw', dw),
                    tf.compat.v1.summary.scalar('hyper/cw', cw),
                    tf.compat.v1.summary.scalar('hyper/sw', sw),
                    tf.compat.v1.summary.scalar('hyper/tw', tw),
                    tf.compat.v1.summary.scalar('acc/src_acc', src_acc),
                    tf.compat.v1.summary.scalar('acc/trg_acc', trg_acc)]

    # Merge summaries
    summary_disc = tf.compat.v1.summary.merge(summary_disc)
    summary_main = tf.compat.v1.summary.merge(summary_main)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('disc'), loss_disc,
                   c('domain'), loss_domain,
                   c('class'), loss_src_class,
                   c('cent'), loss_trg_cent,
                   c('trg_vat'), loss_trg_vat,
                   c('src_vat'), loss_src_vat,
                   c('src'), src_acc,
                   c('trg'), trg_acc]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_main = [summary_main, train_main]
    T.fn_ema_acc = fn_ema_acc
    T.teacher = teacher
    T.update_teacher = update_teacher

    return T