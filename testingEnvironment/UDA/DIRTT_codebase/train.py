import tensorflow.compat.v1 as tf
import tensorbayes as tb
from .args import args
from .datasets_2 import PseudoData
from .utils import delete_existing, save_value, save_model
import os
import numpy as np

# Disable eager execution (for TensorFlow 1.x compatbility in TensorFlow 2)
tf.disable_eager_execution()

def update_dict(M, feed_dict, src=None, trg=None, bs=100):
    """Update feed_dict with new mini-batch

    M         - (TensorDict) the model
    feed_dict - (dict) tensorflow feed dict
    src       - (obj) source domain. Contains train/test Data obj
    trg       - (obj) target domain. Contains train/test Data obj
    bs        - (int) batch size
    """
    if src:
        src_x, src_y = src.train.next_batch(bs)
        feed_dict.update({M.src_x: src_x, M.src_y: src_y})

    if trg:
        trg_x, trg_y = trg.train.next_batch(bs)
        feed_dict.update({M.trg_x: trg_x, M.trg_y: trg_y})

def train(M, saveDirectory, src=None, trg=None, has_disc=True, iterep=10, n_epoch=20, bs=64, saver=None, args=None):
    """Main training function

    Creates log file, manages datasets, trains model

    M          - (TensorDict) the model
    src        - (obj) source domain. Contains train/test Data obj
    trg        - (obj) target domain. Contains train/test Data obj
    has_disc   - (bool) whether model requires a discriminator update
    saver      - (Saver) saves models during training
    model_name - (str) name of the model being run with relevant parms info
    """
    # Training settings
    itersave = 20000
    epoch = 0
    feed_dict = {}

    # Create a log directory and FileWriter
    """log_dir = os.path.join(args.logdir, model_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)"""

    # Create a save directory
    """if saver:
        model_dir = os.path.join('checkpoints', model_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)"""

    # Replace src domain with pseudolabeled trg
    if args["dirt"] > 0:
        print("Setting backup and updating backup model")
        src = PseudoData(args["trg"], trg, M.teacher)
        M.sess.run(M.update_teacher)

        # Sanity check model
        """print_list = []
        if src:
            save_value(M.fn_ema_acc, 'test/src_test_ema_1k',
                       src.test, train_writer, 0, print_list, full=False)

        if trg:
            save_value(M.fn_ema_acc, 'test/trg_test_ema',
                       trg.test, train_writer, 0, print_list)
            save_value(M.fn_ema_acc, 'test/trg_train_ema_1k',
                       trg.train, train_writer, 0, print_list, full=False)

        print(print_list)"""

    print("Batch size:", bs)
    print("Iterep:", iterep)
    print("Total iterations:", n_epoch * iterep)

    for i in range(n_epoch * iterep):
        # Run discriminator optimizer
        if has_disc:
            update_dict(M, feed_dict, src, trg, bs)
            summary, _ = M.sess.run(M.ops_disc, feed_dict)
            #train_writer.add_summary(summary, i + 1)

        # Run main optimizer
        update_dict(M, feed_dict, src, trg, bs)
        summary, _ = M.sess.run(M.ops_main, feed_dict)
        #train_writer.add_summary(summary, i + 1)
        #train_writer.flush()

        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(epoch, i),
                                            display=args["run"] >= 999)

        # Update pseudolabeler
        if args["dirt"] and (i + 1) % args["dirt"] == 0:
            print("Updating teacher model")
            M.sess.run(M.update_teacher)

        if end_epoch:
            print_list = M.sess.run(M.ops_print, feed_dict)
            print_list += ['epoch', epoch]
            print(print_list)



    # Saving final model
    if saver:
        save_model(saver, M, saveDirectory, 0)

    M.sess.run(M.update_teacher)