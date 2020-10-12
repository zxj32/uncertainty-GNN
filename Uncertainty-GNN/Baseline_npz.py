from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, DPN, EDL
from Load_npz import load_npz_data, load_npz_data_ood_train
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from metrics import masked_accuracy_numpy
from uncertainty_utlis import Misclassification_npz, OOD_Detection_npz

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'amazon_electronics_photo', 'Dataset string.')  # 'amazon_electronics_photo', 'amazon_electronics_computers', "ms_academic_phy"
flags.DEFINE_string('model', 'GCN', 'Model string.')  # 'GCN' "DPN" "EDL" "Drop"
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('OOD_detection', 0, '0 for Misclassification, 1 for OOD detection.')


if FLAGS.OOD_detection == 0:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_npz_data(FLAGS.dataset, seed)
else:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_npz_data_ood_train(FLAGS.dataset, seed + 100)

print(FLAGS.dataset)
# Some preprocessing
features = preprocess_features(features)
support = [preprocess_adj(adj)]
num_supports = 1
if FLAGS.model == 'GCN':
    model_func = GCN
elif FLAGS.model == 'DPN':
    model_func = DPN
elif FLAGS.model == 'EDL':
    model_func = EDL
elif FLAGS.model == 'Drop':
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

checkpt_file = 'data/tem_model/{}_{}.ckpt'.format(FLAGS.dataset, FLAGS.model)
cost_val = []
acc_val = []
val_loss_min = np.inf
patience_step = 0

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break
    if cost <= val_loss_min:
        val_loss_min = min(cost, val_loss_min)
        patience_step = 0
        saver.save(sess, checkpt_file)
    else:
        patience_step += 1
    if patience_step >= FLAGS.early_stopping:
        print("Early stopping...")
        break

print("Optimization Finished!")
saver.restore(sess, checkpt_file)

#### show result
if FLAGS.model != "Drop":
    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    ## save output probability
    feed_dict = construct_feed_dict(features, support, y_test, test_mask, placeholders)
    _, outs, prob = sess.run([model.accuracy, model.outputs, model.predict()], feed_dict=feed_dict)
    if FLAGS.OOD_detection == 0:
        roc, pr = Misclassification_npz(outs, FLAGS.dataset, FLAGS.model)
        if FLAGS.model == "EDL":
            print("Misclassification AUROC: ", "Vacuity = ", roc[0], "Dissonance ", roc[1], "Entropy = ", roc[2])
            print("Misclassification AUPR: ", "Vacuity = ", pr[0], "Dissonance ", pr[1], "Entropy = ", pr[2])
        elif FLAGS.model == "DPN":
            print("Misclassification AUROC: ", "Aleatoric = ", roc[0], "Epistemic ", roc[1], "Entropy = ", roc[2])
            print("Misclassification AUPR: ", "Aleatoric = ", pr[0], "Epistemic ", pr[1], "Entropy = ", pr[2])
        elif FLAGS.model == "GCN":
            print("Misclassification AUROC: ", "Entropy = ", roc[0])
            print("Misclassification AUPR: ", "Entropy = ", pr[0])
            ## save output probability
            np.save("data/output/GCN_{}.npy".format(FLAGS.dataset), prob)
            print('finish the save')
    else:
        roc, pr = OOD_Detection_npz(outs, FLAGS.dataset, FLAGS.model)
        if FLAGS.model == "EDL":
            print("OOD_Detection AUROC: ", "Vacuity = ", roc[0], "Dissonance ", roc[1], "Entropy = ", roc[2])
            print("OOD_Detection AUPR: ", "Vacuity = ", pr[0], "Dissonance ", pr[1], "Entropy = ", pr[2])
        elif FLAGS.model == "DPN":
            print("OOD_Detection AUROC: ", "Aleatoric = ", roc[0], "Epistemic ", roc[1], "Entropy = ", roc[2])
            print("OOD_Detection AUPR: ", "Aleatoric = ", pr[0], "Epistemic ", pr[1], "Entropy = ", pr[2])
        elif FLAGS.model == "GCN":
            print("OOD_Detection AUROC: ", "Entropy = ", roc[0])
            print("OOD_Detection AUPR: ", "Entropy = ", pr[0])
            ## save output probability
            np.save("data/output/GCN_{}_ood.npy".format(FLAGS.dataset), prob)
else:
    # Bayesian Inference MC-Dropout
    Baye_result = []
    for p in range(100):
        feed_dict = construct_feed_dict(features, support, y_test, test_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([model.loss, model.outputs], feed_dict=feed_dict)
        Baye_result.append(outs[1])
    Baye_acc = masked_accuracy_numpy(np.mean(Baye_result, axis=0), y_test, test_mask)
    print("Baye accuracy=", "{:.5f}".format(Baye_acc))
    if FLAGS.OOD_detection == 0:
        roc, pr = Misclassification_npz(Baye_result, FLAGS.dataset, FLAGS.model)
        print("Misclassification AUROC: ", "Aleatoric = ", roc[1], "Epistemic ", roc[2], "Entropy = ", roc[0])
        print("Misclassification AUPR: ", "Aleatoric = ", pr[1], "Epistemic ", pr[2], "Entropy = ", pr[0])
    else:
        roc, pr = OOD_Detection_npz(Baye_result, FLAGS.dataset, FLAGS.model)
        print("OOD_Detection AUROC: ", "Aleatoric = ", roc[1], "Epistemic ", roc[2], "Entropy = ", roc[0])
        print("OOD_Detection AUPR: ", "Aleatoric = ", pr[1], "Epistemic ", pr[2], "Entropy = ", pr[0])