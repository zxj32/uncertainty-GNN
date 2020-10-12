from __future__ import division
from __future__ import print_function
import time

from utils import *
from models import S_BGCN_T_K
from metrics import *
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
from Load_npz import load_npz_data, load_npz_data_ood_train
from uncertainty_utlis import Misclassification_npz, OOD_Detection_npz


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'amazon_electronics_photo', 'Dataset string.')  # 'amazon_electronics_computers', 'amazon_electronics_photo'
flags.DEFINE_string('model', 'S_BGCN_T_K', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('OOD_detection', 0, '0 for Misclassification, 1 for OOD detection.')

# Load data
if FLAGS.OOD_detection == 0:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_npz_data(FLAGS.dataset, seed+100)
    teacher_probability = np.load("data/output/GCN_{}.npy".format(FLAGS.dataset))
    prior_alpha = np.load("data/prior/all_prior_alpha_{}_sigma_1.npy".format(FLAGS.dataset))
else:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_npz_data_ood_train(FLAGS.dataset, seed+100)
    teacher_probability = np.load("data/ood/GCN_{}_ood.npy".format(FLAGS.dataset))
    prior_alpha = np.load("data/prior/all_prior_alpha_{}_sigma_1_ood.npy".format(FLAGS.dataset))

print("Dataset : ", FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'S_BGCN_T_K':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = S_BGCN_T_K
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'annealing_step': tf.placeholder(tf.float32),
    'gcn_pred': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'prior_alpha': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),

}

# Create model
model = model_func(placeholders, input_dim=features[2][1], label_num=y_train.shape[1],  logging=True)

# Initialize session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, labels, mask, epoch, gcn_pred, prior_alpha, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict_un_teacher_kl(features, support, labels, mask, epoch, gcn_pred, prior_alpha, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)



# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
val_loss_min = np.inf
patience_step = 0
# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict_un_teacher_kl(features, support, y_train, train_mask, epoch, teacher_probability,
                                                  prior_alpha, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, epoch, teacher_probability, prior_alpha,
                                   placeholders)
    cost_val.append(acc)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

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

#  MC-dropout to sample parameter
Baye_result = []
for p in range(100):
    feed_dict = construct_feed_dict_un_teacher_kl(features, support, y_val, val_mask, epoch, teacher_probability,
                                                  prior_alpha,
                                                  placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run([model.loss, model.outputs], feed_dict=feed_dict)
    Baye_result.append(outs[1])
Baye_acc = masked_accuracy_numpy(np.mean(Baye_result, axis=0), y_test, test_mask)
print("Baye accuracy=", "{:.5f}".format(Baye_acc))
if FLAGS.OOD_detection == 0:
    roc, pr = Misclassification_npz(Baye_result, FLAGS.dataset, FLAGS.model)
    print("Misclassification AUROC: ", "Vacuity = ", roc[0], "Dissonance ", roc[1], "Aleatoric = ", roc[2], "Epistemic ", roc[3], "Entropy = ", roc[4])
    print("Misclassification AUPR: ", "Vacuity = ", pr[0], "Dissonance ", pr[1], "Aleatoric = ", pr[2], "Epistemic ", pr[3], "Entropy = ", pr[4])
else:
    roc, pr = OOD_Detection_npz(Baye_result, FLAGS.dataset, FLAGS.model)
    print("OOD_Detection AUROC: ", "Vacuity = ", roc[0], "Dissonance ", roc[1], "Aleatoric = ", roc[2], "Epistemic ", roc[3], "Entropy = ", roc[4])
    print("OOD_Detection AUPR: ", "Vacuity = ", pr[0], "Dissonance ", pr[1], "Aleatoric = ", pr[2], "Epistemic ", pr[3], "Entropy = ", pr[4])





