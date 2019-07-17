# %load train.py
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
from utils import *
from models import GCN
import pandas as pd
import os

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'wholeset', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1 (graph conv layer).')
flags.DEFINE_integer('hidden2', 7, 'Number of units in hidden layer 2 (graph conv layer).')
flags.DEFINE_integer('attention_bias', 2, 'Attention Bias.')
flags.DEFINE_integer('attention_dim', 5, 'Attention Dimension.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('save_validation', "False", "If you should save validation accuracy")
flags.DEFINE_string('save_test', "False", "If this is optimized run! Use all data and save outputs")
flags.DEFINE_integer('k-fold', -1, "If this run is a k-folded run! If given save_val, save_test, etc are all ignored")
flags.DEFINE_string('test_dataset', 'testset', "If we are testing with a unique test_set")


# Load data
adj_ls, features, y_arr, sequences, labelorder, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Check for independent test_dataset
if FLAGS.test_dataset != "testset":
    adj_ls_test, features_test, y_arr_test, sequences_test, _, _, _, test_mask = load_data(FLAGS.test_dataset)
    adj_ls = np.concatenate((adj_ls, adj_ls_test), axis = 0)
    features = np.concatenate((features, features_test), axis = 0)
    y_arr = np.concatenate((y_arr, y_arr_test), axis = 0)
    sequences = sequences + sequences_test
    
    # make all the indices true and false, then concatenate and invert for test and train
    test_mask[0:len(test_mask)] = True
    train_mask[0:len(train_mask)] = False

    test_mask = np.concatenate((train_mask, test_mask))
    train_mask = np.array([not xi for xi in test_mask], dtype = np.bool)
    val_mask = np.array([False for xi in test_mask], dtype = np.bool)
    

#print("Train {}".format(train_mask))
#print("Val {}".format(val_mask))
#print("Test {}\n".format(test_mask))

print("|Training| {}, |Validation| {}, |Testing| {}".format(np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)))

# Save with a name defined by model params
model_desc = "hidden1_{0}_hidden2_{1}_dropout_{2}_attdim_{3}_attbias_{4}_model_{5}_maxdeg_{6}"
model_desc = model_desc.format(FLAGS.hidden1, FLAGS.hidden2, FLAGS.dropout, FLAGS.attention_dim,
                              FLAGS.attention_bias, FLAGS.model, FLAGS.max_degree)

# determine num_supports and make model function
if FLAGS.model == 'gcn':
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# save validation
save_validation = FLAGS.save_validation
if save_validation == "True":
    save_validation = True
else:
    save_validation = False

# see if we are saving the output by considering testing set
save_test = FLAGS.save_test
if save_test == "True":
    save_test = True
else:
    save_test = False

if save_test:
    epoch_df = pd.DataFrame(np.zeros(shape = (FLAGS.epochs, 5)))
    epoch_df.columns = ["train_loss", "train_acc", "val_acc", "val_loss", "time"]
    labels_df = pd.DataFrame(np.zeros(shape = (sum(test_mask), 5)))
    labels_df.columns = ["Sequence", "Label", "Prediction", "Negative Class Logit", "Positive Class Logit"]
    # add validation to training set for best results
    train_mask = np.array([xi or yi for (xi, yi) in zip(train_mask, val_mask)], dtype = np.bool)

# initial time
ttot = time.time()

# preload support tensor so that it isn't needlessly calculated many times
batch,_,N,M = adj_ls.shape
support_tensor = np.zeros(shape=(batch,num_supports,N,N,M)) # of shape (Batch,Num_Supports,Num_Nodes,Num_Nodes,Num_Edge)
print("Calculating Chebyshev polynomials up to order {}...".format(FLAGS.max_degree))
for b in range(batch):
    adj = adj_ls[b]
    for m in range(M):
        adj = adj_ls[b][:,:,m] # first adjacency list
        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, FLAGS.max_degree)
        # add NxN matrices along the num_supports dimension
        sup = np.stack(support, axis=0)
        # add num_supportsxNxN to support tensor
        support_tensor[b,:,:,:,m] = sup

# normalize all features
features = preprocess_features(features)

# Define placeholders
F = features.shape[2]
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(None,num_supports,N,N,M)), # ?xnum_supportsxNxNxM
    'features': tf.placeholder(tf.float32, shape=(None,N,F)), # ?xNxF
    'labels': tf.placeholder(tf.float32, shape=(None, y_arr.shape[1])), # ?,|labels|
    'dropout': tf.placeholder_with_default(0., shape=()),
}

# Create model
model = model_func(placeholders, input_dim=features.shape[2], logging=True)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders, model):
    t_test = time.time()
    features = features[mask,:,:]
    support = support[mask,:,:,:]
    labels = labels[mask, :]
    feed_dict_val = construct_feed_dict(features, support, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())
sess.run(model.running_vars_initializer)

# Train model
t = time.time()
cost_val = []
acc_val = []
for epoch in range(FLAGS.epochs):
    t_epoch = time.time()
    
    # instantiate all inputs
    features_train = features[test_mask,:,:]
    support = support_tensor[test_mask,:,:,:]
    y_test = y_arr[test_mask, :]

    # Construct feed dictionary
    feed_dict = construct_feed_dict(features_train, support, y_test, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Reset the counters
    sess.run(model.running_vars_initializer)
    
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Reset the counters
    sess.run(model.running_vars_initializer)
    
    # Validation
    cost, acc, duration = evaluate(features, support_tensor, y_arr, val_mask, placeholders, model)
    cost_val.append(cost)
    
    # save training progression
    if save_test:
        epoch_df.iloc[epoch, :] = [outs[1], outs[2], cost, acc, time.time() - t_epoch]
    
    # Print results
    if (epoch + 1) % 20 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
        t = time.time()

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished! Total Time: {} sec".format(time.time() - ttot))

# Reset the counters
sess.run(model.running_vars_initializer)

# Testing
test_cost, test_acc, test_duration = evaluate(features, support_tensor, y_arr, test_mask, placeholders, model)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# save a text file whose name is the model_desc, graph_desc (from dataset), and has this info within it  
if save_validation:
    root = os.getcwd()
    optimization = os.path.join(root, "Optimization")
    validation_accuracy = acc
    filename = "{0:.5f}.{1}.{2}.outcome.txt".format(validation_accuracy, model_desc, FLAGS.dataset)
    filename = os.path.join(optimization, filename)
    with open(filename, "w") as fh:
        fh.write("{}\n{}\n{}".format(validation_accuracy, model_desc, FLAGS.dataset))
    
# Saving results to a file
if save_test:
    # get test values
    features_test = features[test_mask,:,:]
    support_test = support_tensor[test_mask,:,:,:]
    labels_test = y_arr[test_mask, :]
    
    # add sequences
    labels_df.iloc[:, 0] = [sequences[i] for i in range(len(test_mask)) if test_mask[i]]
    
    # add true labels
    labels_df.iloc[:, 1] = [np.where(labels_test[i])[0] for i in range((sum(test_mask)))]
    
    # get logits in final layer
    feed_dict_val = construct_feed_dict(features_test, support_test, labels_test, placeholders)
    logits, predictions = sess.run([model.logits, model.predictions], feed_dict=feed_dict_val)
    labels_df.iloc[:, 3:5] = logits
    
    # get predictions
    labels_df.iloc[:, 2] = predictions
    
    # change indices for labels to their names
    labels_df.iloc[:,1] = labels_df.iloc[:,1].map(lambda x: labelorder[x])
    labels_df.iloc[:,2] = labels_df.iloc[:,2].map(lambda x: labelorder[x])
    
    # write to file
    epoch_df.to_csv("../Results/{}.{}.epoch.csv".format(model_desc, FLAGS.dataset), index = False)
    labels_df.to_csv("../Results/{}.{}.predictions.csv".format(model_desc, FLAGS.dataset), index = False)
