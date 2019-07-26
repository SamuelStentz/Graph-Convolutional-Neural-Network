# %load train.py

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
flags.DEFINE_string('dataset', 'wholeset', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_string('graph_conv_dimensions', '[20,20]', 'Number of units in each graph convolution layer.')
flags.DEFINE_string('connected_dimensions','[]', 'Number of units in each FC layer.')
flags.DEFINE_integer('attention_bias', 2, 'Attention Bias.')
flags.DEFINE_integer('attention_dim', 5, 'Attention Dimension.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('save_validation', "False", "If you should save validation accuracy")
flags.DEFINE_string('save_test', "False", "If this is a optimized run! Use all data and save outputs")
flags.DEFINE_string('test_dataset', 'testset', "If we are testing with a unique test_set")
flags.DEFINE_string('balanced_training', 'False', "use a weighted classwise loss to prevent favoring larger class")

# Load data
adj_ls, features, y_arr, sequences, labelorder, train_mask, val_mask, test_mask = parse_many_datasets(FLAGS.dataset)

# Check for independent test_dataset
if FLAGS.test_dataset != "testset":
    adj_ls_test, features_test, y_arr_test, sequences_test, _, _, _, test_mask = parse_many_datasets(FLAGS.test_dataset)
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

# Save with a name defined by model params
model_desc = "lr_{7}_epoch_{8}__gc_{0}_do_{1}_ad_{2}_ab_{3}_fc_{4}_m_{5}_deg_{6}"
model_desc = model_desc.format(FLAGS.graph_conv_dimensions, FLAGS.dropout, FLAGS.attention_dim,
                              FLAGS.attention_bias, FLAGS.connected_dimensions, FLAGS.model, FLAGS.max_degree,
                              FLAGS.learning_rate, FLAGS.epochs)

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
if save_validation == "True": save_validation = True
else: save_validation = False

# see if we are saving the output by considering testing set
save_test = FLAGS.save_test
if save_test == "True": save_test = True
else: save_test = False

if save_test:
    epoch_df = pd.DataFrame(np.zeros(shape = (FLAGS.epochs, 5)))
    labels_df = pd.DataFrame(np.zeros(shape = (sum(test_mask), 5)))
    # add validation to training set for best results
    train_mask = np.array([xi or yi for (xi, yi) in zip(train_mask, val_mask)], dtype = np.bool)
    
# see size of inputs
print("|Training| {}, |Validation| {}, |Testing| {}".format(np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)))

# initial time
ttot = time.time()

# preload support tensor so that it isn't needlessly calculated many times
batch,_,N,M = adj_ls.shape
support_tensor = np.zeros(shape=(batch,num_supports,N,N,M)) # of shape (Batch,Num_Supports,Num_Nodes,Num_Nodes,Num_Edge)
if FLAGS.model == "gcn_cheby":
    print("Calculating Chebyshev polynomials up to order {}...".format(FLAGS.max_degree))
else:
    print("Preprocessing adjacency lists")
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
    feed_dict = construct_feed_dict(features, support, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())
sess.run(model.running_vars_initializer)

# Train model
t = time.time()
cost_ls = []
for epoch in range(FLAGS.epochs):
    t_epoch = time.time()
    
    # instantiate all inputs
    features_train = features[train_mask,:,:]
    support = support_tensor[train_mask,:,:,:]
    y_train = y_arr[train_mask, :]

    # Construct feed dictionary
    feed_dict = construct_feed_dict(features_train, support, y_train, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Reset the counters
    sess.run(model.running_vars_initializer)
    
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Reset the counters
    sess.run(model.running_vars_initializer)
    
    # Validation
    if save_validation:
        cost, acc, duration = evaluate(features, support_tensor, y_arr, val_mask, placeholders, model)
        cost_ls.append(cost)
    
    # Training
    if save_test:
        cost, acc, duration = evaluate(features, support_tensor, y_arr, test_mask, placeholders, model)
        cost_ls.append(cost)
        epoch_df.iloc[epoch, :] = [outs[1], outs[2], cost, acc, time.time() - t_epoch]
    
    # Print results
    if (epoch + 1) % 20 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]),
          "val/test_loss=", "{:.5f}".format(cost), "val/test_acc=", "{:.5f}".format(acc),
          "time=", "{:.5f}".format(time.time() - t))
        t = time.time()

    if epoch > FLAGS.early_stopping and cost_ls[-1] > np.mean(cost_ls[max(-40, -1 * FLAGS.early_stopping):-1]):
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
    os.chdir('..')
    results = os.path.join(os.getcwd(), "Results")
    os.chdir(root)
    txt = os.path.join(results, "validation_results.txt")
    if not os.path.exists(txt):
        with open(txt, "w+") as fh:
            fh.write("Dataset\tTest Dataset\tValidation Accuracy\tEpochs\tModel\tMax Degree\tLearning Rate\tDropout\t")
            fh.write("Attention Dimension\tAttention Bias\tGraph Convolution Dimensions\tFully Connected Dimensions\t")
            fh.write("Balanced Training\tWeight Decay\tEarly Stopping\n")
    vals = [FLAGS.dataset, FLAGS.test_dataset, acc, FLAGS.epochs, FLAGS.model, FLAGS.max_degree, FLAGS.learning_rate, FLAGS.dropout,
            FLAGS.attention_dim, FLAGS.attention_bias, FLAGS.graph_conv_dimensions, FLAGS.connected_dimensions,
            FLAGS.balanced_training, FLAGS.weight_decay, FLAGS.early_stopping]
    with open(txt, "a") as fh:
        string = ""
        for val in vals: string += str(val) + "\t"
        fh.write(string + "\n")
    
# Saving results to a file
if save_test:
    # add column information
    epoch_df.columns = ["train_loss", "train_acc", "test_loss", "test_acc", "time"]
    labels_df.columns = ["Sequence", "Label", "Prediction", "Negative Class Logit", "Positive Class Logit"]
    
    # get test values
    features_test = features[test_mask,:,:]
    support_test = support_tensor[test_mask,:,:,:]
    labels_test = y_arr[test_mask, :]
    
    # add sequences
    labels_df.iloc[:, 0] = [sequences[i] for i in range(len(test_mask)) if test_mask[i]]
    
    # add true labels
    labels_df.iloc[:, 1] = [np.where(labels_test[i])[0] for i in range((sum(test_mask)))]
    
    # get logits in final layer and attention layer values
    feed_dict = construct_feed_dict(features_test, support_test, labels_test, placeholders)
    logits, predictions, attentions = sess.run([model.logits, model.predictions, model.attentions], feed_dict=feed_dict)
    labels_df.iloc[:, 3:5] = logits
    
    # get predictions
    labels_df.iloc[:, 2] = predictions
    
    # add attentions
    att = np.zeros(shape = (attentions.shape[0] * attentions.shape[1], attentions.shape[2]))
    for bat in range(attentions.shape[0]):
        att[bat*attentions.shape[1]:(bat + 1)*attentions.shape[1],:] = attentions[bat,:,:]
    attention_df = pd.DataFrame(att)
    seq_test = [sequences[i] for i in range(len(test_mask)) if test_mask[i]]
    bias_vals = []
    batch_vals = []
    s = []
    for i in range(attentions.shape[0]): bias_vals += list(range(attentions.shape[1]))
    for i in range(attentions.shape[0]): batch_vals += [i for j in range(attentions.shape[1])]
    for i in range(attentions.shape[0]): s += [seq_test[i] for j in range(attentions.shape[1])]
    attention_df["Bias"] = bias_vals
    attention_df["Batch"] = batch_vals
    attention_df["Sequence"] = s
    attention_df["N"] = attentions.shape[2]
    
    # change indices for labels to their names
    labels_df.iloc[:,1] = labels_df.iloc[:,1].map(lambda x: labelorder[x])
    labels_df.iloc[:,2] = labels_df.iloc[:,2].map(lambda x: labelorder[x])
    
    # write to file
    if FLAGS.test_dataset != "testset":
        datadesc = "train_" + FLAGS.dataset + "_test_" + FLAGS.test_dataset
    else:
        datadesc = FLAGS.dataset
    epoch_df.to_csv("../Results/{}.{}.epoch.csv".format(model_desc, datadesc), index = False)
    labels_df.to_csv("../Results/{}.{}.predictions.csv".format(model_desc, datadesc), index = False)
    attention_df.to_csv("../Results/{}.{}.attentions.csv".format(model_desc, datadesc), index = False)
