# %load train.py

import time
import tensorflow as tf
import numpy as np
from utils import *
from models import GCN
import pandas as pd
import os
import random

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
    # test is now indexes of testset
    test_mask = np.concatenate((train_mask, test_mask))
    # make train test split
    train_mask = np.array([not xi for xi in test_mask], dtype = np.bool)
    idx = [i for i in range(sum(train_mask))]
    np.random.shuffle(idx)
    cutoff = int(6*len(idx)/7)
    val_ind = idx[cutoff:]
    train_ind = idx[:cutoff]
    val_mask = np.array([xi in val_ind for xi in range(train_mask.shape[0])], dtype = np.bool)
    train_mask = np.array([xi in train_ind for xi in range(train_mask.shape[0])], dtype = np.bool)

# Save Name Defined by Model Params
model_desc = "lr_{7}_epoch_{8}_stop_{9}_gc_{0}_do_{1}_ad_{2}_ab_{3}_fc_{4}_m_{5}_deg_{6}"
model_desc = model_desc.format(FLAGS.graph_conv_dimensions, FLAGS.dropout, FLAGS.attention_dim,
                              FLAGS.attention_bias, FLAGS.connected_dimensions, FLAGS.model, FLAGS.max_degree,
                              FLAGS.learning_rate, FLAGS.epochs, FLAGS.early_stopping)

# Determine Number of Supports and Assign Model Function
if FLAGS.model == 'gcn':
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Validating
save_validation = FLAGS.save_validation
if save_validation == "True": save_validation = True
else: save_validation = False

# Testing
save_test = FLAGS.save_test
if save_test == "True": save_test = True
else: save_test = False

# Make Dataframes
if save_test:
    epoch_df = pd.DataFrame(np.zeros(shape = (FLAGS.epochs, 5)))
    labels_df = pd.DataFrame(np.zeros(shape = (sum(test_mask), 5)))

# Print Basic Information
print(f"Graph: {FLAGS.dataset}, {FLAGS.test_dataset}\nModel {model_desc}")

# Size of Different Sets
print("|Training| {}, |Validation| {}, |Testing| {}".format(np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)))

# Initial time
ttot = time.time()

# Preload support tensor so that it isn't needlessly calculated many times
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

# Normalize all features
features = preprocess_features(features)

# Test processed inputs
test_inputs(features, support_tensor, y_arr)

# Define placeholders
F = features.shape[2]
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(None,num_supports,N,N,M)), # ?xnum_supportsxNxNxM
    'features': tf.placeholder(tf.float32, shape=(None,N,F)), # ?xNxF
    'labels': tf.placeholder(tf.float32, shape=(None, y_arr.shape[1])), # ?,|labels|
    'dropout': tf.placeholder_with_default(0., shape=())
}

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders, model):
    t_test = time.time()
    features = features[mask,:,:]
    support = support[mask,:,:,:]
    labels = labels[mask, :]
    feed_dict = construct_feed_dict(features, support, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    return outs_val[0], outs_val[1], (time.time() - t_test)

def optimize():
    # Train model
    print("\nOptimization of Stopping Conditions:")
    t = time.time()
    cost_ls = []
    last_improvement = 0
    best_accuracy = 0
    improved_str = ''
    for epoch in range(FLAGS.epochs):
        t_epoch = time.time()
        # Instantiate all inputs
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
        # Validation evaluation
        cost, acc, duration = evaluate(features, support_tensor, y_arr, val_mask, placeholders, model)
        cost_ls.append(cost)
        # Save the model IF validation is sufficiently accurate
        if acc > best_accuracy:
            best_accuracy = acc
            last_improvement = epoch
            saver.save(sess=sess, save_path=save_path_val)
            improved_str += '*'
        # Print results
        if (epoch + 1) % 20 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),"train_acc=", "{:.5f}".format(outs[2]),
                  "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc),"time=", "{:.5f}".format(time.time() - t), improved_str)
            t = time.time()
            improved_str = ''
        if epoch > FLAGS.early_stopping and epoch - last_improvement > 200:
            print("Early stopping...")
            break
    print("Optimization Finished! Total Time: {} sec".format(time.time() - ttot))
    return best_accuracy, epoch

def testing_results(epoch_final):
    # Initialize session
    print("\nTraining on test set:")
    sess.run(tf.global_variables_initializer())
    sess.run(model.running_vars_initializer)
    # Combine training and validation
    mask = np.array([x or y for (x,y) in zip(test_mask, val_mask)], dtype = np.bool)
    # Train model
    t = time.time()
    cost_ls = []
    last_improvement = 0
    best_accuracy = 0
    improved_str = ''
    for epoch in range(FLAGS.epochs):
        t_epoch = time.time()
        # Instantiate all inputs
        features_train = features[mask,:,:]
        support = support_tensor[mask,:,:,:]
        y_train = y_arr[mask, :]
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features_train, support, y_train, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Reset the counters
        sess.run(model.running_vars_initializer)
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        # Reset the counters
        sess.run(model.running_vars_initializer)
        # Evaluate
        cost, acc, duration = evaluate(features, support_tensor, y_arr, test_mask, placeholders, model)
        cost_ls.append(cost)
        epoch_df.iloc[epoch, :] = [outs[1], outs[2], cost, acc, time.time() - t_epoch]
        # Save the model IF training accuracy is a maximum
        if outs[2] > best_accuracy:
            best_accuracy = outs[2]
            last_improvement = epoch
            saver.save(sess=sess, save_path=save_path_test)
            improved_str += '*'
        # Print results
        if (epoch + 1) % 20 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),"train_acc=", "{:.5f}".format(outs[2]),
                  "test_loss=", "{:.5f}".format(cost), "test_acc=", "{:.5f}".format(acc),"time=", "{:.5f}".format(time.time() - t), improved_str)
            improved_str = ''
            t = time.time()
        # Stop training when we hit old epoch number 
        if epoch > epoch_final and epoch - last_improvement > 200:
            print("Early stopping...")
            break
    print("Optimization for Test Finished! Total Time: {} sec".format(time.time() - ttot))
    return best_accuracy

# Create model
model = model_func(placeholders, input_dim=features.shape[2], logging=True)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(model.running_vars_initializer)

# Make saver
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num = random.randint(100000,999999)
save_path_val = os.path.join(save_dir, f'best_validation_{num}')
save_path_val = os.path.join(os.getcwd(), save_path_val)
save_path_test = os.path.join(save_dir, f'best_training_{num}')
save_path_test = os.path.join(os.getcwd(), save_path_test)

# Do optimization on validation set
accuracy_validation, final_epoch = optimize()

# Validation
if save_validation:
    root = os.getcwd()
    os.chdir('..')
    results = os.path.join(os.getcwd(), "Results")
    os.chdir(root)
    txt = os.path.join(results, "validation_results.txt")
    if not os.path.exists(txt):
        with open(txt, "w+") as fh:
            fh.write("Dataset\tTest Dataset\tValidation Accuracy\tMax Epochs\tFinal Epoch\tModel\tMax Degree\tLearning Rate\tDropout\t")
            fh.write("Attention Dimension\tAttention Bias\tGraph Convolution Dimensions\tFully Connected Dimensions\t")
            fh.write("Balanced Training\tWeight Decay\tEarly Stopping\n")
    vals = [FLAGS.dataset, FLAGS.test_dataset, accuracy_validation, FLAGS.epochs,final_epoch, FLAGS.model, FLAGS.max_degree,
            FLAGS.learning_rate, FLAGS.dropout, FLAGS.attention_dim, FLAGS.attention_bias, FLAGS.graph_conv_dimensions,
            FLAGS.connected_dimensions, FLAGS.balanced_training, FLAGS.weight_decay, FLAGS.early_stopping]
    with open(txt, "a") as fh:
        string = ""
        for val in vals: string += str(val) + "\t"
        fh.write(string + "\n")

# Test
if save_test:
    # Train on validation and train set
    accuracy_train = testing_results(final_epoch)
    # Choose which model to use
    if accuracy_validation > accuracy_train:
        path = save_path_val
    else:
        path = save_path_test
    # Load model
    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path=path)
    # Evaluate
    test_cost, test_acc, test_duration = evaluate(features, support_tensor, y_arr, test_mask, placeholders, model)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    # Saving results to a file
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

for file in os.listdir(save_dir):
    if str(num) in file:
        os.remove(os.path.join(save_dir, file))
