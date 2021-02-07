#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
import argparse
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import stellargraph as sg
from stellargraph.layer import link_classification, LinkEmbedding
from stellargraph.data.explorer import UniformRandomWalk
from stellargraph.random import random_state
from tensorflow.keras.layers import Dense, Activation, Reshape
from cssl import CSSL

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score


parser = argparse.ArgumentParser(description='context subgraph prediction based CSSL for link prediction')
parser.add_argument('--node-features', default=None, help='node feature matrix')
parser.add_argument('--network-edges', default=None, help='edges in the training network')
parser.add_argument('--train-edges', default=None, help='train edges')
parser.add_argument('--test-edges', default=None, help='test edges')
parser.add_argument('--valid-edges', default=None, help='validation edges')
parser.add_argument('--node-number', type=int, default=None, help='the number of nodes in the training network')
parser.add_argument('--walk-number', type=int, default=10, help='the number of random walks')
parser.add_argument('--walk-length', type=int, default=5, help='the length of random walks')
parser.add_argument('--feature-number', type=int, default=None, help='the dimension of node feature vector')
parser.add_argument('--aggregator', default="l2", help='the aggregator used for constructing edge embedding')
parser.add_argument('--layer-size', type=int, default=128, help='the dimesion of node and edge embedding')
parser.add_argument('--regularizer-value', type=float, default=0.01, help='the l2 regularizer weight for link prediction')
parser.add_argument('--batch-size', type=int, default=20, help='the batch size')
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs for training the model')
parser.add_argument('--pretrain-epochs', type=int, default=40, help='the number of epochs for pretraining')
parser.add_argument('--result-file', default=None, help='the file for saving CSSL link prediction result')
parser.add_argument('--pretrain-result-file', default=None, help='the file for saving the pretrain based CSSL link prediction result')

args = parser.parse_args()

print(f"using aggregator: '{args.aggregator}'")

# Load the training network
edge_data = pd.read_csv(args.network_edges, sep=' ', header=None, names=["source", "target"])
edge_data = edge_data.astype(str)

node_column_names = ["w_{}".format(ii) for ii in range(args.feature_number)]
node_data = pd.read_csv(args.node_features, sep=' ', header=None, names=node_column_names)
node_data.index = node_data.index.astype(str)

graph_train = sg.StellarGraph({"people": node_data}, {"knows": edge_data})

# Load the training edges
train_edges = np.loadtxt(args.train_edges, dtype=int)
train_edges_num = np.shape(train_edges)[0]
examples_train = np.array([[str(train_edges[i,0]),str(train_edges[i,1])] for i in range(train_edges_num)])
labels_train = np.array([train_edges[i,2] for i in range(train_edges_num)])

# Load the validation edges
validation_edges = np.loadtxt(args.valid_edges, dtype=int)
validation_edges_num = np.shape(validation_edges)[0]
examples_validation = np.array([[str(validation_edges[i,0]),str(validation_edges[i,1])] for i in range(validation_edges_num)])
labels_validation = np.array([validation_edges[i,2] for i in range(validation_edges_num)])

# Load the test edges
test_edges = np.loadtxt(args.test_edges, dtype=int)
test_edges_num = np.shape(test_edges)[0]
examples_test = np.array([[str(test_edges[i,0]),str(test_edges[i,1])] for i in range(test_edges_num)])
labels_test = np.array([test_edges[i,2] for i in range(test_edges_num)])

# Train the Proposed Model on the Train Graph
graph_train_node_list = list(graph_train.nodes())
# Create the random walker for sampling context nodes.
random_walker = UniformRandomWalk(graph_train, n=args.walk_number, length=args.walk_length)
# Build a map for sampling context nodes with a referred target node.
_, np_random = random_state(1234)
# Use the sampling distribution as per node2vec
degrees = graph_train.node_degrees()
sampling_distribution = np.array([degrees[n] ** 0.75 for n in graph_train.nodes()])
sampling_distribution_norm = sampling_distribution / np.sum(
    sampling_distribution
)
# start random walks at each node
walks = random_walker.run(nodes=graph_train.nodes())
# first item in each walk is the target/head node
targets = [walk[0] for walk in walks]
# collect the positive (target, context) pairs
positive_pairs = [
    (target, walk[1:], 1)
    for target, walk in zip(targets, walks)
]
# collect the negative (target, context) pairs
negative_samples = np_random.choice(
    list(graph_train.nodes()), size=len(targets)*(args.walk_length-1), p=sampling_distribution_norm
)
negative_samples = list(negative_samples)
negative_pairs = [
    (positive_pairs[i][0], negative_samples[i*(args.walk_length-1):(i+1)*(args.walk_length-1)], 0)
    for i in range(len(positive_pairs))
]
# combine positive and negative (target, context) pairs
pairs = positive_pairs + negative_pairs
# build a map from the target node to its context list
target_context_map = {target:[] for target in graph_train.nodes()}
for pair in pairs:
    target_context_map[pair[0]].append((pair[1],pair[2]))

# Build the contextualized self-supervision based link prediction model.
cssl_lp = CSSL(
    layer_sizes=[args.layer_size],
    input_dim=args.feature_number,
    node_num=args.node_number,
    walk_length=args.walk_length-1,
    context_type="subgraph",
)
# Get the input and output sockets of the CSSL layer.
x_inp, x_out = cssl_lp.in_out_tensors()
# Get the occurrence probability of the context node conditioned on node vi.
pred_end1 = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)([x_out[0],x_out[1]])
# Get the occurrence probability of the context node conditioned no node vj.
pred_end2 = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)([x_out[2],x_out[3]])
# Construct the edge embedding and calculate the link probability between node vi and node vj.
edge_feature = LinkEmbedding(activation="linear", method=args.aggregator)([x_out[0],x_out[2]])
pred_cross = Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(args.regularizer_value))(edge_feature)
pred_cross = Reshape((1,))(pred_cross)
# Build the model and compile it.
model = keras.Model(inputs=x_inp, outputs=[pred_end1, pred_end2, pred_cross])
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=[keras.losses.binary_crossentropy, keras.losses.binary_crossentropy, keras.losses.binary_crossentropy],
    metrics=[keras.metrics.binary_accuracy, keras.metrics.binary_accuracy, keras.metrics.binary_accuracy],
)
# Train the link prediction task and context prediction task jointly.
batches = len(examples_train) // args.batch_size
validation_history = []
models = []
for i in range(args.epochs):
    examples_train_index = np.arange(len(examples_train))
    np.random.shuffle(examples_train_index)
    for j in range(batches):
        index_range = examples_train_index[j*args.batch_size:(j+1)*args.batch_size]
        sampled_train = examples_train[index_range]
        sampled_labels = labels_train[index_range]
        sampled_end1 = [
            (sample[0], random.choice(target_context_map[sample[0]]))
            for sample in sampled_train
        ]
        sampled_end2 = [
            (sample[1], random.choice(target_context_map[sample[1]]))
            for sample in sampled_train
        ]

        sampled_end1_src = np.array([sample[0] for sample in sampled_end1])
        if args.feature_number == 0: # training with only network structure
            sampled_end1_src = np.array([graph_train_node_list.index(node) for node in sampled_end1_src])
        else:
            sampled_end1_src = graph_train.node_features(sampled_end1_src)
        sampled_end1_dst = [sample[1][0] for sample in sampled_end1]
        sampled_end1_dst = np.array([
            [graph_train_node_list.index(node) for node in node_list]
            for node_list in sampled_end1_dst
        ])
        sampled_end1_label = np.array([sample[1][1] for sample in sampled_end1])

        sampled_end2_src = np.array([sample[0] for sample in sampled_end2])
        if args.feature_number == 0: # training with only network structure
            sampled_end2_src = np.array([graph_train_node_list.index(node) for node in sampled_end2_src])
        else:
            sampled_end2_src = graph_train.node_features(sampled_end2_src)
        sampled_end2_dst = [sample[1][0] for sample in sampled_end2]
        sampled_end2_dst = np.array([
            [graph_train_node_list.index(node) for node in node_list]
            for node_list in sampled_end2_dst
        ])
        sampled_end2_label = np.array([sample[1][1] for sample in sampled_end2])

        loss = model.train_on_batch(
            [sampled_end1_src, sampled_end1_dst, sampled_end2_src, sampled_end2_dst],
            [sampled_end1_label, sampled_end2_label, sampled_labels]
        )

        if j >= batches-1 and (i+1) % 10 == 0:
            print(f"Epoch: {i+1}/{args.epochs}")
            print('Loss: %s' % loss)

    pred_model = keras.Model(inputs=[x_inp[0],x_inp[2]], outputs=pred_cross)
    if args.feature_number == 0:
        validation_end1_features = np.array([graph_train_node_list.index(node) for node in examples_validation[:,0]])
        validation_end2_features = np.array([graph_train_node_list.index(node) for node in examples_validation[:,1]])
    else:
        validation_end1_features = graph_train.node_features(examples_validation[:,0])
        validation_end2_features = graph_train.node_features(examples_validation[:,1])
    prediction_validation = pred_model.predict([validation_end1_features,validation_end2_features])
    validation_history.append(roc_auc_score(labels_validation,prediction_validation))
    cur_model = keras.models.clone_model(pred_model)
    cur_model.set_weights(pred_model.get_weights())
    models.append(cur_model)

# find the best model for link prediction
best_model_id = validation_history.index(max(validation_history))
best_pred_model = models[best_model_id]
if args.feature_number == 0:
    test_end1_features = np.array([graph_train_node_list.index(node) for node in examples_test[:,0]])
    test_end2_features = np.array([graph_train_node_list.index(node) for node in examples_test[:,1]])
else:
    test_end1_features = graph_train.node_features(examples_test[:,0])
    test_end2_features = graph_train.node_features(examples_test[:,1])
best_prediction_test = best_pred_model.predict([test_end1_features,test_end2_features])
best_auc_val = roc_auc_score(labels_test,best_prediction_test)
best_ap_score = average_precision_score(labels_test,best_prediction_test)

file_robust_res = open(args.result_file, 'w')
file_robust_res.write("AUC="+str(best_auc_val)+"\n")
file_robust_res.write("AP="+str(best_ap_score)+"\n")
file_robust_res.close()

# Train the link prediction model by pretraining the context prediction task.
cssl_lp_pretrain = CSSL(
    layer_sizes=[args.layer_size],
    input_dim=args.feature_number,
    node_num=args.node_number,
    walk_length=args.walk_length-1,
    context_type="subgraph",
)
# Get the input and output sockets of the CSSL layer.
x_inp_pret, x_out_pret = cssl_lp_pretrain.in_out_tensors()
# Get the occurrence probability of the context node conditioned on node vi.
pred_end1_pret = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)([x_out_pret[0],x_out_pret[1]])
# Get the occurrence probability of the context node conditioned no node vj.
pred_end2_pret = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)([x_out_pret[2],x_out_pret[3]])
# Construct the edge embedding and calculate the link probability between node vi and node vj.
edge_feature_pret = LinkEmbedding(activation="linear", method=args.aggregator)([x_out_pret[0],x_out_pret[2]])
pred_cross_pret = Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(args.regularizer_value))(edge_feature_pret)
pred_cross_pret = Reshape((1,))(pred_cross_pret)
# Build the model and compile it.
model_pret = keras.Model(inputs=x_inp_pret, outputs=[pred_end1_pret, pred_end2_pret])
model_pret.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=[keras.losses.binary_crossentropy, keras.losses.binary_crossentropy],
    metrics=[keras.metrics.binary_accuracy, keras.metrics.binary_accuracy],
)

# Pretrain the context prediction task.
batches = len(examples_train) // args.batch_size
for i in range(args.pretrain_epochs):
    examples_train_index = np.arange(len(examples_train))
    np.random.shuffle(examples_train_index)
    for j in range(batches):
        index_range = examples_train_index[j*args.batch_size:(j+1)*args.batch_size]
        sampled_train = examples_train[index_range]
        sampled_labels = labels_train[index_range]
        sampled_end1 = [
            (sample[0], random.choice(target_context_map[sample[0]]))
            for sample in sampled_train
        ]
        sampled_end2 = [
            (sample[1], random.choice(target_context_map[sample[1]]))
            for sample in sampled_train
        ]

        sampled_end1_src = np.array([sample[0] for sample in sampled_end1])
        if args.feature_number == 0:
            sampled_end1_src = np.array([graph_train_node_list.index(node) for node in sampled_end1_src])
        else:
            sampled_end1_src = graph_train.node_features(sampled_end1_src)
        sampled_end1_dst = [sample[1][0] for sample in sampled_end1]
        sampled_end1_dst = np.array([
            [graph_train_node_list.index(node) for node in node_list]
            for node_list in sampled_end1_dst
        ])
        sampled_end1_label = np.array([sample[1][1] for sample in sampled_end1])

        sampled_end2_src = np.array([sample[0] for sample in sampled_end2])
        if args.feature_number == 0:
            sampled_end2_src = np.array([graph_train_node_list.index(node) for node in sampled_end2_src])
        else:
            sampled_end2_src = graph_train.node_features(sampled_end2_src)
        sampled_end2_dst = [sample[1][0] for sample in sampled_end2]
        sampled_end2_dst = np.array([
            [graph_train_node_list.index(node) for node in node_list]
            for node_list in sampled_end2_dst
        ])
        sampled_end2_label = np.array([sample[1][1] for sample in sampled_end2])

        loss = model_pret.train_on_batch(
            [sampled_end1_src, sampled_end1_dst, sampled_end2_src, sampled_end2_dst],
            [sampled_end1_label, sampled_end2_label]
        )

        if j >= batches-1 and (i+1) % 10 == 0:
            print(f"Epoch: {i+1}/{args.pretrain_epochs}")
            print('Loss: %s' % loss)


model_ft = keras.Model(inputs=[x_inp_pret[0], x_inp_pret[2]], outputs=[pred_cross_pret])
model_ft.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=[keras.losses.binary_crossentropy],
    metrics=[keras.metrics.binary_accuracy],
)
# Train the link prediction model to fine tune parameters
models_ft = []
validation_history_ft = []
for i in range(args.epochs):
    examples_train_index = np.arange(len(examples_train))
    np.random.shuffle(examples_train_index)
    for j in range(batches):
        index_range = examples_train_index[j*args.batch_size:(j+1)*args.batch_size]
        sampled_train = examples_train[index_range]
        sampled_labels = labels_train[index_range]

        sampled_end1 = np.array([sample[0] for sample in sampled_train])
        if args.feature_number == 0:
            sampled_end1 = np.array([graph_train_node_list.index(node) for node in sampled_end1])
        else:
            sampled_end1 = graph_train.node_features(sampled_end1)

        sampled_end2 = np.array([sample[1] for sample in sampled_train])
        if args.feature_number == 0:
            sampled_end2 = np.array([graph_train_node_list.index(node) for node in sampled_end2])
        else:
            sampled_end2 = graph_train.node_features(sampled_end2)

        loss = model_ft.train_on_batch(
            [sampled_end1, sampled_end2],
            [sampled_labels]
        )

        if j >= batches-1 and (i+1) % 10 == 0:
            print(f"Epoch: {i+1}/{args.epochs}")
            print('Loss: %s' % loss)

    pred_model_ft = keras.Model(inputs=[x_inp_pret[0],x_inp_pret[2]], outputs=pred_cross_pret)
    if args.feature_number == 0:
        validation_end1_features = np.array([graph_train_node_list.index(node) for node in examples_validation[:,0]])
        validation_end2_features = np.array([graph_train_node_list.index(node) for node in examples_validation[:,1]])
    else:
        validation_end1_features = graph_train.node_features(examples_validation[:,0])
        validation_end2_features = graph_train.node_features(examples_validation[:,1])
    prediction_validation_ft = pred_model_ft.predict([validation_end1_features,validation_end2_features])
    validation_history_ft.append(roc_auc_score(labels_validation,prediction_validation_ft))
    cur_model = keras.models.clone_model(pred_model_ft)
    cur_model.set_weights(pred_model_ft.get_weights())
    models_ft.append(cur_model)

# find the best model for link prediction
best_model_id_ft = validation_history_ft.index(max(validation_history_ft))
best_pred_model_ft = models_ft[best_model_id_ft]
if args.feature_number == 0:
    test_end1_features = np.array([graph_train_node_list.index(node) for node in examples_test[:,0]])
    test_end2_features = np.array([graph_train_node_list.index(node) for node in examples_test[:,1]])
else:
    test_end1_features = graph_train.node_features(examples_test[:,0])
    test_end2_features = graph_train.node_features(examples_test[:,1])
best_prediction_test_ft = best_pred_model_ft.predict([test_end1_features,test_end2_features])
best_auc_val_ft = roc_auc_score(labels_test,best_prediction_test_ft)
best_ap_score_ft = average_precision_score(labels_test,best_prediction_test_ft)

file_robust_res_ft = open(args.pretrain_result_file, 'w')
file_robust_res_ft.write("AUC="+str(best_auc_val_ft)+"\n")
file_robust_res_ft.write("AP="+str(best_ap_score_ft)+"\n")
file_robust_res_ft.close()
