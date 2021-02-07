# -*- coding: utf-8 -*-


"""
CSSL: Contextualized Self-Supervision for Link Prediction

"""


from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Reshape, Embedding
import tensorflow as tf


class CSSL:
    """
    Implementation of the contextualized self-supervised learning (CSSL) for link prediction with Keras layers.

    Args:
        layer_sizes (list): Hidden feature dimensions for each layer. (Currently, we only support one hidden layer.)
        bias (bool): If True a bias vector is learnt for each layer in the CSSL model, default to False.
        activation (str): The activation function of each layer in the CSSL model, which takes values from "linear", "relu" and "sigmoid"(default).
        normalize ("l2" or None): The normalization used after each layer, default to None.
        input_dim (int): The dimensions of the node features used as input to the model.
        node_num (int): The number of nodes in the given graph.
        walk_length (int): The length of short random walk for sampling context subgraph.
        context_type ("node" or "subgraph"): The type of structural context of CSSL.

    """

    def __init__(
        self,
        layer_sizes,
        bias=False,
        activation="sigmoid",
        normalize=None,
        input_dim=None,
        node_num=None,
        walk_length=None,
        context_type=None,
    ):

        if activation == "linear" or activation == "relu" or activation == "sigmoid":
            self.activation = activation
        else:
            raise ValueError(
                "Activation should be either 'linear', 'relu' or 'sigmoid'; received '{}'".format(
                    activation
                )
            )

        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=-1))

        elif normalize is None:
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError(
                "Normalization should be either 'l2' or None; received '{}'".format(
                    normalize
                )
            )

        self.input_node_num = node_num
        self.input_feature_size = input_dim
        self.walk_length = walk_length

        if context_type == "node" or context_type == "subgraph":
            self.context_type = context_type
        else:
            raise ValueError(
                "context_type should be either 'node' or 'subgraph'; received '{}'".format(
                    context_type
                )
            )

        # Model parameters
        self.n_layers = len(layer_sizes)
        if self.n_layers != 1:
            raise ValueError(
                "Currently only one hidden layer is supported!"
            )
        self.bias = bias

        # Feature dimensions for each layer
        self.dims = [self.input_feature_size] + layer_sizes

        if self.input_feature_size == 0: # map from one-hot representations
            self.emb_layer = Embedding(
                self.input_node_num,
                self.dims[self.n_layers],
                input_length=1,
                name="node_embedding",
            )
        else:
            self.emb_layer = Dense(
                self.dims[self.n_layers],
                activation=self.activation,
                use_bias=self.bias,
                input_shape=(self.input_feature_size,),
                name="node_embedding",
            )

        if self.context_type == "node":
            self.context_embedding = Embedding(
                self.input_node_num,
                self.dims[self.n_layers],
                input_length=1,
                name="context_embedding",
            )
        else:
            self.context_embedding = Embedding(
                self.input_node_num,
                self.dims[self.n_layers],
                input_length=self.walk_length,
                name="context_embedding",
            )

    def _node_model(self):
        """
        Builds a node model for context node prediction for CSSL model.

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (target, context) nodes in the node pairs
            and ``x_out`` is a list of output tensors for (target, context) nodes in the node pairs

        """
        # Expose input and output sockets of the model, for target node:
        # Create tensor inputs
        if self.input_feature_size == 0:
            x_inp_src = Input(shape=(1,))
            assert isinstance(self.emb_layer, Embedding)
            x_out_src = self.emb_layer(x_inp_src)
            x_out_src = Reshape((self.dims[self.n_layers],))(x_out_src)
        else:
            x_inp_src = Input(shape=(self.input_feature_size,))
            x_out_src = self._normalization(self.emb_layer(x_inp_src))

        # Expose input and out sockets of the model, for context node:
        if self.context_type == "node":
            x_inp_dst = Input(shape=(1,))
            assert isinstance(self.context_embedding, Embedding)
            x_out_dst = self.context_embedding(x_inp_dst)
            x_out_dst = Reshape((self.dims[self.n_layers],))(x_out_dst)
        else:
            x_inp_dst = Input(shape=(self.walk_length,))
            assert isinstance(self.context_embedding, Embedding)
            x_out_dst = self.context_embedding(x_inp_dst)
            x_out_dst = tf.reduce_sum(x_out_dst, axis=1)

        x_inp = [x_inp_src, x_inp_dst]
        x_out = [x_out_src, x_out_dst]
        return x_inp, x_out

    def _link_model(self):
        """
        Builds a CSSL model for node link prediction.

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (end1, end2) nodes in the linked node pairs
            and ``x_out`` is a list of output tensors for (end1, end2) nodes in the linked node pairs

        """
        # Expose input and output sockets of the model, for end1 node:
        x_inp_end1, x_out_end1 = self._node_model()

        # Expose input and output sockets of the model, for end2 node:
        x_inp_end2, x_out_end2 = self._node_model()


        x_inp = [x_inp_end1[0], x_inp_end1[1], x_inp_end2[0], x_inp_end2[1]]
        x_out = [x_out_end1[0], x_out_end1[1], x_out_end2[0], x_out_end2[1]]
        return x_inp, x_out

    def in_out_tensors(self, multiplicity=None):
        """
        Builds a CSSL model for link prediction.

        Returns:
            tuple: (x_inp, x_out), where ``x_inp`` is a list of Keras input tensors
            for the specified CSSL model and ``x_out`` contains
            model output tensors

        """
        return self._link_model()
