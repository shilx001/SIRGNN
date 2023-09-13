import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from sir.sirgnn import *

class SharedTreePolicy:
    def __init__(self, adj_w, adj, state_dim=139, layer=3, branch=32, hidden_size=64, learning_rate=1e-3, seed=1,
                 max_seq_length=32,stddev=0.03, percentage=0.9, batch_size=1, feature_size=64, gnn_layer=1, topK=10):
        self.state_dim = state_dim
        self.layer = layer
        self.branch = branch
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.stddev = stddev
        self.percentage = percentage
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.gnn_layer = gnn_layer
        self.topK = topK
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.input_action = tf.placeholder(dtype=tf.int32)
        self.input_reward = tf.placeholder(dtype=tf.float32)
        self.WSG_in = tf.placeholder(dtype=tf.float32)
        self.WSG_out = tf.placeholder(dtype=tf.float32)
        self.AIG_in = tf.placeholder(dtype=tf.float32)
        self.AIG_out = tf.placeholder(dtype=tf.float32)
        self.items = tf.placeholder(dtype=tf.int32)
        self.item_feature = tf.placeholder(dtype=tf.float32,shape=[None, max_seq_length-1, 128])
        self.alias = tf.placeholder(dtype=tf.int32)
        self.mask = tf.placeholder(dtype=tf.float32)
        self.gnn = SIR_GNN(ASG_w=adj_w, ASG=adj, n_node=self.branch ** self.layer, batch_size=self.batch_size,
                        max_seq_length=max_seq_length,out_size=self.feature_size,step=self.gnn_layer)
        self.output_action_prob = self.forward_pass()
        action_mask = tf.one_hot(self.input_action, self.branch ** self.layer)  # output the action of each node.
        prob_under_policy = tf.reduce_sum(self.output_action_prob * action_mask, axis=1)
        self.loss = -tf.reduce_mean(self.input_reward * tf.log(prob_under_policy + 1e-13), axis=0)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def mlp(self, id=None, softmax_activation=False):
        '''
        Create a multi-layer neural network as tree node.
        :param id: tree node id
        :param reuse: reuse for the networks
        :return: a multi-layer neural network with output dim equals to branch size.
        '''
        with tf.variable_scope('node_' + str(id), reuse=tf.AUTO_REUSE):
            #state = self.gnn.ggnn_v2(self.A_in, self.A_out, self.items, self.alias, self.mask, self.item_feature)
            state = self.gnn.sir_gnn(self.WSG_in, self.WSG_out, self.AIG_in, self.AIG_out, self.items, self.alias, self.mask, self.topK)
            l1 = slim.fully_connected(state, self.hidden_size)
            l2 = slim.fully_connected(l1, self.hidden_size)
            l3 = slim.fully_connected(l2, self.branch)
            if softmax_activation:
                outputs = tf.nn.softmax(l3)
            else:
                outputs = l3
        return outputs  # [N, branch]


    def forward_pass(self):
        '''
        Partial shared layer parameter. Best Performance.
        :return: a tensor of the tree policy.
        '''
        node = [self.mlp(id=str(_), softmax_activation=False) for _ in range(self.layer)]
        root_node = node[0]
        root_node = slim.fully_connected(root_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
        root_output = slim.fully_connected(root_node, num_outputs=self.branch, activation_fn=tf.nn.softmax)
        for i in range(1, self.layer):  # for each layer
            current_output = []
            for j in range(self.branch ** i):  # for each leaf node
                current_node = slim.fully_connected(node[i], num_outputs=self.branch, activation_fn=tf.nn.relu)
                current_node = slim.fully_connected(current_node, num_outputs=self.branch, activation_fn=tf.nn.softmax)
                current_output.append(tf.expand_dims(root_output[:, j], axis=1) * current_node)
            root_output = tf.concat(current_output, axis=1)  # [N, branch**i], update root_output.
        return root_output


    def create_tree(self):
        '''
        Build the tree-structure policy, random shared parameters.
        :return: a list of nodes, each item denotes a layer.
        '''
        # total_nodes = int((self.branch ** self.layer - 1) / (self.branch - 1))
        layer_nodes = []
        shared_nodes = self.mlp(id='shared', softmax_activation=False)
        count = 0
        for i in range(self.layer):
            current_layer = []
            for j in range(int(self.branch ** i)):
                if np.random.rand() < self.percentage:
                    current_layer.append(shared_nodes)
                else:
                    current_layer.append(self.mlp(id='unshared_' + str(count), softmax_activation=False))
                    count += 1
            layer_nodes.append(current_layer)
        return layer_nodes

    def get_action_prob(self, WSG_in, WSG_out, AIG_in, AIG_out, items, alias, mask):
        return self.sess.run(self.output_action_prob, feed_dict={
            self.mask: mask,
            self.WSG_in: WSG_in,
            self.WSG_out: WSG_out,
            self.AIG_in: AIG_in,
            self.AIG_out: AIG_out,
            self.items: items,
            self.alias: alias
        })
    '''
    def get_action_prob_v2(self, A_in, A_out, items, alias, mask, item_feature):
        return self.sess.run(self.output_action_prob, feed_dict={
            self.mask: mask,
            self.A_in: A_in,
            self.A_out: A_out,
            self.items: items,
            self.alias: alias,
            self.item_feature: item_feature,
        })
    '''
    def learn(self, WSG_in, WSG_out, AIG_in, AIG_out, items, alias, mask, input_reward, input_action):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.input_action: input_action,
            self.input_reward: input_reward,
            self.mask: mask,
            self.WSG_in: WSG_in,
            self.WSG_out: WSG_out,
            self.AIG_in: AIG_in,
            self.AIG_out: AIG_out,
            self.items: items,
            self.alias: alias
        })
        return loss
    '''
    def learn_v2(self, A_in, A_out, items, alias, mask, item_feature, input_reward, input_action):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.input_action: input_action,
            self.input_reward: input_reward,
            self.mask: mask,
            self.A_in: A_in,
            self.A_out: A_out,
            self.item_feature: item_feature,
            self.items: items,
            self.alias: alias
        })
        return loss
    '''

    def save_model(self, path):
        '''
        Save the model at desired path.
        :param path: string, the input desired path.
        :return: None
        '''
        self.saver = tf.train.Saver()  # new add
        self.saver.save(self.sess, path + 'model.ckpt')

