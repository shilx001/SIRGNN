import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

class SIR_GNN:
    def __init__(self, ASG_w, ASG, batch_size=8, max_seq_length=32, out_size=64, step=1,n_node=None, l2=None):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.out_size = out_size
        self.stdv = 1.0 / math.sqrt(self.out_size)
        self.embedding = tf.get_variable(shape=[n_node, out_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.ASG_w = tf.constant(ASG_w, dtype=tf.float32)
        self.ASG = tf.constant(ASG, dtype=tf.int32)
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())

        self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                 initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.q1 = tf.get_variable('q1', [self.out_size, 1], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))


    def gat(self,ASG, ASG_w, item, topK):

        neighbor = tf.gather_nd(ASG, tf.reshape(item, [self.batch_size, -1, 1]))    # [batch_size,max_seq_length,topK]
        neighbor_index = tf.reshape(tf.tile(item,[1,topK]),[self.batch_size,self.max_seq_length,topK,1])  # [batch_size,max_seq_length,topK,1]
        neighbor_index = tf.concat([neighbor_index,tf.reshape(neighbor,[self.batch_size,self.max_seq_length,topK,1])],axis=3)   # [batch_size,max_seq_length,topK,2]
        neighbor_w = tf.reshape(tf.gather_nd(ASG_w,neighbor_index),[self.batch_size,self.max_seq_length,topK,1]) # [batch_size,max_seq_length,topK,1]

        n_state = tf.nn.embedding_lookup(self.embedding, neighbor)  # [batch_size,max_seq_length,topK,out_size]
        s_state = tf.nn.embedding_lookup(self.embedding, item)  # [batch_size,max_seq_length,out_size]
        sess_state = tf.reduce_mean(s_state, axis=1)    # [batch_size,out_size]

        n_att = tf.reshape(sess_state,[self.batch_size, 1, 1, -1]) * n_state  # [batch_size,max_seq_length,topK,out_size]
        # n_att = tf.concat([n_att,neighbor_w],axis=3)
        n_att = slim.fully_connected(n_att, self.out_size, activation_fn=tf.nn.leaky_relu)
        n_att = tf.reshape(tf.matmul(tf.reshape(n_att,[-1,self.out_size]), self.q1), [self.batch_size, self.max_seq_length, topK])    # [batch_size,max_seq_length,topK]
        n_att = tf.contrib.layers.softmax(n_att)    # [batch_size,max_seq_length,topK]

        global_state = tf.reshape(n_att,[self.batch_size,self.max_seq_length,-1,1]) * n_state    # [batch_size,max_seq_length,topK,out_size]
        global_state = tf.reduce_sum(global_state, 2)   # [batch_size,max_seq_length,out_size]
        global_state = tf.concat([global_state, s_state],axis=2)
        global_state = slim.fully_connected(global_state,self.out_size)
        return global_state

    def ggnn(self, adj_in, adj_out, item):
        local_state = tf.nn.embedding_lookup(self.embedding, item)
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                local_state_in = tf.reshape(tf.matmul(tf.reshape(local_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                local_state_out = tf.reshape(tf.matmul(tf.reshape(local_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(adj_in, local_state_in), tf.matmul(adj_out, local_state_out)], axis=-1)
                state_output, local_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2 * self.out_size]), axis=1),
                                      initial_state=tf.reshape(local_state, [-1, self.out_size]))
                local_state = tf.reshape(local_state, [self.batch_size, -1, self.out_size])
        return local_state

    def sir_gnn(self, WSG_in, WSG_out, AIG_in, AIG_out, item, alias, mask, topK):
        item = tf.reshape(item, [self.batch_size, -1])
        alias = tf.reshape(alias, [self.batch_size, -1])
        mask = tf.reshape(mask, [self.batch_size, -1])

        #fin_state = self.gat(self.ASG, self.ASG_w, item, topK) + self.ggnn(WSG_in, WSG_out, item) + self.ggnn(AIG_in, AIG_out, item)
        fin_state = self.gat(self.ASG, self.ASG_w, item, topK) + self.ggnn(WSG_in, WSG_out, item) + self.ggnn(AIG_in, AIG_out, item)

        rm = tf.reduce_sum(mask, 1)
        last_id = tf.gather_nd(alias, tf.stack([tf.range(self.batch_size), tf.to_int32(tf.abs(rm-1))], axis=1))
        last_h = tf.gather_nd(fin_state, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(fin_state[i], alias[i]) for i in range(self.batch_size)], axis=0)
        last = tf.reshape(tf.matmul(last_h, self.nasr_w1), [self.batch_size, 1, -1])
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            mask, [-1, 1])
        ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),
                        tf.reshape(last, [-1, self.out_size])], -1)
        representation = tf.matmul(ma, self.B)
        return representation



