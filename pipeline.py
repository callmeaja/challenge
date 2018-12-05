import tensorflow as tf


class textCNN(object):

    def __init__(self, sequence_length, filter_size, num_filters, num_classes):

        # Extraction from Passages
        self.input_p = tf.placeholder(tf.float32, [None, 50], name="passages")
        self.input_q = tf.placeholder(tf.float32, [None, 10], name="query")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # L2 Loss Tracking
        l2_loss = tf.constant(0.0)

        pooled_outputs = []
        with tf.name_scope("convp-maxpool"):
            filter_shape = [filter_size, 50, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[5]), name='b')
            conv = tf.nn.conv2d(
                self.input_p,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv')

            # Apply non - linearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size+1, 1, 1],
                padding='VALID',
                name='pool'
            )
            pooled_outputs.append(pooled)

        with tf.name_scope("convq-maxpool"):
            filter_shape = [filter_size, 50, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[5]), name='b')
            conv = tf.nn.conv2d(
                self.input_q,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv'
            )

            # Apply non - linearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size+1, 1, 1],
                padding='VALID',
                name='pool'
            )
            pooled_outputs.append(pooled)
        num_filters_total = 2*num_filters
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Adding in dropout layer
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))