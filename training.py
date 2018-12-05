import tensorflow as tf
import os
import time
import datetime
from pipeline import textCNN

os.chdir('E:\Competitions\Microsoft AI Challenge')


# Hyperparameters
filter_size = 3
num_filters = 128
dropout_keep_prob = 0.5

# Training parameters
batch_size = 64
num_epochs = 200
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5

def train(x_train, y_train, x_test, y_test):
    train_p = x_train.iloc[:, 1]
    test_p = x_test.iloc[:, 1]
    train_q = x_train.iloc[:, 0]
    test_q = x_test.iloc[:, 0]

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = textCNN(
                sequence_length=50,
                filter_size=filter_size,
                num_filters=num_filters,
                num_classes=2
            )

            # Defining training procedure
            global_step = tf.Variable(0, name='gloabal_step', trainable=False)
            optimiser = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimiser.compute_gradients(cnn.loss)
            train_op = optimiser.apply_gradients(grads_and_vars, global_step=global_step)

            # Output Directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Initialise all variables
            sess.run(tf.global_variables_initializer())

            def train_step(q_batch, p_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_q: q_batch,
                    cnn.input_p: p_batch
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(q_batch, p_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_q: q_batch,
                    cnn.input_p: p_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = batch_iter(list(zip(train_p, train_q, y_train)), batch_size, num_epochs)

            # Training loop for each batch
            for batch in batches:
                p_batch, q_batch, y_batch = zip(*batch)
                train_step(q_batch, p_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(test_q, test_p, y_test, writer=dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    execfile('Codes/data_cleaning.py')
    train(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    tf.app.run()