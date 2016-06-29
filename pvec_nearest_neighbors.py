from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
import get_enron_data
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE


# Data Preparatopn
# ===============================================================

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = get_enron_data.load_data_and_labels(50000)
# Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_valid = x[:-10000], x[-10000:]
y_train, y_valid = y[:-10000], y[-10000:]
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)))
# print(x_train[0], len(x_train))

# ================================================================

sent_index = 0
word_index = 4
context_size = 4
batch_size = 128
vocabulary_size = len(vocabulary)
sent_size = len(x)
train_size = len(x_train)
test_size = len(x_valid)
sequence_length = x_train.shape[1]
num_labels = 2
hidden_size = 10

# Function to generate training batches
def generate_batch(x, batch_size):
    global sent_index
    global word_index
    batch = np.ndarray(shape=(batch_size, 2 * context_size), dtype = np.int32)
    labels = np.ndarray(shape=(batch_size,1), dtype = np.int32)
    batch_sent = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    for i in range(batch_size):
        batch[i] = [x[sent_index][word_index-4], x[sent_index][word_index-3], 
                    x[sent_index][word_index-2], x[sent_index][word_index-1], 
                    x[sent_index][word_index+1], x[sent_index][word_index+2], 
                    x[sent_index][word_index+3], x[sent_index][word_index+4]]
        batch_sent[i] = [sent_index]
        labels[i] = x[sent_index][word_index]
        word_index = word_index + 1
        if x[sent_index][word_index+2] == 1:
            #print("sent over")
            sent_index = sent_index + 1
            word_index = 4
        if word_index + 4 == sequence_length - 1:
            sent_index = sent_index + 1
            word_index = 4
        if sent_index >= len(x):
            sent_index = 0
    return batch, batch_sent, labels

batch, batch_sent, labels = generate_batch(x_train, batch_size)
# print(len(batch[1]))
print(' batch:', [[vocabulary_inv[i] for i in batch[j]] for j in range(4)])
print(' labels:', [[vocabulary_inv[i] for i in labels[j]] for j in range(4)])
# print(' batch_sent:', [batch_sent[i] for i in range(4)])
# print(' slice_word:', tf.slice(batch, [0,4], [4, 1]))

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# ==========================================================================

def cosine_distance(idx, x, train_sents, sent_embeddings, test_sent_embeddings):
    global train_size
    id_test = np.ndarray(shape=(1, 1), dtype=np.int32)
    id_test[0] = idx
    embedding_test = tf.nn.embedding_lookup(test_sent_embeddings, id_test)
    embedding_train = tf.nn.embedding_lookup(sent_embeddings, 
                                             train_sents)
    distance = np.ndarray(shape=(train_size, 1), dtype=np.int32)
    test_norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_test, 1, keep_dims=True)))
    train_norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_train, 1, keep_dims=True)))
    embedding_train = embedding_train / train_norm
    embedding_test = embedding_test / test_norm
    
    for i in range(train_size):
        distance[i] = np.dot(embedding_test.eval()[0][0], embedding_train.eval()[i][0]) / (
            test_norm * train_norm)
    return distance

# ============================================================================

embedding_size = 300
valid_size = 16
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    
    #Input Data
    init_data = tf.placeholder(tf.int32, shape=[batch_size, 2*context_size])
    init_data_sent = tf.placeholder(tf.int32, shape=[batch_size, 1])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    test_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    
    # train_embed = tf.placeholder(tf.float32, shape=[batch_size, embedding_size])
    # class_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    # class_embed = tf.placeholder(tf.float32, shape=[test_size, embedding_size])

    #Variables
    word_embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    sent_embeddings = tf.Variable(
        tf.zeros([train_size, embedding_size]))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size * (2*context_size + 1)], 
                            stddev=1.0 / math.sqrt(embedding_size * 5)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    test_sent_embeddings = tf.Variable(
        tf.random_uniform([test_size, embedding_size]))

    #Model
    word_embed = tf.nn.embedding_lookup(word_embeddings, init_data)
    # print(tf.shape(word_embed))
    sent_embed = tf.nn.embedding_lookup(sent_embeddings, init_data_sent)
    # print(tf.shape(sent_embed))
    embed = tf.concat(1, [word_embed, sent_embed])
    # print(tf.shape(embed[0]))
    embed_shape = embed.get_shape().as_list()
    embed = tf.reshape(embed, [embed_shape[0], embed_shape[1] * embed_shape[2]])
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                                     train_labels, num_sampled, vocabulary_size))
    #regs = tf.nn.l2_loss(softmax_weights) + tf.nn.l2_loss(softmax_biases)
    #loss = loss + 0.01 * regs

    test_word_embed = tf.nn.embedding_lookup(word_embeddings, init_data)
    test_sent_embed = tf.nn.embedding_lookup(test_sent_embeddings, init_data_sent)
    test_embed = tf.concat(1, [test_word_embed, test_sent_embed])
    test_embed_shape = test_embed.get_shape().as_list()
    test_embed = tf.reshape(test_embed, [test_embed_shape[0], test_embed_shape[1] * test_embed_shape[2]])
    test_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, test_embed,
                                                          test_labels, num_sampled, vocabulary_size))
    
    sent_embed_hist = tf.histogram_summary("sent_embeddings", sent_embeddings)
    test_sent_embed_hist = tf.histogram_summary("test_sent_embeddings", test_sent_embeddings)
    merged = tf.merge_all_summaries()

    # Optimizer
    optimizer = tf.train.AdagradOptimizer(0.5).minimize(loss)
    optimizer_test = tf.train.AdagradOptimizer(0.5).minimize(
        test_loss, var_list = [test_sent_embeddings])
    norm = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
    normalized_embeddings = word_embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    norm_train = tf.sqrt(tf.reduce_sum(tf.square(sent_embeddings), 1, keep_dims=True))
    norm_test = tf.sqrt(tf.reduce_sum(tf.square(test_sent_embeddings), 1, keep_dims=True))
    normalized_train = sent_embeddings / norm_train
    normalized_test = test_sent_embeddings / norm_test
    sim_sent = tf.matmul(normalized_test, tf.transpose(normalized_train))

    # Classifier
    # class_weights = tf.Variable(tf.truncated_normal([embedding_size, hidden_size]))
    # class_biases = tf.Variable(tf.zeros([hidden_size]))
    # hidden_weights = tf.Variable(tf.truncated_normal([hidden_size, num_labels]))
    # hidden_biases = tf.Variable(tf.zeros([num_labels]))
   
    # hidden = tf.nn.relu(tf.matmul(train_embed,  class_weights) + class_biases)
    # logits = tf.matmul(hidden, hidden_weights) + hidden_biases
    # predictor_loss = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(logits, class_labels))
    # regularizers = tf.nn.l2_loss(class_weights) + tf.nn.l2_loss(class_biases) + tf.nn.l2_loss(
    #     hidden_weights) + tf.nn.l2_loss(hidden_biases)
    # predictor_loss = predictor_loss + 0.01 * regularizers
    
    # # Classifier Optimization
    # optimizer_predictor = tf.train.GradientDescentOptimizer(0.5).minimize(
    #     predictor_loss, var_list = [class_weights, class_biases, 
    #                                 hidden_weights, hidden_biases])
    # train_prediction = tf.nn.softmax(logits)
    # test_prediction = tf.nn.softmax(tf.matmul((tf.nn.relu(tf.matmul(
    #     class_embed, class_weights) + class_biases)), hidden_weights) + hidden_biases)

    saver_train = tf.train.Saver({'word_embeddings': word_embeddings, 'sent_embeddings': sent_embeddings, 
            'softmax_weights': softmax_weights, 'softmax_biases': softmax_biases})
    saver_test = tf.train.Saver({'test_sent_embeddings': test_sent_embeddings})
    

# =============================================================================

num_steps = 1000001
num_test_steps = 250001
num_class_train_steps = 4001

with tf.Session(graph=graph) as session:
    writer = tf.train.SummaryWriter("/tmp/pvec_logs",
                                  session.graph.as_graph_def(add_shapes=True))    
    if os.path.exists("./pvec_class_300.ckpt"):
        print("loading saved model")
        saver_train.restore(session, "./pvec_class_300.ckpt")
    else:
        tf.initialize_all_variables().run()
        print('Initialized')
        # print(np.array(sent_embeddings.eval()[0]))
        average_loss = 0
        for step in range(num_steps):
            batch_words, batch_sents, batch_labels = generate_batch(
                x_train, batch_size)
            feed_dict = {init_data : batch_words, init_data_sent: batch_sents,
                         train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 5000 == 0:
                if step > 0:
                    average_loss = average_loss / 5000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
                    # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 50000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = vocabulary_inv[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = vocabulary_inv[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        final_embeddings = normalized_embeddings.eval()
        saver_train.save(session, "./pvec_class_300.ckpt")

    # for step in range(num_class_train_steps):
    #     offset = (step * batch_size) % (y_train.shape[0] - batch_size)
    #     # print(np.array(sent_embeddings.eval()[3]))
    #     batch_data = sent_embeddings[offset:(offset + batch_size), :].eval() 
    #     batch_labels = y_train[offset:(offset + batch_size), :]
    #     feed_dict = { train_embed: batch_data, class_labels : batch_labels}
    #     _, l, predictions = session.run(
    #         [optimizer_predictor, predictor_loss, train_prediction], feed_dict=feed_dict)
    #     if (step % 500 == 0):
    #         print("Minibatch loss at step %d: %f" % (step, l))
    #         print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
    
    if os.path.exists("./pvec_class_300_test.ckpt"):
        print("loading saved test data")
        saver_test.restore(session, "./pvec_class_300_test.ckpt")
    else:
        sent_index = 0
        word_index = 4
        for step in range(num_test_steps):
            average_loss = 0
            batch_words, batch_sents, batch_labels = generate_batch(
                x_valid, batch_size)
            feed_dict = {init_data : batch_words, init_data_sent: batch_sents,
                         test_labels: batch_labels}
            _, l = session.run([optimizer_test, test_loss], feed_dict=feed_dict)
            average_loss += l
            if step % 5000 == 0:
                if step > 0:
                    average_loss = average_loss / 5000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
                sim = similarity.eval()
                # for i in range(valid_size):
                #     valid_word = vocabulary_inv[valid_examples[i]]
                #     top_k = 8 # number of nearest neighbors
                #     nearest = (-sim[i, :]).argsort()[1:top_k+1]
                #     log = 'Nearest to %s:' % valid_word
                #     for k in range(top_k):
                #         close_word = vocabulary_inv[nearest[k]]
                #         log = '%s %s,' % (log, close_word)
                #     print(log)
        saver_test.save(session, "./pvec_class_300_test.ckpt")
                
    # feed_dict = {class_embed: test_sent_embeddings.eval()}
    # print('Test Accuracy: %.1f%%' % accuracy(
    #     test_prediction.eval(feed_dict=feed_dict), y_valid))
    

    # train_sents = np.ndarray(shape=(train_size, 1), dtype=np.int32)
    # for i in range(train_size):
    #     train_sents[i] = i
    # acc = 0
    # for idx in range(len(x_valid/10)):
    #     c = 0
    #     dist = cosine_distance(idx, x_train, train_sents, sent_embeddings, test_sent_embeddings)
    #     nearest_neighbours = np.sort(dist)[:3]
    #     for k in nearest_neighbours:
    #         if y_train[k][0] == y_valid[idx][0]:
    #             c = c + 1
    #     if c > 1:
    #         acc = acc + 1
    #     if idx % 1 == 0:
    #         print("%d correct out of %d" % (acc, idx))
    # acc = float(float(acc) / test_size/10)
    # print("accuracy is: %f" % (acc))

    distance = sim_sent.eval()
    acc = 0.0
    for idx in range(test_size):
        no_of_neighbors = 201
        nearest = (-distance[idx, :]).argsort()[0:no_of_neighbors]
        c = 0
        for k in range(no_of_neighbors):
            if y_train[nearest[k]][0] == y_valid[idx][0]:
                c = c + 1
        if c > no_of_neighbors / 2:
            acc = acc + 1.0
            # print("yes %d" % idx)
            # print("no %d" % idx)
    print("accuracy: %f" % float(acc / test_size))
        
    



# =============================================================================
    


        

