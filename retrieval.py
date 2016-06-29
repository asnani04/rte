from __future__ import print_function
import numpy as np 
import tensorflow as tf 
import os
import get_enron_data
import math
import kmeans

# ===========================================

print("Loading data...")
x, y, vocabulary, vocabulary_inv = get_enron_data.load_data_and_labels(10000)
x_train, x_valid = x[:-1000], x[-1000:]
y_train, y_valid = y[:-1000], y[-1000:]
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)))

# ==============================================

context_size = 4
embedding_size = 300
vocabulary_size = 10000
train_size = 34000

print("Enter the number of keywords.")
no_words = int(raw_input())
words = []
for idx in range(no_words):
	words.append(str(raw_input()))

input_data = tf.placeholder(tf.int32, shape=[1, no_words])
word_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
sent_embeddings = tf.Variable(
        tf.random_uniform([train_size, embedding_size], -1.0, 1.0))
softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size * (2*context_size + 1)], 
                            stddev=1.0 / math.sqrt(embedding_size * 9)))
softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

inp_embed = tf.nn.embedding_lookup(word_embeddings, input_data)
inp_embed_mean = tf.reduce_mean(inp_embed, 1)

norm_train = tf.sqrt(tf.reduce_sum(tf.square(sent_embeddings), 1, keep_dims=True))
norm_test = tf.sqrt(tf.reduce_sum(tf.square(inp_embed_mean), 1, keep_dims=True))
normalized_train = sent_embeddings / norm_train
normalized_test = inp_embed_mean / norm_test
sim_sent = tf.matmul(normalized_test, tf.transpose(normalized_train))


loader = tf.train.Saver({'word_embeddings': word_embeddings, 'sent_embeddings': sent_embeddings, 
            'softmax_weights': softmax_weights, 'softmax_biases': softmax_biases})


sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
loader.restore(sess, "./pvec_35000_300.ckpt")
word_rep = []
for word in words:
	rep = vocabulary[word]
	word_rep.append(rep)

word_rep = np.array(word_rep)
word_rep = np.reshape(word_rep, [1, no_words])
feed = {input_data: word_rep}

distance = sess.run(sim_sent, feed_dict = feed)
no_of_neighbors = 100
nearest = (-distance[0, :]).argsort()[0:no_of_neighbors]
print("nearest computed")
nearest_docs = [sent_embeddings.eval()[nearest[k]] for k in range(no_of_neighbors)]
print("off from nearest docs") 
centroids, assignments, assigned_ind = kmeans.TFKMeansCluster(nearest_docs, 10)

for k in range(10):
	for ind in range(3):
		num = assigned_ind[k][ind]
		doc = x_train[num]
		doc_words = ''
		for number in doc:
			word = vocabulary_inv[number]
			if number == 1:
				continue
			doc_words = doc_words + " " + word
		print(k, ind, doc_words)


