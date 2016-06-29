from __future__ import print_function
import numpy as np 
import os
import re
import itertools
from collections import Counter, OrderedDict 
from nltk.corpus import stopwords
from nltk.stem import *
import cPickle as pkl 
import glob

# =================================================

def clean_str(string):
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " ", string)
    string = re.sub(r"n\'t", " ", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " ", string)
    string = re.sub(r"\'ll", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"br>", " ", string)
    string = re.sub(r">", " ", string)
    return string.strip().lower()

def build_data(sentences):
    x_text = [clean_str(sent) for sent in sentences]
    x_text = [s.split(" ") for s in x_text]
    return x_text


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    sequence_length = 5000
    # print(sequence_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding > 0:
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        else:
            new_sentence = sentence[:sequence_length]
            padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_dataset(sentences, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(Counter(itertools.chain(*sentences)).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for ind, sentence in enumerate(sentences):
      data.append([])
      for word in sentence:
          if word in dictionary:
              index = dictionary[word]
          else:
              index = 0  # dictionary['UNK']
              unk_count = unk_count + 1
          data[ind].append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  vocabulary = dictionary
  vocabulary_inv = reverse_dictionary
  return data, count, dictionary, reverse_dictionary


def build_docs(path):
    olddir = os.getcwd()
    currdir = path
    docs = []
    labels = []
    idx = 0
    no_access = 0
    for folder in os.listdir(path):
        subpath = currdir + "/" + folder + "/all_documents/"
        try:
            os.chdir(subpath)
        except:
            print("couldn't open", folder)
            no_access += 1
            continue
        for ff in glob.glob("*"):
            with open(ff, "r") as f:
                docs.append(f.read().strip())
                labels.append(idx)
        print(idx, folder)    
        idx = idx + 1
        if idx > 35:
            break
    os.chdir(currdir)
    os.chdir(olddir)
    print(len(docs))
    print(no_access, idx)
    return docs, labels

def process_lines(line_docs):
    processed = []
    for idx, doc in enumerate(line_docs):
        processed.append('')
        for line in doc:
            if ':' not in line:
                processed[-1] = processed[-1] + line
        if idx == 0:
            print(processed[0])
        #processed[-1] = list(itertools.chain(*processed[-1]))

    return processed

def tokenize_lines(proc_line_docs):
    tokenized = []
    for idx, doc in enumerate(proc_line_docs):
        tokenized.append([])
        for line in doc:
            words = line.split(" ")
            # if idx == 0:
            #     print(words)
            tokenized[-1].append(words)
        tokenized[-1] = list(itertools.chain(*tokenized[-1]))
    print(tokenized[:3])
    return tokenized

def load_data_and_labels(vocab_size):
    if os.path.exists("./enron_35000.npz"):
        print("loading saved enron data")
        z = np.load("./enron_35000.npz")
        x = z['x']
        y = z['y']
        vocabulary = pkl.load(open("./enron_35000_vocab.p", "rb"))
        vocabulary_inv = pkl.load(open("./enron_35000_vocab_inv.p", "rb"))
        print(vocabulary_inv[5], vocabulary_inv[6], vocabulary_inv[7])
	print(vocabulary["<PAD/>"])
    else:
        path = "/home/nishit/Desktop/rte/maildir"
        docs, labels = build_docs(path)
        docs_in_batches = []
        line_docs = []
        for doc in docs:
            line_docs.append(doc.split("\n"))
        proc_line_docs = process_lines(line_docs)
        print(proc_line_docs[1])
        # line_docs_merged = list(itertools.chain(*proc_line_docs))
        # print(line_docs_merged[1])
        # tokenized_docs = tokenize_lines(proc_line_docs)
        for idx in range(35):
            docs_in_batches.append(proc_line_docs[idx:idx+1000])
        print(len(docs_in_batches))
        # print(docs_in_batches[1][1])
        padded_docs = []
        for i, docs in enumerate(docs_in_batches):
            cleaned_docs = build_data(docs)
            print(i)
            if i == 0:
                print(cleaned_docs[1])
            padded_docs.append(pad_sentences(cleaned_docs))
        print("padded", len(padded_docs))
        print(len(padded_docs))
        merged = list(itertools.chain(*padded_docs))
        print("merged")
        x, count, vocabulary, vocabulary_inv = build_dataset(merged, vocab_size)
        print("built dataset")
        x = np.array(x)
        y = np.array(labels[:35000])
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        
        x = x_shuffled
        y = y_shuffled
        np.savez("./enron_35000.npz", x=x, y=y)
        pkl.dump(vocabulary, open("./enron_35000_vocab.p", "wb"))
        pkl.dump(vocabulary_inv, open("./enron_35000_vocab_inv.p", "wb"))
        
        
    return x, y, vocabulary, vocabulary_inv

# load_data_and_labels(10000)
