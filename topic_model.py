import sys, os, re
import nltk
import numpy as np

class TopicModelLDA:
  """
  does basic lda topic modeling
  a term is represented as a (word_index, topic, doc_id) pair
  """
  # lexicon data structures, represents a bijection between a word and its
  # corresponding id
  word_to_id = {}
  id_to_word = []

  # all of the following are numpy arrays
  topic_word_counts = None
  document_topic_counts = None
  topic_word_totals = None
  document_word_totals = None

  documents = [] # a list of strings of document names indexed by doc_id

  stemmer = nltk.stem.PorterStemmer()
  stopwords = nltk.corpus.stopwords.words("english")

  def load_files(self, files_dir):
    """
    loads the files into the topic_word_count and document_topic_count
    matricies, and populates the requisate lexicon data structures and
    sum vectors
    """
    self.build_lexicon(files_dir)
    for f in os.listdir(files_dir):
      pass

  def build_lexicon(self, files_dir):
    """
    loads the words in the files_dir into the lexicon
    """
    seen_words = set([])
    for f in os.listdir(files_dir):
      print files_dir + '/' + f
      with open(files_dir + '/' + f, 'r') as doc:
        for word in re.split("\W|_", doc.read()):
          word = self.clean_word(word, seen_words)
          if word and not word in self.word_to_id:
            self.word_to_id[word] = len(self.id_to_word)
            self.id_to_word.append(word)

  def clean_word(self, word, seen_words):
    """
    stops, stemms, and removes one-off words
    """
    if word in self.stopwords or len(word) < 2:
      return None
    word = self.stemmer.stem(word)
    if word in seen_words or word in self.word_to_id:
      return word
    seen_words.add(word)
    return None

  def gibbs_sample(self, iterations):
    """
    learns the model through gibbs sampling for the given number of iterations
    """
    pass

  def p_Topic(self, term, topic):
    """
    returns the probability that the given term belongs to the given topic
    """
    pass

  def perplexity(self, n_d_train, test_dir):
    """
    calculates the perplexity of the model by retraining the model using an
    additional n_d_train terms from each of the test files and then calculates
    the perplexity of the remaining terms in the test files
    """
    pass

if __name__ == "__main__":
  model = TopicModelLDA()
  model.load_files(sys.argv[1])
  for word in model.id_to_word:
    print word
  print "-----------------------------"
  print len(model.id_to_word)

