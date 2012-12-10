import sys, os, re, random, itertools, math
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
  topic_word_totals = None # sum of topic_word_counts over all words
  document_word_totals = None # sum of document_topic_counts over all documents

  documents = [] # a list of strings of document names indexed by doc_id
  terms = []
  test_terms = []

  stemmer = nltk.stem.PorterStemmer()
  stopwords = nltk.corpus.stopwords.words("english")

  def __init__(self, alpha, beta, topics):
    self.topics = topics
    self.alpha = alpha
    self.beta = beta

  def load_files(self, test_data, training_data):
    """
    loads the files into the topic_word_count and document_topic_count
    matricies, and populates the requisate lexicon data structures and
    sum vectors
    """
    self.build_lexicon(training_data)
    self.build_lexicon(test_data)
    self.load_training_data(training_data)
    self.load_test_data(test_data)

  def load_training_data(self, files_dir):
    print "Populating Count Matricies"
    self.topic_word_counts = np.zeros((self.topics, len(self.id_to_word)), dtype=np.uint16)
    self.document_topic_counts = np.zeros((len(self.documents), self.topics), dtype=np.uint16)
    self.topic_word_totals = np.zeros(self.topics, dtype=np.uint32)
    self.document_word_totals = np.zeros(len(self.documents), dtype=np.uint32)
    for doc_id, f in enumerate(os.listdir(files_dir)):
      with open(files_dir + '/' + f, 'r') as doc:
        for word in re.split("\W|_", doc.read()):
          word = self.clean_word(word, None)
          if word:
            word_id = self.word_to_id[word]
            topic = random.randint(0, self.topics - 1)
            term = (word_id, topic, doc_id)
            self.terms.append(term)
            self.add_term(term)
    print len(self.terms), " Terms"

  def load_test_data(self, files_dir):
    print "Building test terms"
    for doc_id, f in enumerate(os.listdir(files_dir)):
      with open(files_dir + '/' + f, 'r') as doc:
        for word in re.split("\W|_", doc.read()):
          word = self.clean_word(word, None)
          if word:
            word_id = self.word_to_id[word]
            topic = random.randint(0, self.topics - 1)
            term = (word_id, topic, doc_id)
            self.test_terms.append(term)
    print len(self.test_terms), " Test Terms"

  def add_term(self, term):
    """
    adds term counts into count matricies
    """
    word_id, topic, doc_id = term
    self.topic_word_counts[topic][word_id] += 1
    self.document_topic_counts[doc_id][topic] += 1
    self.topic_word_totals[topic] += 1
    self.document_word_totals[doc_id] += 1

  def remove_term(self, term):
    """
    removes term counts from count matricies
    """
    word_id, topic, doc_id = term
    self.topic_word_counts[topic][word_id] -= 1
    self.document_topic_counts[doc_id][topic] -= 1
    self.topic_word_totals[topic] -= 1
    self.document_word_totals[doc_id] -= 1

  def build_lexicon(self, files_dir):
    """
    loads the words in the files_dir into the lexicon
    """
    seen_words = set([""])
    print "Building Lexicon"
    for f in os.listdir(files_dir):
      self.documents.append(f)
      with open(files_dir + '/' + f, 'r') as doc:
        for word in re.split("\W|_", doc.read()):
          word = self.clean_word(word, seen_words)
          if word and not word in self.word_to_id:
            self.word_to_id[word] = len(self.id_to_word)
            self.id_to_word.append(word)
    print len(self.id_to_word), " words found"

  def clean_word(self, word, seen_words):
    """
    stops, stems, and removes one-off words
    """
    if word in self.stopwords or len(word) < 2:
      return None
    word = self.stemmer.stem(word)
    if (seen_words and word in seen_words) or word in self.word_to_id:
      return word
    if seen_words:
      seen_words.add(word)
    return None

  def gibbs_sample(self, iterations):
    """
    learns the model through gibbs sampling for the given number of iterations
    """
    for i in range(iterations):
      print "Gibbs Iteration ", i
      for t in range(len(self.terms)):
        # print "Term ", t
        term = self.terms[t]
        self.remove_term(term)

        probs = [ self.p_topic(term, topic) for topic in range(self.topics) ]

        rand = random.random() * sum(probs)
        topic_choice = -1
        while True:
          topic_choice += 1
          rand -= probs[topic_choice]
          if rand <= 0:
            break

        term = (term[0], topic_choice, term[2])
        self.add_term(term)
        self.terms[t] = term

  def p_topic(self, term, topic):
    """
    returns the probability that the given term belongs to the given topic
    """
    word_id, _, doc_id = term
    p_Word = (self.topic_word_counts[topic][word_id] + self.beta) / \
             (self.topic_word_totals[topic] + len(self.id_to_word) * self.beta)
    p_Topic = (self.document_topic_counts[doc_id][topic] + self.alpha) / \
              (self.document_word_totals[doc_id] + self.topics * self.alpha)
    return p_Word * p_Topic

  def perplexity(self, n_d_train, iterations):
    """
    calculates the perplexity of the model by retraining the model using an
    additional n_d_train terms from each of the test files and then calculates
    the perplexity of the remaining terms in the test files
    """
    tm = self.copy()

    for doc_test_terms in (list(g) for k, g in itertools.groupby(self.test_terms, key=lambda x: x[2])):
      for i in range(n_d_train):
        index = random.randint(0, len(doc_test_terms) - 1)
        term = doc_test_terms.pop(index)
        tm.terms.append(term)
        tm.add_term(term)

    tm.gibbs_sample(iterations)

    ppx = 0.0
    for term in self.test_terms:
      ppx += math.log(sum( tm.p_topic(term, k) for k in range(tm.topics) ) )
    ppx = math.exp(-ppx / len(self.test_terms))

    return ppx

  def copy(self):
    tm = TopicModelLDA(self.alpha, self.beta, self.topics)
    tm.terms = self.terms
    tm.word_to_id = self.word_to_id
    tm.id_to_word = self.id_to_word

    tm.topic_word_counts = np.copy(self.topic_word_counts)
    tm.document_topic_counts = np.copy(self.document_topic_counts)
    tm.topic_word_totals = np.copy(self.topic_word_totals)
    tm.document_word_totals = np.copy(self.document_word_totals)
    return tm

  def print_topics(self, n):
    for topic in range(self.topics):
      print "Topic: ", topic
      word_count = [ (word_index, count) for word_index, count in enumerate(self.topic_word_counts[topic]) ]
      for word_index, count in sorted(word_count, key=lambda x: x[1])[:n]:
        print self.id_to_word[word_index], " ",
      print

if __name__ == "__main__":
  model = TopicModelLDA(2.5, 0.01, 20)
  model.load_files(sys.argv[1], sys.argv[2])
  print model.perplexity(1, 10)
