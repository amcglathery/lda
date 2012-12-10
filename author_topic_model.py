import sys, os, random, math, itertools
import numpy as np
from author_model import AuthorModel

class AuthorTopicModel(AuthorModel):

  # all of the following are numpy arrays
  author_topic_counts = None
  author_topic_totals = None # sum of author_topic_counts over all topics

  def __init__(self, alpha, beta, topics):
    self.topics = topics
    self.alpha = alpha
    self.beta = beta

  def init_count_matricies(self):
    self.topic_word_counts = np.zeros((self.topics, len(self.id_to_word)), dtype=np.uint16)
    self.topic_word_totals = np.zeros(self.topics, dtype=np.uint32)
    self.author_topic_counts = np.zeros((len(self.id_to_author), self.topics), dtype=np.uint16)
    self.author_topic_totals = np.zeros(len(self.id_to_author), dtype=np.uint16)

  def build_term(self, word, doc_id):
    word_id = self.word_to_id[word]
    author_id = self.author_to_id[random.choice(self.doc_author_map[doc_id])]
    topic = random.randint(0, self.topics - 1)
    term = (word_id, author_id, topic, doc_id)
    return term

  def add_term(self, term):
    word_id, author_id, topic, doc_id = term
    self.topic_word_counts[topic][word_id] += 1
    self.topic_word_totals[topic] += 1
    self.author_topic_counts[author_id][topic] += 1
    self.author_topic_totals[author_id] += 1

  def remove_term(self, term):
    word_id, author_id, topic, doc_id = term
    self.topic_word_counts[topic][word_id] -= 1
    self.topic_word_totals[topic] -= 1
    self.author_topic_counts[author_id][topic] -= 1
    self.author_topic_totals[author_id] -= 1

  def get_probs(self, term):
    word_id, author_id, _, doc_id = term
    probs = []
    for author in self.author_to_id.values():
      if self.id_to_author[author] in self.doc_author_map[doc_id]:
        probs.append([self.prob((word_id, author, _, doc_id), topic) for topic in range(self.topics)])
      else :
        probs.append([0.0 for topic in range(self.topics)])
    return probs

  def prob(self, term, topic):
    """
    returns the probability that the given term belongs to the given topic
    """
    word_id, author_id, _, doc_id = term
    p_Word = (self.topic_word_counts[topic][word_id] + self.beta) / \
             (self.topic_word_totals[topic] + len(self.id_to_word) * self.beta)
    p_Topic = (self.author_topic_counts[author_id][topic] + self.alpha) / \
              (self.author_topic_totals[author_id] + self.topics * self.alpha)
    return p_Word * p_Topic

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

        probs = self.get_probs(term)

        rand = random.random() * sum(sum(p) for p in probs)
        choices = []
        for author_id in self.author_to_id.values():
          for topic in range(self.topics):
            choices.append((author_id, topic))

        for author_id, topic in choices:
          rand -= probs[author_id][topic]
          if rand <= 0:
            author_choice = author_id
            topic_choice = topic
            break

        term = (term[0], author_choice, topic_choice, term[3])
        self.add_term(term)
        self.terms[t] = term

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
      ppx += math.log(sum(sum(x) for x in tm.get_probs(term)))
    ppx = math.exp(-ppx / len(self.test_terms))

    return ppx

  def copy(self):
    tm = AuthorTopicModel(self.alpha, self.beta, self.topics)
    tm.terms = self.terms
    tm.word_to_id = self.word_to_id
    tm.id_to_word = self.id_to_word

    tm.author_to_id = self.author_to_id
    tm.id_to_author = self.id_to_author

    tm.topic_word_counts = np.copy(self.topic_word_counts)
    tm.author_topic_counts = np.copy(self.author_topic_counts)
    tm.topic_word_totals = np.copy(self.topic_word_totals)
    tm.author_topic_totals = np.copy(self.author_topic_totals)

    tm.doc_author_map = self.doc_author_map
    return tm

if __name__ == "__main__":
  model = AuthorTopicModel(2.5,0.01, 20)
  model.load_files(sys.argv[1], sys.argv[2], sys.argv[3])
  print model.perplexity(1, 10)
