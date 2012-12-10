import cPickle, sys, os, random, math
import numpy as np
from topic_model import TopicModelLDA

class AuthorModel(TopicModelLDA):

  # all of the following are numpy arrays
  author_word_counts = None
  author_word_totals = None # sum of author_word_counts over all words

  doc_author_map = None

  # bijection from authors to author ids. author ids are used to index into
  # the count matricies
  id_to_author = []
  author_to_id = {}

  def __init__(self, beta):
    self.beta = beta

  def load_files(self, test_data, training_data, map_file):
    with open(map_file, 'r') as f:
      self.doc_author_map = cPickle.load(f)

    doc_set = os.listdir(test_data) + os.listdir(training_data)

    # the pickle has the document names in int format (no leading zeros)
    # so the filenames must be converted
    doc_set_strings = set([ str(int(d[:7])) for d in doc_set ])

    self.id_to_doc = list(doc_set_strings)
    self.doc_to_id = {doc : doc_id for (doc_id, doc) in enumerate(self.id_to_doc)}

    # remove all document entries that do not appear in either the training or
    # test docs, and have the keys be ids, not names
    self.doc_author_map = { self.doc_to_id[d] : alist for (d, alist) in self.doc_author_map.items() if d in doc_set_strings }

    self.id_to_author = set([])
    for alist in self.doc_author_map.values():
      self.id_to_author.update(alist)
    self.id_to_author = list(self.id_to_author)

    self.author_to_id = {author : author_id for (author_id, author) in enumerate(self.id_to_author)}

    self.build_lexicon(training_data)
    self.build_lexicon(test_data)
    self.init_count_matricies()
    self.load_training_data(training_data)
    self.load_test_data(test_data)

  def init_count_matricies(self):
    self.author_word_counts = np.zeros((len(self.id_to_author), len(self.id_to_word)), dtype=np.uint16)
    self.author_word_totals = np.zeros(len(self.id_to_author), dtype=np.uint16)

  def build_term(self, word, doc_id):
    word_id = self.word_to_id[word]
    author_id = self.author_to_id[random.choice(self.doc_author_map[doc_id])]
    term = (word_id, author_id, doc_id)
    return term

  def add_term(self, term):
    word_id, author_id, doc_id = term
    self.author_word_counts[author_id][word_id] += 1
    self.author_word_totals[author_id] += 1

  def remove_term(self, term):
    word_id, author_id, doc_id = term
    self.author_word_counts[author_id][word_id] -= 1
    self.author_word_totals[author_id] -= 1

  def get_probs(self, term):
    word_id, author_id, doc_id = term
    probs = []
    for author in self.author_to_id.values():
      if self.id_to_author[author] in self.doc_author_map[doc_id]:
        numer = self.author_word_counts[author][word_id] + self.beta
        denom = self.author_word_totals[author] + len(self.id_to_word) * self.beta
        probs.append(numer / denom)
      else :
        probs.append(0.0)
    return probs

  def copy(self):
    am = AuthorModel(self.beta)
    am.terms = self.terms
    am.word_to_id = self.word_to_id
    am.id_to_word = self.id_to_word

    am.author_to_id = self.author_to_id
    am.id_to_author = self.id_to_author

    am.author_word_counts = np.copy(self.author_word_counts)
    am.author_word_totals = np.copy(self.author_word_totals)

    am.doc_author_map = self.doc_author_map
    return am

if __name__ == "__main__":
  model = AuthorModel(0.01)
  model.load_files(sys.argv[1], sys.argv[2], sys.argv[3])
  print model.perplexity(1, 10)
