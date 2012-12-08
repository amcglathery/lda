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

  def load_files(files_dir):
    """
    loads the files into the topic_word_count and document_topic_count
    matricies, and populates the requisate lexicon data structures and
    sum vectors
    """

  def gibbs_sample(iterations):
    """
    learns the model through gibbs sampling for the given number of iterations
    """

  def p_Topic(term, topic):
    """
    calculates the probability that the given term belongs to the given topic
    """

  def perplexity(n_d_train, test_dir):
    """
    calculates the perplexity of the model by retraining the model using an
    additional n_d_train terms from each of the test files and then calculates
    the perplexity of the remaining terms in the test files
    """

