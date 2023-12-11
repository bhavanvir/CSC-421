import os
import string
import re
import numpy as np

# Your code goes here
class BernoulliNaiveBayes:
  def __init__(self, pos_probs, neg_probs):
    self.pos_prob = pos_probs
    self.neg_prob = neg_probs

  def predict(self, X_test):
    preds = []

    # Add a small constant to avoid divide by zero in log10
    epsilon = 1e-10

    # Values are too small to multiply together, so scale using log10 then sum
    pos_probs = [np.log10(self.pos_prob.get(word, epsilon)) for word in X_test]
    neg_probs = [np.log10(self.neg_prob.get(word, epsilon)) for word in X_test]
    pos_given_doc = np.sum(pos_probs)
    neg_given_doc = np.sum(neg_probs)

    pred = 1 if pos_given_doc > neg_given_doc else 0
    return pred

  def evaluate(self, X, y):
    preds = []
    for instance in X:
      preds.append(self.predict(instance))
    correct_preds = np.sum(preds == y)
    accuracy = correct_preds / len(y)

    tp = fp = tn = fn = 0
    for actual, predicted in zip(y, preds):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 1 and predicted == 0:
            fn += 1

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    confusion_matrix = np.array([[tp, fp], [fn, tn]])
    return accuracy, confusion_matrix
  
def clean_string(input_str):
    translator = str.maketrans("", "", string.punctuation)
    cleaned = input_str.translate(translator)
    output_str = re.sub(r'\s+', ' ', cleaned)
    return output_str

def dictionarize_string(input_str):
  dictionary = {}
  words = input_str.split(" ")
  for word in words:
    dictionary[word] = dictionary.get(word, 0) + 1 / len(words)
  return dictionary

def read_file(path):
  all = ""
  for dot_txt in os.listdir(path):
    f = open(path + dot_txt)
    all += f.read() + " "
  return all

def get_random_case():
  pos_or_neg = np.random.choice([0, 1])
  path = pos_path if pos_or_neg == 1 else neg_path

  lib = os.listdir(path)
  choice = np.random.choice(lib)
  f = open(path + choice)

  content = f.read()
  cleaned = clean_string(content).split()
  return cleaned, pos_or_neg

def generate_training():
  pos_set = []
  neg_set = []
  dataset = []

  for pos_txt, neg_txt in zip(os.listdir(pos_path), os.listdir(neg_path)):
    f_pos = open(pos_path + pos_txt)
    f_neg = open(neg_path + neg_txt)
    pos_set.append(clean_string(f_pos.read()).split())
    neg_set.append(clean_string(f_neg.read()).split())

  dataset = pos_set + neg_set
  labels = np.ones(len(dataset), dtype=int)
  labels[len(pos_set):] = 0
  return dataset, labels

def train_test_split(dataset, labels):
  np.random.seed(42)

  num_samples = len(dataset)
  indices = np.arange(num_samples)
  np.random.shuffle(indices)

  dataset = np.array(dataset, dtype=object)

  # 80:20 train-test split
  split_index = int(0.8 * num_samples)
  train_dataset, test_dataset = dataset[indices[:split_index]], dataset[indices[split_index:]]
  train_labels, test_labels = labels[indices[:split_index]], labels[indices[split_index:]]
  return train_dataset, test_dataset, train_labels, test_labels

pos_path = os.getcwd() + '/txt_sentoken/pos/'
neg_path = os.getcwd() + '/txt_sentoken/neg/'

raw_pos, raw_neg = read_file(pos_path), read_file(neg_path)
cleaned_pos, cleaned_neg = clean_string(raw_pos), clean_string(raw_neg)
pos_dict, neg_dict = dictionarize_string(cleaned_pos), dictionarize_string(cleaned_neg)

X_test, y_test = get_random_case()

classifier = BernoulliNaiveBayes(pos_dict, neg_dict)
pred = classifier.predict(X_test)
print(f"Actual: {'Positive' if y_test == 1 else 'Negative'}\nPredicted: {'Positive' if pred == 1 else 'Negative'}")

dataset, labels = generate_training()
X_train, X_test, y_train, y_test = train_test_split(dataset, labels)

train_accuracy, train_confusion_matrix = classifier.evaluate(X_train, y_train)
print("\nTraining Accuracy:", train_accuracy)
print("Training Confusion Matrix:")
print(train_confusion_matrix)
test_accuracy, test_confusion_matrix = classifier.evaluate(X_test, y_test)
print("Testing Accuracy:", test_accuracy)
print("Testing Confusion Matrix:")
print(test_confusion_matrix)