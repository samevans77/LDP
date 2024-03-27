import numpy as np


#  Entropy: see https://datascience.stackexchange.com/questions/58565/conditional-entropy-calculation-in-python-hyx
def entropy(y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(y, return_counts=True, axis=0)
    prob = count / len(y)
    en = np.sum((-1) * prob * np.log2(prob))
    return en


def entropy_count(count, data_length):
    prob = count / data_length
    en = np.sum((-1) * prob * np.log2(prob))
    return en


# Joint Entropy
def j_entropy(y, x):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    yx = np.c_[y, x]
    print("yx")
    print(yx)
    return entropy(yx)


# Conditional Entropy
def c_entropy(y, x):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return j_entropy(y, x) - entropy(x)


# Information Gain
def gain(y, x):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(y) - c_entropy(y, x)
