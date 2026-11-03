import math
from collections import Counter
import numpy as np

def calculate_entropy(labels: list) -> float:
    _,counts = np.unique(labels, return_counts=True)
    total = len(labels)
    if total==0:
        return 0.0
    probabs = counts/total
    entropy = -np.sum(probabs*np.log2(probabs))
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    parent_labels = [example[target_attr] for example in examples]
    parent_entropy = calculate_entropy(parent_labels)
    partitions = {}
    for example in examples:
        val = example[attr]
        if val not in partitions:
            partitions[val] = []
        partitions[val].append(example)
    total = len(examples)
    weighted_entropy = 0
    for subset in partitions.values():
        subset_labels = [example[target_attr] for example in subset]
        weight = len(subset)/total
        weighted_entropy += weight*calculate_entropy(subset_labels)
    return parent_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    counts = Counter(example[target_attr] for example in examples)
    if not counts:
        return None
    max_count = max(counts.values())
    winners = [cls for cls,count in counts.items() if count==max_count]
    return sorted(winners)[0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    if not examples:
        return None
    labels = [example[target_attr] for example in examples]
    if len(set(labels)) == 1:
        return labels[0]
    if not attributes:
        return majority_class(examples, target_attr)
    best_attr = attributes[0]
    best_gain = calculate_information_gain(examples, best_attr, target_attr)
    for attr in attributes[1:]:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    tree = {best_attr: {}}
    remaining = [a for a in attributes if a != best_attr]
    values = sorted(set(example[best_attr] for example in examples))
    for value in values:
        subset = [example for example in examples if example[best_attr] == value]
        tree[best_attr][value] = learn_decision_tree(subset, remaining, target_attr)
    return tree