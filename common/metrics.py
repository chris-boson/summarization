from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import rouge_score
import re
import six
from typing import List, Dict
from rouge_score import rouge_scorer
#from nltk.corpus import wordnet as wn

class Metrics():
    def __init__(self, config: Dict=None):
        self.config = config
        #self.wordnet = wn
        self.create_rouge_scorer()
    
    def rouge_tokenizer(self, text, stem_limit=3):
        text = text.lower()
        
        # Replace any non-alpha-numeric characters with spaces.
        text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

        tokens = re.split(r"\s+", text)
        
        if self.stemmer is not None:
            tokens = [stemmer.stem(x) if len(x) > stem_limit else x for x in tokens]

        tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]
        
        return tokens
    
    def create_rouge_scorer(self):
        if self.config is None:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=False)
            self.lemmatized_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
        else:
            raise NotImplementedError("Rouge Scorer has not been written yet to handle a config object.")
    
    def score_rouge(self, predictions: List[str], targets: List[str]) -> Dict:
        scores = {}
        rouge_metrics = self.rouge_scorer.rouge_types
        
        #Create rouge metrics for precision, recall and f-measure
        for metric in rouge_metrics:
            scores[metric] = {"precision": np.array([]), "recall": np.array([]), "f1": np.array([])} #precision, recall, f-score
        
        #Get all the scores
        for prediction, target in zip(predictions, targets):
            rouge_scores = self.rouge_scorer.score(prediction, target)
            for metric in rouge_metrics:
                scores[metric]["precision"] = np.append(scores[metric]["precision"], rouge_scores[metric][0])
                scores[metric]["recall"] = np.append(scores[metric]["recall"], rouge_scores[metric][1])
                scores[metric]["f1"] = np.append(scores[metric]["f1"], rouge_scores[metric][2])
        
        #Average them out
        for metric in rouge_metrics:
            scores[metric]["precision"] = scores[metric]["precision"].mean()
            scores[metric]["recall"] = scores[metric]["recall"].mean()
            scores[metric]["f1"] = scores[metric]["f1"].mean()
        return scores

    def score_hypernym_rouge(self, predictions: List[str], targets: List[str]):
        pass
#         for prediction, target in zip(predictions, targets):
#             for word in prediction:
#                 print("potato")
    
    def score(self, predictions: List[str], targets: List[str]) -> Dict:
        scores = {}
        rouge_scores = self.score_rouge(predictions, targets)
        return rouge_scores
