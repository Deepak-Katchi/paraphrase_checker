#tokenizer
from transformers import glue_convert_examples_to_features
import config
import tensorflow as tf
import numpy as np
class TokenisedDataset():
    def __init__(self, a_sentences, b_sentences,labels,ids):
        self.a_sentences = a_sentences
        self.b_sentences = b_sentences
        self.labels = labels
        self.tokenizer = config.TOKENIZER #config.TOKENIZER
        self.max_len = config.MAX_LEN #config.MAX_LEN
        self.idx = ids

    def tokenize_data(self):
      dataset=tf.data.Dataset.from_tensor_slices((self.a_sentences,
                                          self.b_sentences,self.labels,self.idx)).map(lambda a,b,c,d : {'sentence1':a,'sentence2':b,'label':c,'idx':d})
      tokenised_dataset = glue_convert_examples_to_features(dataset, self.tokenizer, self.max_len, 'mrpc')
      return tokenised_dataset
      
