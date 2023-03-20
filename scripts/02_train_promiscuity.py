import sys
import os

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "miniautoml"))

from miniautoml.binary_classifier import train_binary_classifier
from fragmentembedding import FragmentEmbedder

emb = 
mdl = train_binary_classifier()