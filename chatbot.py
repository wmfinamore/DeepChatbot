# Construçãodo chatbot com DeepNLP

# Importação das bibliotecas
import numpy as np
import tensorflow as tf
import re
import time

# ---Parte 1 - pré-processamento dos dados -----

# Importação das bases de dados

linhas = open('dataset/movie_lines.txt', encoding=('utf-8'), errors=('ignore')).read().split('\n')
conversas = open('dataset/movie_conversations.txt', encoding=('utf-8'), errors=('ignore')).read().split('\n')

