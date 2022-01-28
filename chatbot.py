# Construçãodo chatbot com DeepNLP

# Importação das bibliotecas
import numpy as np
import tensorflow as tf
import re
import time

# ---Parte 1 - pré-processamento dos dados -----

# Importação das bases de dados
# Fonte: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

linhas = open('dataset/movie_lines.txt', encoding=('utf-8'), errors=('ignore')).read().split('\n')
conversas = open('dataset/movie_conversations.txt', encoding=('utf-8'), errors=('ignore')).read().split('\n')

# Criação de um dicionário para mapear cada linha com seu ID
id_para_linha = {}
for linha in linhas:
    # print(linha)
    _linha = linha.split(' +++$+++ ')
    # print(_linha)
    if len(_linha) == 5:
        # print(_linha[4])
        id_para_linha[_linha[0]] = _linha[4]
        
# Criação de uma lista com todas as conversas
conversas_id = []
for conversa in conversas[:-1]:
    # print(conversa)
    _conversa = conversa.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    # print(_conversa)
    conversas_id.append(_conversa.split(','))
    
# Separação das perguntas e respostas
