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
perguntas = []
respostas = []
for conversa in conversas_id:
    #print(conversa)
    #print('******************')
    for i in range(len(conversa)-1):
        #print(i)
        perguntas.append(id_para_linha[conversa[i]])
        respostas.append(id_para_linha[conversa[i+1]])
        
def limpa_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"i'm", "i am", texto)
    texto = re.sub(r"he's", "he is", texto)
    texto = re.sub(r"she's", "she is", texto)
    texto = re.sub(r"that´s", "that is", texto)
    texto = re.sub(r"what's", "what is", texto)
    texto = re.sub(r"where's", "where is", texto)
    texto = re.sub(r"/'ll", "will", texto)
    texto = re.sub(r"/'ve", "have", texto)
    texto = re.sub(r"/'re", "are", texto)
    texto = re.sub(r"/'d", "would", texto)
    texto = re.sub(r"won't", "will not", texto)
    texto = re.sub(r"cant't", "can not", texto)
    texto = re.sub(r"[-()#/@;:<>{}~+=?.|,*]", "", texto)
    return texto
    
limpa_texto("Exemplo i'm**")

# Limpeza de perguntas
perguntas_limpas = []
for pergunta in perguntas:
    perguntas_limpas.append(limpa_texto(pergunta))

# limpeza das respostas
respostas_limpas = []
for resposta in respostas:
    respostas_limpas.append(limpa_texto(resposta))



















        