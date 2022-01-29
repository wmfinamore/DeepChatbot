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
    texto = re.sub(r"you're", "you are", texto)
    texto = re.sub(r"he's", "he is", texto)
    texto = re.sub(r"she's", "she is", texto)
    texto = re.sub(r"it's", "it is", texto)
    texto = re.sub(r"we're", "we are", texto)
    texto = re.sub(r"they're", "they are", texto)
    texto = re.sub(r"that's", "that is", texto)
    texto = re.sub(r"what's", "what is", texto)
    texto = re.sub(r"where's", "where is", texto)
    texto = re.sub(r"'ll", " will", texto)
    texto = re.sub(r"'ve", " have", texto)
    texto = re.sub(r"'re", " are", texto)
    texto = re.sub(r"'d", " would", texto)
    texto = re.sub(r"isn't", "is not", texto)
    texto = re.sub(r"aren't", "are not", texto)
    texto = re.sub(r"won't", "will not", texto)
    texto = re.sub(r"can't", "can not", texto)
    texto = re.sub(r"don't", "do not", texto)
    texto = re.sub(r"doesn't", "does not", texto)
    texto = re.sub(r"hasn't", "has not", texto)
    texto = re.sub(r"haven't", "have not", texto)
    texto = re.sub(r"wasn't", "was not", texto)
    texto = re.sub(r"weren't", "were not", texto)
    texto = re.sub(r"didn't", "did not", texto)
    texto = re.sub(r"wouldn't", "would not", texto)
    texto = re.sub(r"there's", "there is", texto)
    texto = re.sub(r"there're", "there are", texto)
    texto = re.sub(r"[-()#/@;:<>{}~+=?!.|,*]", "", texto)
    return texto
    
limpa_texto("Exemplo i'm** that's we'd")

# Limpeza de perguntas
perguntas_limpas = []
for pergunta in perguntas:
    perguntas_limpas.append(limpa_texto(pergunta))

# limpeza das respostas
respostas_limpas = []
for resposta in respostas:
    respostas_limpas.append(limpa_texto(resposta))

# Criação de um dicionário que mapeie cada palavra e o número de ocorrências: pode-se
# user a biblioteca NLTK
palavras_contagem = {}
for pergunta in perguntas_limpas:
    # print(pergunta)
    for palavra in pergunta.split():
        if palavra not in palavras_contagem:
            palavras_contagem[palavra] = 1
        else:
            palavras_contagem[palavra] += 1
            
for resposta in respostas_limpas:
    # print(pergunta)
    for palavra in resposta.split():
        if palavra not in palavras_contagem:
            palavras_contagem[palavra] = 1
        else:
            palavras_contagem[palavra] += 1
















        