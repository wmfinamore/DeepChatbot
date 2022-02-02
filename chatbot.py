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

# Remoção de palavras não frequentes e tokenização (dois dicionários)
# recomendação é retirar as palavras 5% menos frequentes
limite = 20
perguntas_palavras_int = {}
numero_palavra = 0
for  palavra, contagem in palavras_contagem.items():
    if contagem >= limite:
        perguntas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1
        
respostas_palavras_int = {}
numero_palavra = 0
for  palavra, contagem in palavras_contagem.items():
    if contagem >= limite:
        respostas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1

# Adição de tokens no dicionário
tokens= ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    perguntas_palavras_int[token] = len(perguntas_palavras_int) + 1
    
for token in tokens:
    respostas_palavras_int[token] = len(respostas_palavras_int) + 1

#Criação do dicionário inverso com o dicionário respostas
respostas_int_palavras = {p_i: p for p, p_i in respostas_palavras_int.items()}

# Adição do token final de string <EOS> para o final de cada resposta
for i in range(len(respostas_limpas)):
    respostas_limpas[i] +=' <EOS>'

# Tradução de todas as perguntas e respotas para inteiros
# Substituição das palavras menos frequentes por <OUT>
perguntas_para_int = []
for pergunta in perguntas_limpas:
    ints = []
    for palavra in pergunta.split():
        if palavra not in perguntas_palavras_int:
             ints.append(perguntas_palavras_int['<OUT>'])
        else:
            ints.append(perguntas_palavras_int[palavra])
    perguntas_para_int.append(ints)

respostas_para_int = []
for resposta in respostas_limpas:
    ints = []
    for palavra in resposta.split():
        if palavra not in respostas_palavras_int:
             ints.append(respostas_palavras_int['<OUT>'])
        else:
            ints.append(respostas_palavras_int[palavra])
    respostas_para_int.append(ints)

# Ordenação das perguntas e respostas pelo tamanho das perguntas
perguntas_limpas_ordenadas = []
respostas_limpas_ordenadas = []
for tamanho in range(1, 25 + 1):
    # print(tamanho)
    for i in enumerate(perguntas_para_int):
        # print(i[1])
        if len(i[1]) == tamanho:
            perguntas_limpas_ordenadas.append(perguntas_para_int[i[0]])
            respostas_limpas_ordenadas.append(respostas_para_int[i[0]])


# ------ Parte 2 --------Construção do modelo Seq2Seq -----
# Criação de placeholder para as entradas e saídas
def entradas_model():
    entradas - tf.placeholder(tf.int32, [None, None], name='entradas')
    saidas = tf.placeholder(tf.int32, [None, None], name='saidas')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return entradas, saidas, lr, keep_prob

# Pré-processamento das saídas (alvos)
def preprocessamento_saidas(saidas, palavra_para_int, batch_size):
    esquerda = tf.fill([batch_size, 1], palavra_para_int['<SOS>'])
    direita = tf.strided_slice(saidas, [0,0], [batch_size, -1], strides= [1,1])
    saidas_preprocessadas = tf.concat([esquerda, direita], 1)
    return saidas_preprocessadas


# Criação da camada RNN do codificador
def rnn_codificador(rnn_entradas, rnn_tamanho, numero_camadas, keep_prob, tamanho_sequencia):
    lstm = tf.contrib.rnn.LSTMCell(rnn_tamanho)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_celula = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * numero_camadas)
    # a função bidirectional_dynamic_rnn retorna dois valores. O primeiro não será usado
    _, encoder_estado = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_celula,
                                                     cell_bw = encoder_celula,
                                                     squence_lenght = tamanho_sequencia,
                                                     inputs = rnn_entradas,
                                                     dtype = tf.float32)
    return encoder_estado


# Decodificador da base de treinamento
def decodifica_base_treinamento(encoder_estado, decodificador_celula,
                                decodificador_embedded_entrada, tamanho_sequencia,
                                decodificador_escopo, funcao_saida, 
                                keep_prob, batch_size):
    estados_atencao = tf.zeros([batch_size, 1, decodificador_celula.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = \
        tf.contrib.seq2seq.prepare_attention(estados_atencao,
                                             attention_option = 'bahdanau',
                                             num_units = decodificador_celula.output_size)
    funcao_decodificador_treinamento = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_estado[0],
                                                                                     attention_keys,
                                                                                     attention_values,
                                                                                     attention_score_function,
                                                                                     attention_construct_function,
                                                                                     name = 'attn_dec_train')
    decodificador_saida, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decodificador_celula,
                                                                       funcao_decodificador_treinamento,
                                                                       decodificador_embedded_entrada,
                                                                       tamanho_sequencia,
                                                                       scope = decodificador_escopo)
    decodificador_saida_dropout = tf.nn.dropout(decodificador_saida, keep_prob)
    return funcao_saida(decodificador_saida_dropout)


# Decodificação da base de teste/validação
def decodifica_base_teste(encoder_estado, decodificador_celula,
                                decodificador_embedding_matrix, sos_id, eos_id, tamanho_maximo,
                                numero_palavras, decodificador_escopo, funcao_saida,
                                keep_prob, batch_size):
    estados_atencao = tf.zeros([batch_size, 1, decodificador_celula.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = \
        tf.contrib.seq2seq.prepare_attention(estados_atencao,
                                             attention_option = 'bahdanau',
                                             num_units = decodificador_celula.output_size)
    funcao_decodificador_teste = tf.contrib.seq2seq.attention_decoder_fn_inference(funcao_saida, 
                                                                                     encoder_estado[0],
                                                                                     attention_keys,
                                                                                     attention_values,
                                                                                     attention_score_function,
                                                                                     attention_construct_function,
                                                                                     decodificador_embedding_matrix,
                                                                                     sos_id, 
                                                                                     eos_id,
                                                                                     tamanho_maximo,
                                                                                     numero_palavras,
                                                                                     name = 'attn_dec_inf')
    previsoes_teste, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decodificador_celula,
                                                                   funcao_decodificador_teste,
                                                                   scope = decodificador_escopo
                                                                   )
    return previsoes_teste







    
















        