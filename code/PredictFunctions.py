# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:50:24 2019

@author: mjbigdel
"""

# Imports
from collections import Counter
import codecs
import xml.etree.cElementTree as etree
import pickle
import nltk
nltk.download('wordnet')
import numpy as np
import copy
from nltk.corpus import wordnet as wn
import tensorflow as tf

# Added Chars to dictionaries
UNK = "<UNK>"
PAD = "<PAD>"

# some Basic Hyper parameters
BATCH_SIZE = 4
HIDDEN_LAYER_DIM = 256
LEARNING_RATE = 1
L2_REGU_LAMBDA=0.0001
NUM_LAYERS = 2
CLIP=10
crf_lambda = 0.05
VERY_BIG_NUMBER = 1e30
summaries = []

# -----------------------------------------------------------------------------
def WN_to_BN_dic(mapping_file):
    print('WN_to_BN_dic is started ....')
    BN_ID = []
    WN_ID = []
    with codecs.open(mapping_file,'rb') as f:        
        for line in f:            
            line_synsets = line.decode().strip().split('\t')
            BN_ID.append(line_synsets[0])
            WN_ID.append(line_synsets[1])    
    WN2BN_map_dic = dict(zip(WN_ID,BN_ID))
    print('WN_to_BN_dic is done ....')

    return WN2BN_map_dic


# -----------------------------------------------------------------------------
def BN_to_WNDOMAIN_dic(mapping_file):
    print('WN_to_BN_dic is started ....')
    BN_ID = []
    WN_DOMAIN = []
    with codecs.open(mapping_file,'rb') as f:        
        for line in f:            
            line_synsets = line.decode().strip().split('\t')
            BN_ID.append(line_synsets[0])
            WN_DOMAIN.append(line_synsets[1])    
    BN2WnDomain_dic = dict(zip(BN_ID, WN_DOMAIN))
    print('BN_to_WNDOMAIN_dic is done ....')
    return BN2WnDomain_dic



# -----------------------------------------------------------------------------
def BN_to_LexNames_dic(mapping_file):
    print('BN_to_LexNames is started ....')
    BN_ID = []
    LEXNAMES = []
    with codecs.open(mapping_file,'rb') as f:        
        for line in f:            
            line_synsets = line.decode().strip().split('\t')
            BN_ID.append(line_synsets[0])
            LEXNAMES.append(line_synsets[1])    
    BN2LexNames_dic = dict(zip(BN_ID, LEXNAMES))
    print('BN_to_LexNames_dic is done ....')

    return BN2LexNames_dic



## ----------------------------------------------------------------------------
def getMFS_(word, WN2BN_map_dic):
    MFS_wnId = UNK
    all_synsets = wn.synsets(word)

    # check if the word has synsets or not!
    if len(all_synsets) == 0 or all_synsets is None:
        return word
    
    synset = all_synsets[0]
    MFS_wnId = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
    if WN2BN_map_dic.get(MFS_wnId) is not None:
        MFS_bnId = WN2BN_map_dic.get(MFS_wnId)
        
    return MFS_bnId

#mfs = getMFS_('how')


def getPOSTagOfCharPOS(pos):
    if pos == 'n':
        return 'noun'
    if pos == 'v':
        return 'verb'
    if pos == 'a':
        return 'adj'
    if pos == 'r':
        return 'adv'

def get_WN_POSTagOfCharPOS(pos):
    if pos == 'noun':
        return wn.NOUN
    if pos == 'verb':
        return wn.VERB
    if pos == 'adj':
        return wn.ADJ
    if pos == 'adv':
        return wn.ADV

## ----------------------------------------------------------------------    
def get_bnSenseCandydtsAndPOSOfLemma(lemma, pos_lemma, WN2BN_map_dic):
    synsets_ = []
    pos_ = []
    for synset in wn.synsets(lemma, pos = get_WN_POSTagOfCharPOS(pos_lemma)): # pos=wn.VERB
        synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos() # out: wn:05899087n                
        if WN2BN_map_dic.get(synset_id) is not None:
            bnId_ = WN2BN_map_dic.get(synset_id)
#            print(WN2BN_map_dic.get(synset_id))
            synsets_.append(bnId_)
            pos_.append(getPOSTagOfCharPOS(bnId_[-1:]))

    return synsets_, pos_

#synsets_ = get_bnSenseCandydtsOfLemma('long' ,)

## ----------------------------------------------------------------------    
def get_LexOfLemmaSense(bnIds, BN2LexNames_dic):
    lexs_ = []
    for bnId_ in bnIds:
        if BN2LexNames_dic.get(bnId_) is not None:
            lexs_.append(BN2LexNames_dic.get(bnId_))
        else:
            lexs_.append('other')

    return lexs_

#lexs_ = get_LexOfLemmaSense(synsets_)
    


## ----------------------------------------------------------------------    
def get_WnDomainsOfLemmaSense(bnIds, BN2WnDomain_dic):
    wnDomains_ = []
    for bnId_ in bnIds:
        if BN2WnDomain_dic.get(bnId_) is not None:
            wnDomains_.append(BN2WnDomain_dic.get(bnId_))
        else:
            wnDomains_.append('other')

    return wnDomains_

#wnDomains_ = get_WnDomainsOfLemmaSense(synsets_)




def parse_xml(filename, WN2BN_map_dic, BN2LexNames_dic, BN2WnDomain_dic):  
    print('parse_xml is started ....')   
    
    x_IdInstance_Sent, x_Raw_Sent, y_POS_Sent = [],[],[]
    y_BNsense_Sent, y_Lex_Sent, y_WnDomain_Sent, y_MFS_Sent = [],[],[],[]
    y_allSenses_Sent, y_allPos_Sent, y_allLexs_Sent, y_allwnDomains_Sent = [],[],[],[]
    
    IdInstance_sent, Raw_sent, POS_sent, BN_sent, Lex_sent, WnDomain_sent, MFS_sent,\
                    allSenses_sent, allpos_sent, allLexs_sent, allwnDomains_sent = [],[],[],[],[],[],[],[],[],[],[]
    
    context = etree.iterparse(filename, events=("start", "end"))   
    
    id_sent = 0;

    for event, element in context:            
        if event == "start":        
            if element.tag =='sentence':                
                if id_sent % 5000 == 0:
                    print(id_sent)
                id_sent = id_sent + 1
                
            if element.tag == "wf":  
                id_ = 'None'
                lemma_ = (str(element.attrib['lemma'])).lower()
                pos_ = (str(element.attrib['pos'])).lower()                
                
                IdInstance_sent.append(id_)
                Raw_sent.append(lemma_)            
                POS_sent.append(pos_)
                allSenses_sent.append([lemma_])
                allpos_sent.append([pos_])
                allLexs_sent.append(['other'])
                allwnDomains_sent.append(['other'])
                
            if element.tag == "instance": 
                instanceId_ = (str(element.attrib['id'])).lower()
                lemma_ = (str(element.attrib['lemma'])).lower()
                pos_ = (str(element.attrib['pos'])).lower()
                    
                IdInstance_sent.append(instanceId_)
                Raw_sent.append(lemma_)            
                POS_sent.append(pos_)
                                                
                candSense, CandPos = get_bnSenseCandydtsAndPOSOfLemma(lemma_, pos_, WN2BN_map_dic)
                            
                allSenses_sent.append(candSense)
                allpos_sent.append(CandPos)
                allLexs_sent.append(get_LexOfLemmaSense(candSense, BN2LexNames_dic))
                allwnDomains_sent.append(get_WnDomainsOfLemmaSense(candSense, BN2WnDomain_dic))
                                                                                    
        if event == "end":
            if element.tag == "sentence" :
                x_IdInstance_Sent.append(IdInstance_sent)
                x_Raw_Sent.append(Raw_sent)            
                y_POS_Sent.append(POS_sent)
                y_allSenses_Sent.append(allSenses_sent)
                y_allPos_Sent.append(allpos_sent)
                y_allLexs_Sent.append(allLexs_sent)
                y_allwnDomains_Sent.append(allwnDomains_sent) 
                
                IdInstance_sent, Raw_sent, POS_sent, BN_sent, Lex_sent, WnDomain_sent, MFS_sent,\
                    allSenses_sent, allpos_sent, allLexs_sent, allwnDomains_sent = [],[],[],[],[],[],[],[],[],[],[]
                
        element.clear()
        
    return x_IdInstance_Sent, x_Raw_Sent, y_POS_Sent,\
             y_BNsense_Sent, y_Lex_Sent, y_WnDomain_Sent, y_MFS_Sent,\
                 y_allSenses_Sent, y_allPos_Sent, y_allLexs_Sent, y_allwnDomains_Sent


def get_test_sent_PKL_file(DATASET_FILE, save_path, WN2BN_map_dic, BN2LexNames_dic, BN2WnDomain_dic):                    
    # function for parsing large xml file and getting needed data from that as pkl files

    x_IdInstance_Sent, x_Raw_Sent, y_POS_Sent, y_BNsense_Sent, y_Lex_Sent, y_WnDomain_Sent,\
        y_MFS_Sent, y_allSenses_Sent, y_allPos_Sent, y_allLexs_Sent, y_allwnDomains_Sent\
    = parse_xml(DATASET_FILE, WN2BN_map_dic, BN2LexNames_dic, BN2WnDomain_dic)
    
    print('parse xml is ok .... ')
    with open(save_path + 'x_IdInstance_Sent.pkl', "wb") as outfile:
        pickle.dump(x_IdInstance_Sent, outfile)
        
    with open(save_path + 'x_Raw_Sent.pkl', "wb") as outfile:
        pickle.dump(x_Raw_Sent, outfile)
        
    with open(save_path + 'y_POS_Sent.pkl', "wb") as outfile:
        pickle.dump(y_POS_Sent, outfile)

    with open(save_path + 'y_allSenses_Sent.pkl', "wb") as outfile:
        pickle.dump(y_allSenses_Sent, outfile)
        
    with open(save_path + 'y_allPos_Sent.pkl', "wb") as outfile:
        pickle.dump(y_allPos_Sent, outfile)
        
    with open(save_path + 'y_allLex_Sent.pkl', "wb") as outfile:
        pickle.dump(y_allLexs_Sent, outfile)
        
    with open(save_path + 'y_allWnDomain_Sent.pkl', "wb") as outfile:
        pickle.dump(y_allwnDomains_Sent, outfile)
                
        
    print('PKL file saved successfully ......')


## -----------------------------------------------------------------------------
def get_Raw_data_2id(x_Raw_Sent_pkl, RawWords2id, file_name, resources_path):   
    print('get_Raw_data_2id is started ....')
    
    x_Raw_Sent = pickle.load(open(x_Raw_Sent_pkl, "rb")) 
    x_Raw_Sent_id = [[] for i in range(len(x_Raw_Sent))]        
    
    for i in range(len(x_Raw_Sent)):
        for j in range(len(x_Raw_Sent[i])):                        
            x_Raw_Sent_id[i].append(RawWords2id.get(x_Raw_Sent[i][j], RawWords2id[UNK]))            
            
    with open(resources_path + '/{}/x_Raw_Sent_id.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(x_Raw_Sent_id, outfile)
        
    print('get_Raw_data_2id is done ....')                
#     return x_Raw_Sent_id



## -----------------------------------------------------------------------------
def get_BNSense_data_2id(y_allSenses_Sent_pkl, SenseWords2id, RawWords2id, file_name, resources_path):   
    print('get_BNSense_data_2id is started ....')

    y_allSenses_Sent = pickle.load(open(y_allSenses_Sent_pkl, "rb")) 

    y_allSenses_Sent_id = [[] for i in range(len(y_allSenses_Sent))]    
    
    for i in range(len(y_allSenses_Sent)):
        for j in range(len(y_allSenses_Sent[i])):                                                                                   
            y_allSenses_Sent_id[i].append([])
                        
            for k in range(len(y_allSenses_Sent[i][j])):                                            
                y_allSenses_Sent_id[i][j].append(SenseWords2id.get(y_allSenses_Sent[i][j][k], RawWords2id[UNK]))                  
        
    with open(resources_path + '/{}/y_allSenses_Sent_id.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(y_allSenses_Sent_id, outfile)
        
    print('get_BNSense_data_2id is done ....')                
#     return y_BNsense_Sent_id, y_allSenses_Sent_id




## -----------------------------------------------------------------------------
def get_POS_data_2id(y_POS_Sent_pkl, y_allPos_Sent_pkl, pos2id, file_name, resources_path):   
    print('get_POS_data_2id is started ....')
        
    y_POS_Sent = pickle.load(open(y_POS_Sent_pkl, "rb")) 
    y_allPos_Sent = pickle.load(open(y_allPos_Sent_pkl, "rb"))
    
    y_POS_Sent_id = [[] for i in range(len(y_POS_Sent))]     
    y_allPos_Sent_id = [[] for i in range(len(y_allPos_Sent))]        
    
    for i in range(len(y_POS_Sent)):
        for j in range(len(y_POS_Sent[i])):                                    
            y_POS_Sent_id[i].append(pos2id.get(y_POS_Sent[i][j]))   
            
            y_allPos_Sent_id[i].append([])           
            
            for k in range(len(y_allPos_Sent[i][j])):                                                            
                y_allPos_Sent_id[i][j].append(pos2id.get(y_allPos_Sent[i][j][k]))                

    with open(resources_path + '/{}/y_POS_Sent_id.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(y_POS_Sent_id, outfile)
        
    with open(resources_path + '/{}/y_allPos_Sent_id.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(y_allPos_Sent_id, outfile)
        
    print('get_POS_data_2id is done ....')                
#     return y_POS_Sent_id, y_allPos_Sent_id



## -----------------------------------------------------------------------------
def get_LEX_data_2id(y_allLexs_Sent_pkl, lex2id, file_name, resources_path):   
    print('get_LEX_data_2id is started ....')

    y_allLexs_Sent = pickle.load(open(y_allLexs_Sent_pkl, "rb"))
    
    y_allLexs_Sent_id = [[] for i in range(len(y_allLexs_Sent))]
           
    for i in range(len(y_allLexs_Sent)):
        for j in range(len(y_allLexs_Sent[i])):                                                
            y_allLexs_Sent_id[i].append([])
            
            for k in range(len(y_allLexs_Sent[i][j])):                                                            
                y_allLexs_Sent_id[i][j].append(lex2id.get(y_allLexs_Sent[i][j][k]))                
    
    with open(resources_path + '/{}/y_allLex_Sent_id.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(y_allLexs_Sent_id, outfile)
        
    print('get_LEX_data_2id is done ....')                
#     return y_Lex_Sent_id, y_allLexs_Sent_id


## -----------------------------------------------------------------------------
def get_WnDomain_data_2id(y_allwnDomains_Sent_pkl, wnDomain2id, file_name, resources_path):   
    print('get_WnDomain_data_2id is started ....')
        
    y_allwnDomains_Sent = pickle.load(open(y_allwnDomains_Sent_pkl, "rb"))
    
    y_allwnDomains_Sent_id = [[] for i in range(len(y_allwnDomains_Sent))]
        
    for i in range(len(y_allwnDomains_Sent)):
        for j in range(len(y_allwnDomains_Sent[i])):                                    

            y_allwnDomains_Sent_id[i].append([])
            
            for k in range(len(y_allwnDomains_Sent[i][j])):                                                            
                y_allwnDomains_Sent_id[i][j].append(wnDomain2id.get(y_allwnDomains_Sent[i][j][k]))     
        
    with open(resources_path + '/{}/y_allWnDomain_Sent_id.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(y_allwnDomains_Sent_id, outfile)
        
    print('get_WnDomain_data_2id is done ....')                
#     return y_WnDomain_Sent_id, y_allwnDomains_Sent_id


## ----------------------------------------------------------------------------
def mask_PosToSenses(y_allPos_Sent_pkl, y_allSenses_Sent_id_pkl, pos2id, SenseWords2id, file_name, resources_path):
    print('mask_PosToSenses is started ....')
    
    y_allPos_Sent = pickle.load(open(y_allPos_Sent_pkl, "rb"))
    y_allSenses_Sent_id = pickle.load(open(y_allSenses_Sent_id_pkl, "rb"))
    
    mask_pos2sense = [0]*np.ones(shape = (len(pos2id), len(SenseWords2id)))
    
    for i in range(len(y_allPos_Sent)):
        for j in range(len(y_allPos_Sent[i])):
            for k in range(len(y_allPos_Sent[i][j])):
                if pos2id.get(y_allPos_Sent[i][j][k]) is not None:
                    mask_pos2sense[pos2id[y_allPos_Sent[i][j][k]]][y_allSenses_Sent_id[i][j][k]] = 1                                    
    
    with open(resources_path + '/{}/mask_pos2sense.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(mask_pos2sense, outfile)
        
    print('mask_PosToSenses is done ....')
#     return mask_pos2sense   


## ----------------------------------------------------------------------------
def mask_LexToSenses(y_allLexs_Sent_pkl, y_allSenses_Sent_id_pkl, lex2id, SenseWords2id, file_name, resources_path):
    print('mask_LexToSenses is started ....')
    
    y_allLexs_Sent = pickle.load(open(y_allLexs_Sent_pkl, "rb"))
    y_allSenses_Sent_id = pickle.load(open(y_allSenses_Sent_id_pkl, "rb"))
    
    mask_lex2sense = [0]*np.ones(shape = (len(lex2id), len(SenseWords2id)))
    
    for i in range(len(y_allLexs_Sent)):
        for j in range(len(y_allLexs_Sent[i])):
            for k in range(len(y_allLexs_Sent[i][j])):
                if lex2id.get(y_allLexs_Sent[i][j][k]) is not None:
                    mask_lex2sense[lex2id[y_allLexs_Sent[i][j][k]]][y_allSenses_Sent_id[i][j][k]] = 1    
    
    with open(resources_path + '/{}/mask_lex2sense.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(mask_lex2sense, outfile)
        
    print('mask_LexToSenses is done ....')                                
#     return mask_lex2sense   



## ----------------------------------------------------------------------------
def mask_WnDomansToSenses(y_allwnDomains_Sent_pkl, y_allSenses_Sent_id_pkl, wnDomain2id, SenseWords2id, file_name, resources_path):
    print('mask_WnDomansToSenses is started ....')
    
    y_allwnDomains_Sent = pickle.load(open(y_allwnDomains_Sent_pkl, "rb"))
    y_allSenses_Sent_id = pickle.load(open(y_allSenses_Sent_id_pkl, "rb"))    
    
    mask_wnDomain2sense = [0]*np.ones(shape = (len(wnDomain2id), len(SenseWords2id)))
    
    for i in range(len(y_allwnDomains_Sent)):
        for j in range(len(y_allwnDomains_Sent[i])):
            for k in range(len(y_allwnDomains_Sent[i][j])):
                if wnDomain2id.get(y_allwnDomains_Sent[i][j][k]) is not None:
                    mask_wnDomain2sense[wnDomain2id[y_allwnDomains_Sent[i][j][k]]][y_allSenses_Sent_id[i][j][k]] = 1    
    
    with open(resources_path + '/{}/mask_wnDomain2sense.pkl'.format(file_name), "wb") as outfile:
        pickle.dump(mask_wnDomain2sense, outfile)
    
    print('mask_WnDomansToSenses is done ....')    
#     return mask_wnDomain2sense   

# ===-----------------------------------------------------------------------===
# Trainig Section
# ===-----------------------------------------------------------------------===
# function for add padding for train_y in every batch based on maximum length of sentence in that batch
# this method boost the performance of running.
def padding(X,padding_word):
	max_len = 0
	for x in X:
		if len(x) > max_len:
			max_len = len(x)
	padded_X = np.ones((len(X), max_len), dtype=np.int32) * padding_word
	for i in range(len(X)):
		for j in range(len(X[i])):
			padded_X[i, j] = X[i][j]
	return padded_X


# padding function for train_x
def padding3(X,padding_word):
	max_len = 0
	for x in X:
		if len(x) > max_len:
			max_len = len(x)
	padded_X = np.ones((len(X), max_len), dtype=np.int32) * padding_word
	for i in range(len(X)):
		for j in range(len(X[i])):
			padded_X[i, j] = X[i][j]
	return padded_X

## ----------------------------------------------------------------------------
def mask_lemmaToSenses_batch(x_Raw_Sent_batch, y_allSenses_Sent_batch, y_allSenses_Sent_id_batch, SenseWords2id):
     
    max_len = 0    
    for inp in x_Raw_Sent_batch:
        if len(inp) > max_len:
            max_len = len(inp)
            
    mask_lemma2sense_batch = [0]*np.ones(shape = (len(x_Raw_Sent_batch), max_len, len(SenseWords2id)))
            
    for i in range(len(x_Raw_Sent_batch)):
        for j in range(len(x_Raw_Sent_batch[i])):
            for k in range(len(y_allSenses_Sent_batch[i][j])):            
                mask_lemma2sense_batch[i][j][y_allSenses_Sent_id_batch[i][j][k]] = 1    
            
    return mask_lemma2sense_batch    

def reading_pretrained_sense_embeddings(filename, RawWords2id):
    # Reading Pretrained Embeddings from file
    pretrain_embeddigs = {}
    with codecs.open(filename, "r", "utf-8") as f:
        for line in f:
            pre_train = line.split()
            if len(pre_train) > 2:
                word = pre_train[0]
                if word in RawWords2id:
                    vec = pre_train[1:]
                    pretrain_embeddigs[word] = vec                
    
    print("pretrained embeddings files reading finished ...")
    # making embeddings for all RawWords2id.
    embedding_dim = len(next(iter(pretrain_embeddigs.values())))
    out_of_vocab = 0
    out = np.ones((len(RawWords2id), embedding_dim))*0.001
    for word in RawWords2id.keys():
        if len(word) == 1:
            if word in pretrain_embeddigs.keys():        
                out[RawWords2id[word]]=np.array(pretrain_embeddigs[word])
            else:                
                out_of_vocab+=1
                np.random.uniform(-1.0, 1.0, embedding_dim)
        
    return out,out_of_vocab



# --------------------- Tensorflow part ---------------------------------------------------
def create_tensorflow_model(vocab_size, embedding_dim, hidden_layer_dim, sense_embed_bool, total_loss,\
                            NUM_CLASSES, num_sense_pos, num_sense_lex, num_sense_wnDomain, PAD_ID, sense_embeddings):
    print("Creating TENSORFLOW model")
     
    # Inputs have (batch_size, timesteps) shape.
    input_ = tf.placeholder(dtype = tf.int32, shape=[None, None], name='input_x')   
    # Labels have (batch_size,) shape.
    labels = tf.placeholder(dtype = tf.int32, shape=[None, None], name='labels_BN')
    
    y_pos = tf.placeholder(dtype = tf.int32, shape=[None, None], name='pos_tag')
        
    y_wnDomain = tf.placeholder(dtype = tf.int32, shape=[None, None],name='wnDomain_tag')        
    
    # dropout_keep_prob is a scalar.
    dropout_keep_prob = tf.placeholder(dtype = tf.float32, name='dropout_keep_prob')
    
    seq_length = tf.reduce_sum(tf.cast( tf.not_equal(input_[:,:], tf.ones_like(input_[:,:] ) * PAD_ID ), tf.int32), 1)    
    print('seq_length: ', seq_length)
    
    max_sent_size = tf.size(input_[0,:])    
    print('max_sent_size: ', max_sent_size)
    
    batch_size = tf.size(input_[:,0])    
    print('batch_size: ', batch_size)
            
    mask_RawTosense_batch = tf.placeholder(dtype = tf.bool, shape=[None, None, None], name='mask_Raw2Senses')    
    print('mask_RawTosense_batch: ', mask_RawTosense_batch)
    
    mask_pos2sense = tf.placeholder(dtype = tf.bool, shape=[None, None], name='mask_Raw2Senses')    
    print('mask_pos2sense: ', mask_pos2sense)
    mask_matrix_pos = tf.tile(tf.expand_dims(mask_pos2sense, 0), [batch_size, 1, 1])     
    print('mask_matrix_pos: ', mask_matrix_pos)
    
    mask_wnDomain2sense = tf.placeholder(dtype = tf.bool, shape=[None, None], name='mask_Raw2Senses')    
    print('mask_wnDomain2sense: ', mask_wnDomain2sense)
    mask_matrix_wnDomain = tf.tile(tf.expand_dims(mask_wnDomain2sense, 0), [batch_size, 1, 1])     
    print('mask_matrix_wnDomain: ', mask_matrix_wnDomain)
    
    x_mask = tf.not_equal(input_[:,:], tf.ones_like(input_[:,:] ) * PAD_ID )    
    print('x_mask: ', x_mask)
    
    attention_mask = (tf.cast(x_mask, 'float') -1) * VERY_BIG_NUMBER     
    print('attention_mask: ', attention_mask)

    # initialize weights randomly from a Gaussian distribution
    weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            
## ---------------------------- embedding Block --------------------------------------------
## -----------------------------------------------------------------------------------------  
    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
        if sense_embed_bool:
            embedding_matrix = tf.Variable(sense_embeddings, dtype=tf.float32, name='embedding')
        else:
            embedding_matrix = tf.get_variable("embeddings", shape=[vocab_size, embedding_dim])
        
        embeddings = tf.nn.embedding_lookup(embedding_matrix, input_)        
        print('embeddings: ', embeddings)
        
        embeddings = tf.reshape(embeddings,[batch_size, max_sent_size, embedding_dim]) #         
        print('embeddings: ', embeddings)

        embeddings=tf.nn.dropout(tf.cast(embeddings, tf.float32), dropout_keep_prob) #     embeddings shape (batch size, sentence length with padding, 100) 

        print ('embeddings is ok \n')
                
    
## --------------------------- POS Block ---------------------------------------------------
## -----------------------------------------------------------------------------------------  
    with tf.variable_scope('rnn_cell_pos', reuse=tf.AUTO_REUSE):
            print ('lstm_cell_pos is is ok ... ')
            def lstm_cell1():
                return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_layer_dim), output_keep_prob=dropout_keep_prob)

            stacked_fw_lstm_pos = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell1() for _ in range(NUM_LAYERS)])

            stacked_bw_lstm_pos = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell1() for _ in range(NUM_LAYERS)])

    with tf.variable_scope('rnn_pos', reuse=tf.AUTO_REUSE):                        
        (forward_output_pos, backword_output_pos), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = stacked_fw_lstm_pos,
            cell_bw = stacked_bw_lstm_pos,
            inputs = embeddings,
            sequence_length = seq_length,
            dtype=tf.float32
        )                        
        outputBD_pos = tf.concat([forward_output_pos, backword_output_pos], axis=2)
        print('outputBD_pos: ', outputBD_pos) # shape is batch*max_len*(2*hidden_layer_size)                         

        print ('outputBD_pos is ok \n')

    with tf.variable_scope("softmax_layer_pos", reuse=tf.AUTO_REUSE):
        W_pos = tf.get_variable("W_pos", shape=[2*hidden_layer_dim, num_sense_pos], initializer = weight_initer)
        b_pos = tf.get_variable("b_pos", shape=[num_sense_pos], initializer=tf.zeros_initializer())

        flat_softmax_pos = tf.reshape(outputBD_pos, [-1, tf.shape(outputBD_pos)[2]]) # shape is (batch*max_len)*(2*hidden_layer_size)
        print('flat_softmax_pos: ', flat_softmax_pos)

        drop_flat_softmax_pos = tf.nn.dropout(flat_softmax_pos, dropout_keep_prob) # shape is (batch*max_len)*(2*hidden_layer_size)
        print('drop_flat_softmax_pos: ', drop_flat_softmax_pos)

        flat_pos_logits = tf.matmul(drop_flat_softmax_pos, W_pos) + b_pos # shape is (batch*max_len)*num_sense_pos
        print('flat_pos_logits: ', flat_pos_logits)

        pos_logits = tf.reshape(flat_pos_logits, [batch_size, max_sent_size, num_sense_pos]) # shape is batch*max_len*num_sense_pos ->3D
        print('pos_logits: ', pos_logits)

        predict_pos = tf.argmax(pos_logits , axis=2)

        loss_pos = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y_pos, logits= pos_logits))

        total_loss = total_loss + loss_pos

        float_mask_matrix_pos = tf.cast(mask_matrix_pos, dtype=tf.float32)
        print('float_mask_matrix_pos: ', float_mask_matrix_pos)

        pos_masked_senses = tf.matmul(pos_logits, float_mask_matrix_pos) #mask_mat = 2*11*56 or batch*pos_num*NUM_CLASSES
        print('pos_masked_senses: ', pos_masked_senses)  # size is: batch*max_len*NUM_CLASSES

        print ('softmax_layer_pos is ok \n')


## --------------------------- wnDomain Block ------------------------------------------------
## -------------------------------------------------------------------------------------------
    with tf.variable_scope('rnn_cell_wnDomain', reuse=tf.AUTO_REUSE):
        print ('lstm_cell_wnDomain is is ok ... ')
        def lstm_cell1():
            return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_layer_dim), output_keep_prob=dropout_keep_prob)

        stacked_fw_lstm_wnDomain = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell1() for _ in range(NUM_LAYERS)])

        stacked_bw_lstm_wnDomain = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell1() for _ in range(NUM_LAYERS)])

    with tf.variable_scope('rnn_wnDomain', reuse=tf.AUTO_REUSE):                        
            (forward_output_wnDomain, backword_output_wnDomain), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = stacked_fw_lstm_wnDomain,
                cell_bw = stacked_bw_lstm_wnDomain,
                inputs = outputBD_pos,
                sequence_length = seq_length,
                dtype=tf.float32
            )

            outputBD_wnDomain = tf.concat([forward_output_wnDomain, backword_output_wnDomain], axis=2)
            print('outputBD_wnDomain: ', outputBD_wnDomain) # shape is batch*max_len*(2*hidden_layer_size) 

            print ('outputBD_wnDomain is ok \n')         

    with tf.variable_scope("softmax_layer_wnDomain", reuse=tf.AUTO_REUSE):
        W_wnDomain = tf.get_variable("W_wnDomain", shape=[2*hidden_layer_dim, num_sense_wnDomain], initializer = weight_initer)
        b_wnDomain = tf.get_variable("b_wnDomain", shape=[num_sense_wnDomain], initializer=tf.zeros_initializer())

        flat_softmax_wnDomain = tf.reshape(outputBD_wnDomain, [-1, tf.shape(outputBD_wnDomain)[2]]) # shape is (batch*max_len)*(2*hidden_layer_size)
        print('flat_softmax_wnDomain: ', flat_softmax_wnDomain)

        drop_flat_softmax_wnDomain = tf.nn.dropout(flat_softmax_wnDomain, dropout_keep_prob) # shape is (batch*max_len)*(2*hidden_layer_size)
        print('drop_flat_softmax_wnDomain: ', drop_flat_softmax_wnDomain)

        flat_wnDomain_logits = tf.matmul(drop_flat_softmax_wnDomain, W_wnDomain) + b_wnDomain # shape is (batch*max_len)*num_sense_lex
        print('flat_wnDomain_logits: ', flat_wnDomain_logits)

        wnDomain_logits = tf.reshape(flat_wnDomain_logits, [batch_size, max_sent_size, num_sense_wnDomain]) # shape is batch*max_len*num_sense_lex ->3D
        print('wnDomain_logits: ', wnDomain_logits)

        predict_wnDomain = tf.argmax(wnDomain_logits , axis=2)

        loss_wnDomain = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y_wnDomain, logits= wnDomain_logits))
        total_loss = total_loss + loss_wnDomain

        float_mask_matrix_wnDomain = tf.cast(mask_matrix_wnDomain, dtype=tf.float32)
        print('float_mask_matrix_wnDomain: ', float_mask_matrix_wnDomain)

        wnDomain_masked_senses = tf.matmul(wnDomain_logits, float_mask_matrix_wnDomain) #mask_mat = 2*11*56 or batch*lex_num*NUM_CLASSES
        print('wnDomain_masked_senses: ', wnDomain_masked_senses)  # size is: batch*max_len*NUM_CLASSES

        print ('softmax_layer_wnDomain is ok \n')                         

    input_rnn_sense = outputBD_wnDomain

        
## --------------------------- SENSE Block ------------------------------------------------   
## ----------------------------------------------------------------------------------------   
    with tf.variable_scope('rnn_cell_sense', reuse=tf.AUTO_REUSE):
            print ('lstm_sense is ok')
            def lstm_cell():
                return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_layer_dim), output_keep_prob=dropout_keep_prob)
        
            stacked_fw_lstm_Sense = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(NUM_LAYERS)])
            
            stacked_bw_lstm_Sense = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(NUM_LAYERS)])
            
    with tf.variable_scope('rnn_sense', reuse=tf.AUTO_REUSE):                        
            (forward_output_Sense, backword_output_Sense), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = stacked_fw_lstm_Sense,
                cell_bw = stacked_bw_lstm_Sense,
                inputs = input_rnn_sense,
                sequence_length = seq_length,
                dtype=tf.float32
            )
            
            outputBD_Sense = tf.concat([forward_output_Sense, backword_output_Sense], axis=2)            
            print('outputBD_Sense: ', outputBD_Sense) # shape is batch*max_len*(2*hidden_layer_size) 
            
            print ('outputBD_Sense is ok \n')   

            
## --------------------------- attention_layer --------------------------------------------------
    with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
        W_attention_L = tf.get_variable("W_attention_L", shape=[2*hidden_layer_dim, 1], initializer = weight_initer )
        flat_outputBD_Sense = tf.reshape(outputBD_Sense, [batch_size*max_sent_size, tf.shape(outputBD_Sense)[2]])        
        print('flat_outputBD_Sense: ', flat_outputBD_Sense) # shape is  shape is (batch*max_length*(2*hidden_layer_size)).(51(2*hidden_layer_size)2*1) = (batch*max_length)*1 ->2D
        
        flat_outputBD_Sense_tanh = tf.tanh(flat_outputBD_Sense)
        
        u_flat_outputBD_Sense_tanh = tf.matmul(flat_outputBD_Sense_tanh, W_attention_L) # shape is batch*max_length -> 2D
        print('u_flat: ', u_flat_outputBD_Sense_tanh)
        
        u = tf.reshape(u_flat_outputBD_Sense_tanh, [batch_size, max_sent_size]) + attention_mask #
        print('u: ', u)
        
        u_softmax = tf.nn.softmax(u, 1)
        
        a = tf.expand_dims(u_softmax, 2) # shape is expand to batch*max_len*1 -> 3D
        print('a: ', a)
        
        c = tf.reduce_sum(tf.multiply(outputBD_Sense, a), axis=1) # shape is batch*max_len*(2*hidden_layer_size) and then reduce_sum with axis 1 to: batch*(2*hidden_layer_size)
        print('c: ', c)
        
        tiled_c = tf.tile(tf.expand_dims(c, 1), [1, max_sent_size, 1]) # shape is batch*max_len*(2*hidden_layer_size)
        print('tiled_c: ', tiled_c)
        
        attention_output = tf.concat([tiled_c, outputBD_Sense], 2) # batch*max_len*(2*hidden_layer_size) "concat with" batch*max_len*(2*hidden_layer_size) -> batch*max_len*(4*hidden_layer_size)
        print('attention_output: ', attention_output)
        
        flat_attention_output = tf.reshape(attention_output, [batch_size*max_sent_size, tf.shape(attention_output)[2]]) # reshape to (batch*max_len)*(4*hidden_layer_size) -> 2D
        print('flat_attention_output: ', flat_attention_output)

        print ('global_attention is ok \n') 
        
## --------------------------- hidden_layer Sense ------------------------------------------------------------
    with tf.variable_scope("hidden_layer", reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=[4*hidden_layer_dim, 2*hidden_layer_dim], initializer = weight_initer)
        b = tf.get_variable("b", shape=[2*hidden_layer_dim], initializer = tf.zeros_initializer())
        
        drop_flat_attention_output = tf.nn.dropout(flat_attention_output, dropout_keep_prob) # shape is (batch*max_len)*(4*hidden_layer_size) -> 2D
        print('drop_flat_attention_output: ', drop_flat_attention_output)
        
        hidden_layer_output = tf.matmul(drop_flat_attention_output, W) + b  # shape is (batch*max_len)*(2*hidden_layer_size) -> 2D
        print('hidden_layer_output: ', hidden_layer_output)
                
        print ('hidden_layer is ok \n') 
        
## --------------------------- softmax_layer Sense ------------------------------------------------------------
    with tf.variable_scope("softmax_layer", reuse=tf.AUTO_REUSE):
        W_sense = tf.get_variable("W_sense", shape=[2*hidden_layer_dim, NUM_CLASSES], initializer = weight_initer)
        b_sense = tf.get_variable("b_sense", shape=[NUM_CLASSES], initializer=tf.zeros_initializer())
        
        drop_hidden_layer_output = tf.nn.dropout(hidden_layer_output, dropout_keep_prob)
        print('drop_hidden_layer_output: ', drop_hidden_layer_output)
        
        flat_sense_logits = tf.matmul(drop_hidden_layer_output, W_sense) + b_sense
        print('flat_sense_logits: ', flat_sense_logits)
        
        sense_logits = tf.reshape(flat_sense_logits, [batch_size, max_sent_size, NUM_CLASSES])
        print('sense_logits: ', sense_logits)

        masked_sense_logits = tf.multiply(sense_logits, tf.multiply(pos_masked_senses, wnDomain_masked_senses))
        print('masked_sense_logits: ', masked_sense_logits)  

            
        float_mask_RawTosense_batch = tf.cast(mask_RawTosense_batch, dtype=tf.float32)
        print('float_mask_RawTosense_batch: ', float_mask_RawTosense_batch)
        
        final_sense_logits = tf.multiply(masked_sense_logits, float_mask_RawTosense_batch)
        print('final_sense_logits: ', final_sense_logits)

        predictions = tf.argmax(final_sense_logits , axis=2)
        print('predictions: ', predictions)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels, logits= final_sense_logits))
        total_loss = total_loss + loss
        print ('softmax_layer is ok \n')
        
              
## --------------------------- train_op Block ----------------------------------------------
## -----------------------------------------------------------------------------------------          
    with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
                        
        optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE)
        
#         optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
#         optimizer=tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE, use_locking=False, name='GradientDescent')
        print ('AdamOptimizer is ok .... \n')
        
        tvars=tf.trainable_variables()
        print ('tvars is ok ....')
    
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        print ('l2_loss is ok ....')
        
        total_loss = total_loss + L2_REGU_LAMBDA*l2_loss             
        print ('total_loss is ok ....')
        
        summaries.append(tf.summary.scalar("loss", loss))
        summaries.append(tf.summary.scalar("total_loss", total_loss))
        
        grads,_ = tf.clip_by_global_norm(tf.gradients(total_loss,tvars),CLIP)
        print ('grads is ok ....')
        
        train_op = optimizer.apply_gradients(zip(grads,tvars))
        print ('train_op apply_gradients is ok ....')
                               
    return input_, labels, y_pos, y_wnDomain, mask_RawTosense_batch, train_op, predictions, dropout_keep_prob, total_loss, seq_length, mask_pos2sense, mask_wnDomain2sense



def predict(input_path, output_path, resources_path, output_format):    
    
    MAP_WN2BN_FILE = resources_path + '/babelnet2wordnet.tsv'
    MAP_BN2WNDOMAIN_FILE = resources_path + '/babelnet2wndomains.tsv'
    MAP_BN2LEXNAMES_FILE = resources_path + '/babelnet2lexnames.tsv'
            
    WN2BN_map_dic = WN_to_BN_dic(MAP_WN2BN_FILE)
    BN2WnDomain_dic = BN_to_WNDOMAIN_dic(MAP_BN2WNDOMAIN_FILE)
    BN2LexNames_dic = BN_to_LexNames_dic(MAP_BN2LEXNAMES_FILE)
    
    DATASET_FILE = input_path
    PATH = resources_path + '/secret_test_pkl/'
    
    get_test_sent_PKL_file(DATASET_FILE, PATH, WN2BN_map_dic, BN2LexNames_dic, BN2WnDomain_dic)
    
    all_dict_data = pickle.load(open(resources_path + '/all_dict_data.pkl', "rb"))
    
    RawWords2id = all_dict_data["RawWords2id"]
    SenseWords2id = all_dict_data["SenseWords2id"] 
    pos2id = all_dict_data["pos2id"] 
    lex2id = all_dict_data["lex2id"] 
    wnDomain2id = all_dict_data["wnDomain2id"]
    
    file_name = 'secret_test_pkl'
    
    get_Raw_data_2id(resources_path + '/{}/x_Raw_Sent.pkl'.format(file_name), RawWords2id, file_name, resources_path)
    get_BNSense_data_2id(resources_path + '/{}/y_allSenses_Sent.pkl'.format(file_name), SenseWords2id, RawWords2id, file_name, resources_path)
    get_POS_data_2id(resources_path + '/{}/y_POS_Sent.pkl'.format(file_name), resources_path + '/{}/y_allPos_Sent.pkl'.format(file_name), pos2id, file_name, resources_path)
    get_LEX_data_2id(resources_path + '/{}/y_allLex_Sent.pkl'.format(file_name), lex2id, file_name, resources_path)
    get_WnDomain_data_2id(resources_path + '/{}/y_allWnDomain_Sent.pkl'.format(file_name), wnDomain2id, file_name, resources_path)
    print('processing is done for -----> ', file_name)
    
    
    mask_PosToSenses(resources_path + '/{}/y_allPos_Sent.pkl'.format(file_name), resources_path + '/{}/y_allSenses_Sent_id.pkl'.format(file_name), pos2id, SenseWords2id, file_name, resources_path)
    mask_LexToSenses(resources_path + '/{}/y_allLex_Sent.pkl'.format(file_name), resources_path + '/{}/y_allSenses_Sent_id.pkl'.format(file_name), lex2id, SenseWords2id, file_name, resources_path)
    mask_WnDomansToSenses(resources_path + '/{}/y_allWnDomain_Sent.pkl'.format(file_name), resources_path + '/{}/y_allSenses_Sent_id.pkl'.format(file_name), wnDomain2id, SenseWords2id, file_name, resources_path)
    print('processing is ok for -----> ', file_name)    
    
    PAD_ID = RawWords2id[PAD]
    id2Sensewords = dict(zip(SenseWords2id.values(), SenseWords2id.keys()))
    
    sense_embeddings, out_of_vocab = reading_pretrained_sense_embeddings(resources_path + '/allWordsSensesembeddings.vec', RawWords2id)
    print('sense_embeddings: ', len(sense_embeddings))
    print('out_of_vocab: ', out_of_vocab)
    
    # we are using pretrained embedding - len(words2id)
    VOCAB_SIZE =  len(RawWords2id)
    WORD_EMBEDDING_DIM = 100
    print('WORD_EMBEDDING_DIM: ', WORD_EMBEDDING_DIM)
    print('VOCAB_SIZE: ', VOCAB_SIZE)
        
    NUM_CLASSES = len(SenseWords2id)
    print('num sense classes: ', NUM_CLASSES)
    num_sense_pos = len(pos2id)
    print('num_sense_pos: ', num_sense_pos)
    num_sense_lex = len(lex2id)
    print('num_sense_lex: ', num_sense_lex)
    num_sense_wnDomain = len(wnDomain2id)
    print('num_sense_wnDomain: ', num_sense_wnDomain)
    total_loss = 0
    sense_embed_bool = True
    
    # create tensorflow model with sense embedding BDLSTM's
    input_, labels, y_pos, y_wnDomain, mask_RawTosense_batch, train_op, predictions, dropout_keep_prob,\
    total_loss, seq_length, mask_pos2sense, mask_wnDomain2sense = create_tensorflow_model(VOCAB_SIZE,\
      WORD_EMBEDDING_DIM, HIDDEN_LAYER_DIM, sense_embed_bool, total_loss, NUM_CLASSES, num_sense_pos,\
      num_sense_lex, num_sense_wnDomain, PAD_ID, sense_embeddings)

    
    # model loading, prediction and write part #Restore Saved Tensorflow model.
    file_name = 'secret_test_pkl'
    test_path = resources_path + '/{}/'.format(file_name)
    
    x_test_IdInstance_Sent = pickle.load(open(test_path + 'x_IdInstance_Sent.pkl', "rb"))
    x_test_Raw_Sent = pickle.load(open(test_path + 'x_Raw_Sent.pkl', "rb"))
    x_test_Raw_Sent_id = pickle.load(open(test_path + 'x_Raw_Sent_id.pkl', "rb"))
    y_test_allSenses_Sent = pickle.load(open(test_path + 'y_allSenses_Sent.pkl', "rb"))
    y_test_allSenses_Sent_id = pickle.load(open(test_path + 'y_allSenses_Sent_id.pkl', "rb"))    
    
    mask_test_wnDomain2sense = pickle.load(open(test_path + 'mask_wnDomain2sense.pkl', "rb"))
    mask_test_pos2sense = pickle.load(open(test_path + 'mask_pos2sense.pkl', "rb"))
        
    saver = tf.train.Saver()
    test_pred_BN = []
    test_pred_Lex = []
    test_pred_WnDomain = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:        
        
        saver.restore(sess, resources_path + '/POS_WnDomain_SenseEmbed_models_49.43566591422122/model.ckpt')    
        print("Model restored.")         
        
        for i in range(0, len(x_test_Raw_Sent_id), BATCH_SIZE):                            
            batch_x_Lemma = x_test_Raw_Sent[slice(i, i + BATCH_SIZE)]
            batch_x_Raw = x_test_Raw_Sent_id[slice(i, i + BATCH_SIZE)]
    
            y_test_allSenses_Sent_batch = y_test_allSenses_Sent[slice(i, i + BATCH_SIZE)]
            y_test_allSenses_Sent_id_batch = y_test_allSenses_Sent_id[slice(i, i + BATCH_SIZE)]
    
            mask_lemma2Sense_b = mask_lemmaToSenses_batch(batch_x_Raw, y_test_allSenses_Sent_batch, y_test_allSenses_Sent_id_batch, SenseWords2id)
    
            batch_x_Raw = padding3(batch_x_Raw, PAD_ID)
    
            lengths, predict = sess.run(
                    [seq_length, predictions], feed_dict = {input_ : batch_x_Raw, mask_RawTosense_batch : mask_lemma2Sense_b, mask_pos2sense: mask_test_pos2sense ,\
                                                                        mask_wnDomain2sense: mask_test_wnDomain2sense, dropout_keep_prob: 1.0 })            
    
            for sent_num in range(len(predict)):
                pr_BN, pr_Lex, pr_WnDomain = [], [], []
                for L_ in range(lengths[sent_num]):
                    # if our prediction is UNK means that word is not in our dictionary and we can't predict anything for that, so we put BackOff plan and MFS as our prediction.
                    if id2Sensewords[predict[sent_num][L_]] != UNK:
                        pr_BN.append(id2Sensewords[predict[sent_num][L_]])
                        pr_Lex.append(BN2LexNames_dic.get(id2Sensewords[predict[sent_num][L_]])) # BN2LexNames_dic
                        pr_WnDomain.append(BN2WnDomain_dic.get(id2Sensewords[predict[sent_num][L_]])) # BN2WnDomain_dic
                    else:
                        pr_BN.append(getMFS_(batch_x_Lemma[sent_num][L_], WN2BN_map_dic))
                        pr_Lex.append(BN2LexNames_dic.get(getMFS_(batch_x_Lemma[sent_num][L_], WN2BN_map_dic)))
                        pr_WnDomain.append(BN2WnDomain_dic.get(getMFS_(batch_x_Lemma[sent_num][L_], WN2BN_map_dic)))
    
                test_pred_BN.append(pr_BN)
                test_pred_Lex.append(pr_Lex)
                test_pred_WnDomain.append(pr_WnDomain)
        
        print('predict files successfully saved')
        
    if output_format == 'BabelNet':
        bnId_file = open(output_path, "w")
        for i in range(len(x_test_IdInstance_Sent)):
            for j in range(len(x_test_IdInstance_Sent[i])):
                if x_test_IdInstance_Sent[i][j] != 'None':
        #             print(x_test_IdInstance_Sent[i][j], ' ' , test_pred_BN[i][j])
                    bnId_file.write(str(x_test_IdInstance_Sent[i][j]) + ' ' + str(test_pred_BN[i][j])) 
                    bnId_file.write('\n')
        print('BabelNet format predicted file saved')

                    
    if output_format == 'LexNames':
        lex_file = open(output_path, "w")
        for i in range(len(x_test_IdInstance_Sent)):
            for j in range(len(x_test_IdInstance_Sent[i])):
                if x_test_IdInstance_Sent[i][j] != 'None':
        #             print(x_test_IdInstance_Sent[i][j] , ' ' , test_pred_Lex[i][j])
                    lex_file.write(str(x_test_IdInstance_Sent[i][j]) + ' ' + str(test_pred_Lex[i][j]))
                    lex_file.write('\n')
        print('LexNames format predicted file saved')
                    

    if output_format == 'WnDomains':
        wDomain_file = open(output_path, "w")
        for i in range(len(x_test_IdInstance_Sent)):
            for j in range(len(x_test_IdInstance_Sent[i])):
                if x_test_IdInstance_Sent[i][j] != 'None':
        #             print(x_test_IdInstance_Sent[i][j] , ' ' , test_pred_WnDomain[i][j])
                    wDomain_file.write(str(x_test_IdInstance_Sent[i][j]) + ' ' + str(test_pred_WnDomain[i][j]))
                    wDomain_file.write('\n')
        print('WnDomain format predicted file saved')
                        