from collections import defaultdict, Counter
import math
import operator
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import random
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import re

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import recall_score, precision_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input
from keras import backend as K
from tensorflow.keras.layers import Dense, MaxPooling1D, Reshape, Dropout, Add, Conv1D, Flatten




########################################################################################
############################ CREATION OF SYNTHETIC DATA ################################
########################################################################################


############################### EXTRACTION OF DATA #####################################

def adjacency_matrix(sequences, nodes=False, synthetic=True, from_element = 0): #Traje=data, nodes=list of nodes labels, from_element= from the element in the sequence from which it should begin to register, default:beginning of the sequence
    '''
    to build the ajacency matrix from the the sequences data 

    IN: 1)sequences: list of list of the sequences data (a sequence in each list of the whole list)
        2)nodes: either False or list /False ->creates the list of nodes from the sequences data / else -> requires the list of nodes,
        3)synthetic: True or False / True->you need to create synthetic data (and will return the nodes not with labels but with ints as labels)/ False: you just need the adjacency matrix of the real nodes
        4)from_element: int from the element in the sequence from which it should begin to register
    
    OUT: 1)adjacency matrix of all the nodes in the data 
         2)list of all the nodes labels
         3)list containing the lenght of each sequence (int number for each element of the list)
         4)list containing as elements the label of the nodes appearing at the beginning of each sequence (one label/element for each sequence)
    ''' 
    if nodes==False:
        nodes = set()
        for record in sequences: # In this loop we take every sequence and from it stores nodes' label
            sequence = record[from_element:]
            for node in sequence: 
                nodes.add(node)
        nodes=list(nodes)
    ad_matrix = np.zeros((len(nodes),len(nodes)))
    if synthetic:
        adj_matrix = pd.DataFrame(ad_matrix, columns=nodes) #empty adjacency matrix
    else:
        adj_matrix = pd.DataFrame(ad_matrix, columns=nodes, index=nodes) #empty adjacency matrix
    len_sequences = [] #when we will create the sequences, we will base both the number and the lenght of them on real data
    first_nodes = [] #here we register the first nodes of every sequence, not visible in the adjacency matrix
    for record in sequences:
        sequence = record[from_element:]
        len_sequences.append(len(sequence))
        node_prev = "First"
        for node in sequence: # node_prev = in rows, node = in columns
            if synthetic:
                raw=str(node)
            else:
                raw=node
                column=node_prev
            if node_prev!="First": #if we are at the second or higher node of the sequence
                if synthetic:
                    column=nodes.index(str(node_prev))
                adj_matrix[raw][column] = adj_matrix[raw][column] + 1 #the meaning of raw and column is correct, if you switch the two, remember to switch also in the if!!
                node_prev=node
            else: #if the node is the first of the sequence, we register it and move to the second one
                node_prev=raw
                column=node_prev
                if synthetic:
                    column=nodes.index(str(node_prev))
                first_nodes.append(column)
    return adj_matrix,nodes,len_sequences,first_nodes



########################## HIGHER ORDERS CONSTRUCTION ###########################à


def basic_data(adj_matrix,frequency_creation=True,nodes=None, n_hod=None, max_order=None, first_nodes=None, prob_df = False):
    '''
    from the adjacency matrix, builds and returns the dict of the ajacency matrix and, if requested builds the synthetic higher order paths and their distributions

    IN: adjacency matrix of the data
    OUT: if frequency creation==True -> returns 1)dictionary of first order connections, 2)dictionary of higher order connections, 3)dataframe containing change in probability between the first and the higher order connection with the picked node
         else -> returns dictionary of first order connections
    '''
    Connections = {} #dictionary to register 1st order probabilities of nodes to end up in other nodes
    if frequency_creation == True: #in case we need to return the connections
        nodes_labels = range(len(nodes)) #just simpler to have numbers
        adj_matrix.columns = nodes_labels
        sum_columns = adj_matrix.sum() #this is telling us how many times every node is pointed at
        t_adj_matrix = adj_matrix.T #a column n of transp give us the probability for n to end up in the other nodes in the real network
        for label in nodes_labels: 
            sum_col = t_adj_matrix[label].sum() #we will use it to normalize
            Connections[label] = {}
            for i in range(len(nodes_labels)):
                if (t_adj_matrix[label][i]/sum_col > 0) and (t_adj_matrix[label][i]/sum_col != np.nan):
                    Connections[label][i] = t_adj_matrix[label][i]/sum_col #here we register the probability for every node to end up in other ones with prob>0
        Connections_higher, Prob_Change = higher_order_connections(Connections, nodes, n_hod, max_order, first_nodes,sum_columns, prob_df)
        return Connections, Connections_higher, Prob_Change
    else: #in case it is used inside the construction of the distributions to build the frequencies with which a order is expected to be present in the data
        nodes_labels = list(adj_matrix.columns) 
        t_adj_matrix = adj_matrix.T #a column n of transp give us the probability for n to end up in the other nodes in the real network
        for label in nodes_labels: 
            sum_col = t_adj_matrix[label].sum() #we will use it to normalize
            Connections[label] = {}
            for i in nodes_labels:
                if (t_adj_matrix[label][i]/sum_col > 0) and (t_adj_matrix[label][i]/sum_col != np.nan):
                    Connections[label][i] = t_adj_matrix[label][i]/sum_col #here we register the probability for every node to end up in other ones with prob>0
        return Connections


def higher_order_connections(Connections, nodes, n_hod, max_order, first_nodes, sum_col, prob_df = False): 
    '''
    builds the synthetic higher order paths and their distributions

    IN: 1)dictionary of first order connections
        2)list of nodes
        3)(int) number of informative paths to build
        4)(int) the set maximum order
        5)list of the nodes appearing at the beginning
        6)list of sum on columns, prob_df tells wether or not to return dataframe of probabilities
    
    OUT: 1)dictionary of higher order connections, 
         2)dataframe containing change in probability between the first and the higher order connection with the picked node
    '''
    
    Prob_Change=pd.DataFrame({'Higher Order':[(0)], 'Changed Node':[0], 'N. Of Nodes':[0], 'Prob. Change':[0], 'Initial Prob':[0]})

    Connections_higher = {} #it will contain all the higher order probabilities
    nodes_list = range(len(nodes))
    k=0
    #right_first_nodes = RightFirstNodes2(nodes_list,first_nodes,sum_col)
    while k <= n_hod:
        k += 1
        for i in range(2,max_order+1):
            higher_order_dependency = nodes_concatenation(Connections,i,nodes_list)
            if higher_order_dependency == False:
                continue
            else:
                node_0, Connections_higher[higher_order_dependency] , change = change_probability(Connections,higher_order_dependency[-1], nodes_list)
                #register in a file the higher order dependencies
                if prob_df:
                    Prob_Change.loc[len(Prob_Change.index)] = [tuple(higher_order_dependency), node_0, len(Connections_higher[higher_order_dependency].keys()),change, Connections[higher_order_dependency[-1]][node_0]]
                #f.write(str(higher_order_dependency)+str(node_0)+" prob. change: "+ str(change)+"\n")    
    Connections_higher = remove_duplicated_keys(Connections_higher, Connections, sum_col, first_nodes) 
    return Connections_higher, Prob_Change



def nodes_concatenation(Connections,max_i,right_first_nodes):
    '''
    concatenates the nodes to build the synthetic informative paths
    '''
    ordered_nodes = []
    for i in range(0,max_i):
        keep_going=True
        if i==0:
            while keep_going:
                node = random.choice(right_first_nodes)
                if (condition_hon_nodes(Connections,node,i,max_i)):
                    keep_going=False
                    ordered_nodes.append(node)
        else:
            count_nodes=[]
            while keep_going:
                node_prev = ordered_nodes[-1]
                node = random.choice(list(Connections[node_prev].keys()))
                count_nodes.append(node)
                if (condition_hon_nodes(Connections,node,i,max_i)): #second one because otherwise the higher order could appear very rarely
                    keep_going=False                                                                            
                    ordered_nodes.append(node)                                            
                if len(np.unique(count_nodes))==len(list(Connections[node_prev].keys())) and keep_going:
                    if len(ordered_nodes)>1:
                        tuple_sequence = tuple(ordered_nodes)
                        return tuple_sequence
                    else:
                       return False
    tuple_sequence = tuple(ordered_nodes)
    return tuple_sequence


### the higher order path gets registered as informative if it satisfies this conditions ###
def condition_hon_nodes(Connections,node,i,max_i):
    '''
    sets the conditions for the path to be registered as informative
    '''
    if i < max_i-2: #if we are 2 steps ahead from the end of the sequence we require the node to have at least one out-connections and, one of its connections to have one as well
        if (len(Connections[node].keys())>0):
            for key in Connections[node].keys():
                if (len(Connections[key].keys())>0):
                    return True #the requirments are there, we return True and the step is accepted 
            return False #otherwise we return false, if no one of the connections has a connection
        else:
            return False #otherwise we return false, the node has no of the outgoing connections has at least a connection
    else: #if we are at the last node,we just require to have two connections, to change the probability at least in one of the two
        if (len(Connections[node].keys())>1):
            return True
        else:
            return False 



def change_probability(Connections,node_1,nodes_list):
    '''
    creation of the higher orders distribution
    changes the probability of one destination node from the first order dependency, of a value that could go from 0.1 to 0.9
    '''
    higher_dict = {}
    keep_going = True
    if len(Connections[node_1].keys())>1:
        while keep_going:
            node_0 = random.choice(list(Connections[node_1].keys()))
            change = 1-Connections[node_1][node_0]
            if change>0.1:
                keep_going = False
                change = 0.1 + random.random()*0.9
                sorted_by_val = {k: b for k, b in sorted(Connections[node_1].items(), key=lambda element: element[1], reverse=True)}
                sorted_high = {}
                sorted_low = {}
                for k in sorted_by_val.keys():
                    if (sorted_by_val[k]>=0.03) and (k!=node_0):#this treshold of 0.03 has to be checked better. Maybe in
                        sorted_high[k] = sorted_by_val[k]       #literature/some calculations there's some hint of a treshold 
                    elif (sorted_by_val[k]<0.03):               #--> a probability not > certain value when combining them 
                        sorted_low[k] = sorted_by_val[k]        #as the change on 2nd order will influence prob on 1st order 
                    else:                                       #always the same idea, we will give random new prob and if something
                        continue                                #changes from 0.0001 to 0.4 too much change (also too detectable)
                sum_prob = Connections[node_1][node_0]+change+sum(sorted_low.values()) #sum without the lowest probabilities
                if (sum_prob)<1: #otherwise the sum of all destinations prob won't be 1 but higher
                    change_neg = 1 - sum_prob
                    keys_new_value = np.random.random(len(sorted_high.keys()))#generating the probabilities to add to higher order dict (only for high value)
                    keys_new_value = (keys_new_value/sum(keys_new_value))*change_neg  #normalizing them
                    keys_low_new_value = np.array([random.choice([np.random.random(),0]) for k in range(len(sorted_low.keys()))])#np.random.random(len(sorted_low.keys()))#generating the probabilities to add to higher order dict (only for low values)
                    #print(keys_low_new_value)
                    keys_low_new_value = (keys_low_new_value/sum(keys_low_new_value))*sum(sorted_low.values())  #normalizing them
                    i=0
                    j=0
                    for key in sorted_by_val.keys():
                        if (key in sorted_high):
                            higher_dict[key]= keys_new_value[i]
                            i += 1
                        elif (key in sorted_low):
                            higher_dict[key]=keys_low_new_value[j]
                            j += 1
                        else:
                            higher_dict[node_0] = Connections[node_1][node_0]+change
                else:
                    higher_dict[node_0] = 1
                    for key in Connections[node_1].keys():
                        if (key != node_0):
                            higher_dict[key]= 0
    else:#here we have only one possible destination, node_0
        node_0 = random.choice(list(Connections[node_1].keys()))
        node_2 = random.choice(nodes_list)
        change = 0.1 + random.random()*0.9
        higher_dict[node_0]=1-change
        higher_dict[node_2]=change
    return node_0,higher_dict,change


def remove_duplicated_keys(Connections, Connections_0, sum_col, first_nodes):
    '''
    to remove paths in case we have the same synthetic informative path or in case one is the "extension" of the other
    '''
    Duplicates=[]
    No_Duplicates = {} #we store the remaining higher order sequences
    # if we have, for example, 2 keys like (66,66,66) (66,66,66,66), we remove one of the two keys randomly, because if you have the first one the hod for the 2nd one is not there anymore if they have the same values
    for key0 in Connections.keys():
        max_key0 = max(Connections[key0].items(), key=operator.itemgetter(1))[0]
        for key1 in Connections.keys():
            max_key1 = max(Connections[key1].items(), key=operator.itemgetter(1))[0]
            if (len(key0) < len(key1)) and (tuple(key1[-len(key0):])==key0) and (max_key0==max_key1) and (abs(Connections[key0][max_key0]-Connections[key1][max_key1])<0.4):
                if calculate_expected_frequency(key1, Connections_0, sum_col, first_nodes)<0.0025:#this treshold is derived from the study of sequences: some sequences never appear in sequences and it only happens if their expected frequency is<0.0025
                    Duplicates.append(key1)
                else:
                    Duplicates.append(random.choice([key0,key1]))
    #here we remove the keys that are duplicates
    for key in Connections.keys():
        if (key in No_Duplicates.keys()) or (key in Duplicates):
            continue
        else:
            No_Duplicates[key] = Connections[key]
    return No_Duplicates


def calculate_expected_frequency(key, Connections, sum_col, first_nodes):
    '''
    to calculate expected frequencies of appearence of a path
    '''
    frequency_expected = 0
    frequency_in_first_nodes=first_nodes.count(key[0])/len(first_nodes)
    frequency_in_next_steps=sum_col[key[0]]/sum(list(sum_col))
    frequency_key0 = frequency_in_first_nodes + frequency_in_next_steps
    frequency_expected+=frequency_key0
    for node in range(0,len(key)-1):
        added_prob=Connections[key[node]][key[node+1]]
        frequency_expected=frequency_expected*added_prob
    return frequency_expected




###################################### SEQUENCES CONSTRUCTION #########################################



def sequences_construction(Connections,Connections_higher,len_sequences,first_nodes,max_order):
    '''
    constructs the synthetic sequences data

    IN: 1)first order dependencies dict
        2)higher order dependencies dict
        3)list of the lenght of the sequences
        4)list of the labels of the first nodes
        5)the maximum order of the higher order dependencies
    
    OUT:synthetic sequences
    '''
    n_sequences = len(len_sequences) #number of sequences to create (the same number of the real ones)
    sequences = []
    keys_list = list(Connections_higher.keys()) #I am changing in lists, but maybe tuples would work as well?
    for i in range(n_sequences):
        if i % 100000 == 0:
            print("N. of constructed sequences: ", i)
        sequence = [] #we initialize a new sequence
        sequence_length = random.choice(len_sequences) #the length of the sequences will have appear with same frequency of real ones
        for k in range(1,sequence_length+1): #it is more convenient to have k as the real number of steps in the sequence
            if k==1:
                node_0 = random.choice(first_nodes) #the initial nodes will appear with same frequency of real ones
                sequence.append(node_0)
                next_step = select_next_step(node_0,Connections) #next step after first node, no higher order possible
                sequence.append(next_step)
            else:
                if any(sequence[-1]==v[-1] for v in keys_list): #if the last node of sequence is not the last element of any of keys, no need to look for a higher order
                    key = higher_order_key(Connections_higher,sequence,k,max_order)
                    if key==False: #there's no higher order dependency in sequence
                        next_step = select_next_step(sequence[-1],Connections)
                        sequence.append(next_step)
                    else:
                        next_step = select_next_step(tuple(key),Connections_higher)
                        sequence.append(next_step)
                else:
                    if len(Connections[sequence[-1]].keys())>0:
                        next_step = select_next_step(sequence[-1],Connections)
                        sequence.append(next_step)
                    else:
                        #print(str(sequence[-1])+": No keys")
                        break
        sequences.append(sequence)
    return sequences


def select_next_step(node_prev,Connections):#This will work regardless of first or higher order, you just have to put the right dict
    '''
    Picks the next node to build a higher order path.
    e.g. if you have node A, you look at the connections A has in the first order dependencies (let's say B and C) and choose one of the destination nodes (let's say C)
    if you have the path A,C and need to build a path of the third order, you will need another node after C, so you look at the destinations after C, let's say B, F and G, and choose one of them, let's say F. It will return F

    IN: 1)node label of the last node in the path
        2) first order dependencies dict

    OUT: the chosen node label among the ones that are possible destination of node_prev
    '''
    keep_going = True
    while keep_going:
        node = random.choice(list(Connections[node_prev].keys()))
        rand_p = random.random()
        if Connections[node_prev][node]>=rand_p:
            keep_going = False
            return node
        

#to check if in the sequence a higher order dependency appears,
#return it if it does or return False otherwise
#so that we know which dict/prob we have to use
def higher_order_key(Connections,sequence,k,max_order):
    '''
    To check if in the sequence a higher order dependency appears,
    return it if it does or return False otherwise
    so that we know which dict/prob we have to use
    '''
    key=[]
    for i in range(2,min(k,max_order)+1):
        if tuple(sequence[-i:]) in Connections.keys():
            key = sequence[-i:] #letting the loop go, we always return the higher order sequence for example
                             #if we have two hoc in Connections_higher (188,188) and (36,188,188) and  
    if len(key)>0:           #our sequence is (43,36,188,188) we will return (36,188,188)
        return key
    else:
        return False









#############################################################################
############ EXTRACTION OF HO DEPENDENCIES IN SYNTHETIC DATA ################
#############################################################################


Prob_Change=pd.DataFrame({'Higher Order':[(0)], 'Changed Node':[0], 'N. Of Nodes':[0], 'Prob. Change':[0], 'Initial Prob':[0]})


def extract_paths_quantities(data, sequences_basic, max_order, min_support, scale_factor, Connections_2={}, n=4, synthetic=False, from_element=0, min_support_w=0, weight_element=None, first_treshold=0, treshold_element=None):
    '''
    To create the dataframe with higher and lower order paths and their JSD distances

    arguments: data = list of list: sequences data (also with the part that contain weights and information on the sequences that will be used to weight the paths and importance in the network)
               sequences_basic = list of list: the sequences with nodes label only (same as data if there are no wieghts to take into consideration)
               max_rder = int: maximum order to study in the sequences
               min_support = ---
               scale_factor = int: scaling factor of the minimum support for the validation set 
               Connections_2= dictionary: dependencies of the informative paths (in case you have synthetic data and want to evaluate the model performance)
               n= int: division of the dataset, for example if n=4, 1 set will be used for the validation set and 3 for the training set
               synthetic= True/False: if True you are processing synthetic data, False if you are processing real data
               from_element= int: from which element of the list to take the sequences nodes
               min_support_w= int: minimum acceptable appearence of the path in the sequence to be considered
               weight_element= int: position of the element in the list of sequence from which to take the element that weights every sequence (which would be a float)
               first_threshold = float: a threshold under which a sequence is not considered
               treshold_element= int: position of the element in the list of sequence from which to take the element that should be over/under the threshold for the sequence to be accepted/discarded
    '''
    Count = {}
    Count_tot = {}
    Count_w = {}
    Count_tot_adapted= {}
    #we count patterns and their targets 
    Count = build_observations(Count,data,max_order, from_element, weight_element, first_treshold, treshold_element)
    Count_tot, Count_w = adapt_dict(Count,Count_tot,Count_w,n)
    adj_matrix2,nodes2,_,first_nodes2 = adjacency_matrix(sequences_basic,nodes=False, synthetic=False, from_element=from_element)#,synthetic=synthetic)
    sum_col = adj_matrix2.sum()
    Connections1 = create_dict_norm_destination(adj_matrix2)
    for r in range(0,n):
        #training distribution
        Count_train = aggregate_count(Count_w,  Count_tot, Count_tot_adapted, r, n, val=False)
        Distribution_train = build_distributions(Count_train,min_support_w) 
        #validation distribution
        Count_val = aggregate_count(Count_w,  Count_tot, Count_tot_adapted, r, n, val=True)
        Distribution_val = build_distributions(Count_val,min_support_w/scale_factor)
        #Generation of rules/quantities
        Frequencies_Store0 = generate_paths_quantities(max_order, Count_tot_adapted, Count_train, Count_val, Distribution_train, Distribution_val, Connections_2,Connections1,first_nodes2,sum_col)
        print("k-fold n. " + str(r) + " done")
        if r==0:
            Frequencies_Store_final = Frequencies_Store0.set_index(['Higher Order', 'Lower Order'])
        else:
            #we ordinate the higher orders in the new dataset as the ones in the old dataset
            Frequencies_Store0 = Frequencies_Store0.set_index(['Higher Order', 'Lower Order'])
            Frequencies_Store_final = add_columns(r, Frequencies_Store_final, Frequencies_Store0) #we add JSD High|JSD Low|JSD(Extr,Distr) to the old DataFrame
    #NOTE!!!! In the need of the dictionary of the distributions to create the network, uncomment the "Distribution_train" from return line, you will have a dictionary usable with the hon library
    #we return the dataframe asking to remove the positives (Type==g) with distances between the lower and higher distributions all equal to zero in every repetition of the cross-validation
    return Frequencies_Store_final[(((Frequencies_Store_final['JSD(Extr,Distr)']!=0)|(Frequencies_Store_final['JSD(Extr,Distr)1']!=0)|(Frequencies_Store_final['JSD(Extr,Distr)2']!=0)|(Frequencies_Store_final['JSD(Extr,Distr)3']!=0))&(Frequencies_Store_final['Type']=='g'))|(Frequencies_Store_final['Type']=='b')] #,Distribution_train



def build_observations(Coun,sequences, max_order, from_element, weight_element, first_treshold, treshold_element):
    '''
    Creates the dictionary containing all the first and higher order dependencies in the data of all the paths 
    '''
    for record in sequences:
        sequence = record[from_element:] #in case not all the sequence needs to be used but only from one element of the sequence
        if treshold_element == None: 
            condition = 0 #in case we don't need a first treshold to accept or not to include the sequence, this will be 0, as first_treshold is and every sequence is accepted
        else: #in case there is one element that we use as a first treshold to accept or not to include the sequence
            condition = record[treshold_element]
        if float(condition) >= float(first_treshold): #this is useful in Orbis data, because not all companies gets accepted, only the ones having profits higher than a certain treshold
            if weight_element == None: 
                weight = 1 #in case we don't want/have a weighted network, then weight =1 which means just counting
            else: #in case we need to include the weight
                weight = float(record[weight_element])
            for order in range(2, max_order+2):
                    SubSequence = extract_subsequences(sequence, order)
                    for path in SubSequence:
                        target = path[-1]
                        source = path[:-1]
                        Coun = increase_counter(Coun,source, target, weight) #here we construct the dict Coun
    return Coun


def extract_subsequences(sequences, order):
    '''
    Extracts all the paths in a sequence 
    '''
    SubSequence = []
    for starting in range(len(sequences) - order + 1):
        SubSequence.append(tuple(sequences[starting:starting + order]))
    return SubSequence


def increase_counter(Coun, source, Target, weight):
    '''
    Adds every path to the dictionary of the higher order dependencies with its targets (the following nodes) and the amounts of times they appear
    In the form of {path1: {[node3,weight1],[node6,weight4],[node3, weight4],[node8,weight9]} , path2:{[node5, weight6], ecc...}}
    in this way we will be able to divide, for every path, the destinations of each path in validation and training set
    the first 3 destination nodes (the first 3 lists of path1) in the example will be used for the training set, the other one for the validation set
    '''
    if not source in Coun:
        Coun[source] = defaultdict(list)
    Coun[source][len(Coun[source].keys())+1] = [Target, weight] #we save, for every source (path) and for every of its pattern, a list with its destination and its weight
    return Coun


def adapt_dict(Coun, Coun_tot,Count_w,n):
    '''
    Creates a dictionary in which the paths appearences are divided in four parts
    '''
    for r in range(0,n):
        Coun_tot[r] = defaultdict(dict) #we divide the counter in n parts, 1 will work for the validation and the other three for the training set
        Count_w[r] = defaultdict(dict)
        for source in Coun.keys():
            tot = len(Coun[source].keys())
            if tot<n: #if we have less than n elements of the same pattern, we do not record it, otherwise we cannot have the same source in every of the n parts of the counter
                continue
            else:
                index0 = int(r*tot/n) #index of the first element in the division of the dictionary
                index1 = int((r+1)*tot/n)
                Coun_tot[r][source] = defaultdict(float)
                Count_w[r][source] = defaultdict(float)
                for i in range(index0,index1):
                    target_list = Coun[source][i+1]
                    Target = target_list[0]
                    Weight = target_list[1]
                    Coun_tot[r][source][Target] += 1 #we go back to the original form of the Xu et al. paper with the difference that we have the division in 4 part
                    Count_w[r][source][Target] += Weight #same as Count_tot, but with the weight instead of a simple counting
    return Coun_tot, Count_w

def build_distributions(Coun,min_support):
    '''
    Builds the distributions of the given set from the counter dictionary
    '''
    Distr = defaultdict(dict)
    for source in Coun:
        for Target in Coun[source].keys():
            if (Coun[source][Target] < min_support):
                Coun[source][Target] = 0
        for Target in Coun[source]:
            if Coun[source][Target] > 0:
                Distr[source][Target] = Coun[source][Target] / sum(Coun[source].values())
    return Distr
        


def aggregate_count(Coun, Count_tot, Count_tot_adapted, r, n, val):
    '''
    Aggregates the counts of destinations in the dictionary that need to be used in the training set and returns the part that needs to be used in the validation set
    '''
    if val: #for the validation set it's easy, we just take one of the n parts
        Coun_val = Coun[r]
        return Coun_val
    else:#for training set we need to aggregate
        train_sets_n = list(range(n))
        train_sets_n.remove(r)
        Coun_train = defaultdict(dict)
        for source in Coun[train_sets_n[0]].keys():
            Coun_train[source]=dict(Counter(Coun[train_sets_n[0]][source])+Counter(Coun[train_sets_n[1]][source])+Counter(Coun[train_sets_n[2]][source])) #summing all the targets values in two different parts of the counter, everything is aggregated in Count_train
            if not source in Count_tot_adapted:
                Count_tot_adapted[source] = dict(Counter(Count_tot[0][source])+Counter(Count_tot[1][source])+Counter(Count_tot[2][source])+Counter(Count_tot[3][source])) #we will need this to record later how many times a HO appears in the data
    return Coun_train




def generate_paths_quantities(max_order, Count_tot_adapted, Count_train, Count_val, Distribution_train, Distribution_val, Connections_2,Connections1,first_nodes2,sum_col):
    '''
    To generate the dataframe of higher and lower order paths and their JSD distances
    '''
    Frequencies_Store=pd.DataFrame({'Higher Order':[0], 'Lower Order':[0], 'JSD High':[0], 'JSD Low':[0], 'JSD(Extr,Distr)':[0], 'High Frequency':[0], 'Low Frequency': [0], 'High Fr. Adj':[0], 'Low Fr. Adj.':[0], 'Type': ['black']})
    SourceToExtSource_train = build_source_to_extsource(Distribution_train) # to speed up lookups
    SourceToExtSource_val = build_source_to_extsource(Distribution_val)
    loop_counter = 0
    #RulesVal = defaultdict(dict)
    for source in Distribution_train:
        if source in Distribution_val:
            if len(source) == 1:
                #RulesVal = add_to_rules(RulesVal,Count_train, source)
                Frequencies_Store = extend_rule(Count_tot_adapted, Count_train, Count_val, Distribution_train, Distribution_val, source, source, SourceToExtSource_train, SourceToExtSource_val, 1, max_order, Connections_2,Frequencies_Store,Connections1,first_nodes2,sum_col)
                loop_counter += 1
        else:
            continue
    return Frequencies_Store



def build_source_to_extsource(Distr):
    '''
    Creates a dictionary with all the subpaths of a path
    '''
    SourceToExtSource = {}
    for source in Distr:
        if len(source) > 1:
            new_order = len(source)
            for starting in range(1, len(source)):
                curr = source[starting:]
                if not curr in SourceToExtSource:
                    SourceToExtSource[curr] = {}
                if not new_order in SourceToExtSource[curr]:
                    SourceToExtSource[curr][new_order] = set()
                SourceToExtSource[curr][new_order].add(source)
    return SourceToExtSource

#in this case it is not used but it will be to reconstruct the network
def add_to_rules(RulesVal, Count_train, source):
    '''
    Creates a dictionary containing all the informative paths and their subpaths as in the context tree
    Used only to reconstruct the network, not to build dependencies
    '''
    if len(source) > 0:
        ## To output frequencies instead of probabilities, change "Distribution" to "Count"
        ## and filter out zero values
        RulesVal[source] = Count_train[source]
        prev_source = source[:-1]
        RulesVal = add_to_rules(RulesVal, Count_train, prev_source)
    return RulesVal


def extend_rule(Count_tot_adapted, Count_train, Count_val, Distribution_train, Distribution_val, Valid, Curr, SourceToExtSource_train, SourceToExtSource_val, order, max_order, Connections_2,Frequencies_Store,Connections1,first_nodes2,sum_col):
    '''
    Fills the final output dataframe
    '''
    if order >= max_order:
        return Frequencies_Store
    else:
        Distr_train = Distribution_train[Valid]
        Distr_val = Distribution_val[Valid]
        new_order = order + 1
        Extended_t = extend_source(SourceToExtSource_train,Curr, new_order)
        Extended_v = extend_source(SourceToExtSource_val,Curr, new_order)
        if len(Extended_t) == 0:
            return Frequencies_Store
        else:
            for ExtSource in Extended_t:
                if ExtSource in Extended_v:
                    ExtDistr_t = Distribution_train[ExtSource]
                    ExtDistr_v = Distribution_val[ExtSource]
                    low_frequency_adj = calculate_frequencies(Valid,Connections1,first_nodes2,sum_col)
                    high_frequency_adj = calculate_frequencies(ExtSource,Connections1,first_nodes2,sum_col)
                    low_frequency = sum(Count_tot_adapted[Valid].values())#sum(Count_train[Valid].values()) + sum(Count_val[Valid].values())
                    high_frequency = sum(Count_tot_adapted[ExtSource].values())#sum(Count_train[ExtSource].values()) + sum(Count_val[ExtSource].values())
                    jsd_low = jsd(Distr_val, Distr_train) 
                    jsd_high = jsd(ExtDistr_t, ExtDistr_v)
                    jsd_lowhigh = jsd(ExtDistr_t, Distr_train)
                        # higher-order dependencies exist for order new_order
                        # keep comparing probability distribution of higher orders with current order
                    if ExtSource in Connections_2.keys():
                        Frequencies_Store.loc[len(Frequencies_Store.index)] = [ExtSource, Valid, jsd_high, jsd_low, jsd_lowhigh, high_frequency, low_frequency, high_frequency_adj, low_frequency_adj,'g']
                    else:
                        Frequencies_Store.loc[len(Frequencies_Store.index)] = [ExtSource, Valid, jsd_high, jsd_low, jsd_lowhigh, high_frequency, low_frequency, high_frequency_adj, low_frequency_adj, 'b'] 
                    Frequencies_Store = extend_rule(Count_tot_adapted, Count_train, Count_val, Distribution_train, Distribution_val, ExtSource, ExtSource, SourceToExtSource_train, SourceToExtSource_val, new_order, max_order, Connections_2,Frequencies_Store,Connections1,first_nodes2,sum_col)
                        # higher-order dependencies do not exist for current order
                        # keep comparing probability distribution of higher orders with known order
   
                else:
                    continue
            return Frequencies_Store
        

def extend_source(SourceToExt, Curr, new_order):
    '''
    Returns the subpath (lower order) of the path
    '''
    if Curr in SourceToExt:
        if new_order in SourceToExt[Curr]:
            return SourceToExt[Curr][new_order]
    return []


def calculate_frequencies(key,Connections1,first_nodes2,sum_col):
    '''
    calculates the expected frequency of appearence of a path from the first order dependencies
    '''
    frequency_expected = 0
    frequency_in_first_nodes=first_nodes2.count(key[0])/len(first_nodes2)
    frequency_in_next_steps=sum_col[key[0]]/sum(list(sum_col))
    frequency_key0 = frequency_in_first_nodes + frequency_in_next_steps
    frequency_expected+=frequency_key0
    for node in range(0,len(key)-1):
        added_prob=Connections1[key[node]][key[node+1]]
        frequency_expected=frequency_expected*added_prob
    return frequency_expected


def jsd(a, b):
    '''
    calculates the JSD of two distributions 
    '''
    M = {}
    for key in np.unique(list(a.keys())+list(b.keys())):
        if (key in a.keys()) and (key in b.keys()):
            M[key]=(a[key] + b[key])/2
        elif (key in a.keys()) and (key not in b.keys()):
            M[key]=(a[key])/2
        elif (key not in a.keys()) and (key in b.keys()):
            M[key]=(b[key])/2
    divergence = (kld(a,M)+kld(b,M))/2
    return divergence


def kld(a, b):
    '''
    calculates the KLD of two distributions 
    '''
    divergence = 0
    for target in a:
        if get_probability(b, target)==0 and get_probability(a, target)!=0:
            continue
        else:
            divergence += get_probability(a, target) * math.log((get_probability(a, target)/get_probability(b, target)), 2)
    return divergence


def get_probability(d, key):
    
    '''
    takes the probability of a node to a target in the distribution 
    '''
    if not key in d:
        return 0
    else:
        return d[key]
    



def add_columns(r,datafr1, datafr2): 
    '''
    Adds the columns of the new DataFrame to the old one in the right order
    arguments: r=iteration or n. of the column to add ; datafr1 = old DataFrame ; datafr2 = new DataFrame
    '''
    new_columns = {col: col+str(r) for col in datafr2.columns}
    datafr2 = datafr2.rename(columns = new_columns)
    dataframe_final = pd.concat([datafr1, datafr2], axis=1)
    ###here we want the columns that are not needed to be more than one to avoid the NaN (for example the "Type" column)
    for i in range(0,5):
        head2 = datafr2.columns.values
        head1 = datafr1.columns.values
        new_col2= head2[i-5]
        new_col1= head1[i-5]
        col = pd.DataFrame({new_col1: dataframe_final[new_col2]})
        dataframe_final.update(col)
        dataframe_final = dataframe_final.drop([new_col2],axis=1)
    #####to order columns in the right way####
    #we want the same types of columns to appear one after another, like that --> |rose|rose1|rose2|rose3|violet|violet1|violet2| ecc...
    #dataframe_final = dataframe_final.reset_index()
    headings = dataframe_final.columns.values
    range_max = len(headings)-5
    columns_order = []
    count = 0
    for head in range(0,range_max):
        if (head+1)%(r+1)==0:
            columns_order.append(count-3)
            count+=1
        else:
            columns_order.append(head-count)
    columns_order = columns_order+[head-8 for head in range(0,5)]
    dataframe_final = dataframe_final.iloc[:, columns_order]
    return dataframe_final



#############################################################################
##################### MULTIPLE DISTANCES COLLECTION #########################
#############################################################################


#HERE WE COLLECT THE DATA, POSITIVE AND NEGATIVES#
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from time import time

def process_iteration(inputs):
    '''
    Processes multiple iterations at a time to create synthetic data and extract the quantities/distances of all the paths
    
    IN: n= int : max order of the paths
        adj_matrix = dataframe of first order dependencies of the adjacency matrix
        len_sequences = list of the lenght of sequences (each element a int of the lenght of the corresponding sequences)
        first_nodes = list containing the label (in int) of each first node in the sequences

    returns: dataframe containing the higher and lower order label and the various JSD distances 

    '''
    n, adj_matrix, nodes, len_sequences, first_nodes = inputs

    start = time()
    Connections, Connections_2, Prob_Change = create_higher_order_patterns(adj_matrix, frequency_creation=True, nodes=nodes,
                                                        n_hod=900, max_order=5, first_nodes=first_nodes)
    print("Basic data", time() - start)

    start = time()
    sequences = sequences_construction(Connections, Connections_2, len_sequences, first_nodes, 5)
    sequences_basic = sequences
    print("sequences construction", time() - start)

    start = time()
    Frequencies_Store0 = extract_paths_quantities(sequences, sequences_basic, 6, 1, 1,
                                                            Connections_2=Connections_2, n=n, synthetic=False)
    Frequencies_Store0 = Frequencies_Store0[Frequencies_Store0["Type"] != 'black']
    Frequencies_Store0 = Frequencies_Store0.reset_index()
    print("Extract rules", time() - start)

    return Frequencies_Store0

#HERE WE COLLECT THE DATA, POSITIVE AND NEGATIVES#
def collect_synthetic_data(data, iterations=20, n=4, adj_matrix=[], nodes=[], len_sequences=[], first_nodes=[]):
    '''
    Processes multiple iterations at a time to create synthetic data and extract the quantities/distances of all the paths
    
    IN: data= list of list: sequences data
        interations= int: number of synthetic datasets you want to create
        n= int: number k for cross-validation
        adj_matrix(optional) = dataframe: adjacency matrix containing the first order dependencies
        len_sequences (optional)= list of the lenght of sequences (each element a int of the lenght of the corresponding sequences)
        first_nodes (optional)= list containing the label (in int) of each first node in the sequences

    returns: dataframe containing the higher and lower order label and the various JSD distances 

    '''
    if len(adj_matrix)==0:#in case we start from paths, otherwise we insert all the other quantities in the argument
        # Reconstruction of real adjacency matrix and acquisition of nodes, sequences length, frequency of first nodes
        adj_matrix, nodes, len_sequences, first_nodes = create_adj_matrix_from_paths(data, nodes=None)

    #This makes it much faster (depends on the number of cores of your computer), but makes debugging harder, so I'm commenting it out
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(executor.map(process_iteration, [(n,adj_matrix, nodes, len_sequences, first_nodes)]*iterations))
    #results = [process_iteration((data, n)) for i in range(iterations)]

    Frequencies_Store = pd.concat(results).reset_index(drop=True)
    #we return the dataframe asking to remove the positives (Type==g) with distances between the lower and higher distributions all equal to zero in every repetition of the cross-validation
    return Frequencies_Store[(((Frequencies_Store['JSD(Extr,Distr)']!=0)|(Frequencies_Store['JSD(Extr,Distr)1']!=0)|(Frequencies_Store['JSD(Extr,Distr)2']!=0)|(Frequencies_Store['JSD(Extr,Distr)3']!=0))&(Frequencies_Store['Type']=='g'))|(Frequencies_Store['Type']=='b')]


#syntethis=True is not used, from_element is likely not used
def create_adj_matrix_from_paths(data, nodes=None, synthetic=True, from_element=0):
    """
    Creates an adjacency matrix from path data

    IN: data=list of list: sequences data
        nodes=list containing label of the nodes
        synthetic=True for the future creation of synthetic data with ints as nodes labels, False to obtain the adjacency matrix with real labels
        from_element=int, says from which element to take the sequence
    
    returns: n= int : max order of the paths
             adj_matrix = dataframe of first order dependencies of the adjacency matrix
             len_sequences = list of the lenght of sequences (each element a int of the lenght of the corresponding sequences)
             first_nodes = list containing the label (in int) of each first node in the sequences
    """

    # If nodes are not provided, get them from the data
    if nodes is None:
        nodes = list(set([element for path in data for element in path[from_element:]]))
    
    # Create a mapping of airport codes to indices (speed up lookup)
    node_indices = {node: index for index, node in enumerate(sorted(nodes))}
    
    # Initialize the adjacency matrix with zeros 
    num_nodes = len(node_indices)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Populate the adjacency matrix based on the paths
    for path in data:
        for i in range(from_element, len(path) - 1):
            start_index = node_indices[path[i]]
            end_index = node_indices[path[i + 1]]
            adjacency_matrix[start_index][end_index] += 1
    
    if synthetic: 
        adj_mat = pd.DataFrame(adjacency_matrix)
        # Record first nodes (useful to create realistic paths later on)
        first_nodes = [node_indices[path[0]] for path in data]
    else:
        adj_mat = pd.DataFrame(adjacency_matrix, columns=nodes, index=nodes) 
        first_nodes = [path[0] for path in data]

    # Calculate length of paths (useful to create realistic paths later on)
    len_sequences = [len(path) for path in data]

    return adj_mat, nodes, len_sequences, first_nodes

def create_dict_norm_destination(adj_matrix):
    """
    Creates a dictionary of form {"node_source": {"node_target1": share_weight1, "node_target2": share_weight2}}

    IN: adj_matrix=dataframe of the adjacency matrix of the sequences data
    returns: dictionary of the corresponding first order dependencies between nodes

    """
    a = adj_matrix.T #transpose to normalize by row
    a = a/np.sum(a) #normalize by column (by row in original)
    a = a.replace(0, np.NaN).T #back to original

    #convert to dictionary excluding missing values
    return {i: {k:v for k,v in m.items() if pd.notnull(v)} for i, m in zip(a.index, a.to_dict(orient='rows'))}

def create_higher_order_patterns(adj_matrix,frequency_creation=True,nodes=None, n_hod=None, max_order=None, first_nodes=None, prob_df = False):
    """
    Based on an adjacency matrix, create dictionaries with higher order patterns

    IN: 1)2D array of the adjacency matrix
        2)frequency_creation=True to create synthetic data
        3)list of nodes' labels
        4)(int) number of informative paths to build
        5)(int) the set maximum order
        6)list of the nodes appearing at the beginning
        7)list of sum on columns, prob_df tells wether or not to return dataframe of probabilities
    """
    # Remove the column/rows (if any)
    Connections = create_dict_norm_destination(pd.DataFrame(adj_matrix.values))

    # Create higher orders    
    Connections_higher, Prob_Change = higher_order_connections(Connections, nodes, n_hod, max_order, first_nodes, adj_matrix.sum(), prob_df)

    
    return Connections, Connections_higher, Prob_Change


#############################################################################
######################### HIGHER ORDERS DETECTION ###########################
#############################################################################




def classify_higher_order_paths(data,n_negatives, n_positives,synthetic=False,var_order=False):
    '''
    Classifies the paths asinformative or not

    arguments:  data= dataframe: contains all the higher and lower orders and their JSD distances to classify
                n_negatives= int/df: number of negatives used in the HGBC to train the algorithm OR dataframe containing the negatives example to train the algorithm
                n_positives= int/df: number of positives used in the HGBC to train the algorithm OR dataframe containing the positives example to train the algorithm
                synthetic= True/False: if True you are classifying synthetic data (and thus returns also the validation metrics), if False return the classification only 
                var_order= True/False: if True you are going to include the order of the higher order path as input variable of the HGBC, if False you won't use it
    '''
    #training the model with previously collected data
    indep_vars,dep_var,X_train,y_train,positives,negatives = prepare_training_data(n_negatives,n_positives,var_order)
    grid_search = model_algorithm()
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    #classifing on data
    data = process_data(data,var_order)
    x_test = data[indep_vars]
    pred = best_model.predict(x_test)
    data["prob"] = best_model.predict_proba(x_test)[:,1]
    data["classification"]= np.where(pred>=0.5, 1,0)
    #should add the roc curve plot
    if synthetic:
        data = mapping(data)
        y_test = data[dep_var].astype(int)
        y_test = np.array(y_test)
        ### Performance Metrics ###
        print("Precision: ", precision_score(y_test, pred))
        print("Recall: ", recall_score(y_test, pred))
        print("AUC: ",  roc_auc_score(y_test, pred))
        false_neg = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)
        false_neg.update_state(y_test, pred)
        false_negatives = false_neg.result().numpy()
        false_pos = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
        false_pos.update_state(y_test, pred)
        false_positives = false_pos.result().numpy()
        true_neg = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
        true_neg.update_state(y_test, pred)
        true_negatives = true_neg.result().numpy()
        true_pos = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
        true_pos.update_state(y_test, pred)
        true_positives = true_pos.result().numpy()
        F_1 = sklearn.metrics.f1_score(y_test, pred)
        # Add predictions to test dataset
        print("Best F1 Score: ", F_1)
        print('False Negatives:\t', false_negatives)
        print('False Positives:\t', false_positives)
        print('True Negatives:\t', true_negatives)
        print('True Positives:\t', true_positives)
        ##### ROC CURVE #####
        fpr, tpr, threshold = roc_curve(y_test, data["prob"])
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        return data, [F_1, precision_score(y_test, pred), recall_score(y_test, pred), roc_auc_score(y_test, pred)]
    return data,positives,negatives



def prepare_training_data(n_negatives,n_positives,var_order):
    '''
    divides training data in training and validation set and sets the variable to use to classify
    '''
    df,positives,negatives=extract_training_data(n_negatives,n_positives)
    df=process_data(df,var_order)
    indep_vars,dep_var,X_train,y_train=prepare_train_test_sets(df)
    return indep_vars,dep_var,X_train,y_train,positives,negatives

def mapping(df):
    '''
    maps the results in b:not informative path, g:informative path
    '''
    #map for classification
    real = {"b": False,
            "g": True
        }
    #mapping
    df["Class"] = df["Type"].map(real)
    return df


def extract_training_data(n_negatives,n_positives):
    '''
    Imports and cleans the training data
    '''
    if type(n_negatives)!=int:#in case you want to upload the df files with the positives and negatives
        negatives=n_negatives
        positives=n_positives
    else:
        #Data collection positives/negatives
        positives=pd.read_csv("../data/output/hon_analysis/Frequencies_Positives_Final_cleaned.csv", index_col=0).sample(n=n_positives)
        negatives=pd.read_csv("../data/output/hon_analysis/Frequencies_Negatives_Final.csv", index_col=0).sample(n=n_negatives)
    df = pd.concat([
        negatives,
        positives
        ])
    df = df[df["Type"]!='black']
    df = mapping(df)
    return df,positives,negatives

def process_data(df,var_order):
    '''
    Prepares the data variables to be used in the HGBC 
    '''
    if var_order:
        df = df.reset_index()
        df['Order'] = df['Lower Order'].apply(lenght_order)
    df = df[df["Type"]!='black'] #removing useless columns
    df["JSD_high_t_mean"] = df[['JSD High', 'JSD High1', 'JSD High2', 'JSD High3']].mean(axis=1, numeric_only=True)
    df["JSD_low_t_mean"] = df[['JSD Low', 'JSD Low1', 'JSD Low2', 'JSD Low3']].mean(axis=1, numeric_only=True)
    df["JSD_test_mean"] = df[['JSD(Extr,Distr)', 'JSD(Extr,Distr)1', 'JSD(Extr,Distr)2', 'JSD(Extr,Distr)3']].mean(axis=1, numeric_only=True)
    df["JSD_high_t_std"] = df[['JSD High', 'JSD High1', 'JSD High2', 'JSD High3']].std(axis=1, numeric_only=True)
    df["JSD_low_t_std"] = df[['JSD Low', 'JSD Low1', 'JSD Low2', 'JSD Low3']].std(axis=1, numeric_only=True)
    df["JSD_test_std"] = df[['JSD(Extr,Distr)', 'JSD(Extr,Distr)1', 'JSD(Extr,Distr)2', 'JSD(Extr,Distr)3']].std(axis=1, numeric_only=True)
    df["JSD_high_t_min"] = df[['JSD High', 'JSD High1', 'JSD High2', 'JSD High3']].min(axis=1, numeric_only=True)
    df["JSD_low_t_min"] = df[['JSD Low', 'JSD Low1', 'JSD Low2', 'JSD Low3']].min(axis=1, numeric_only=True)
    df["JSD_test_min"] = df[['JSD(Extr,Distr)', 'JSD(Extr,Distr)1', 'JSD(Extr,Distr)2', 'JSD(Extr,Distr)3']].min(axis=1, numeric_only=True)
    df["JSD_high_t_max"] = df[['JSD High', 'JSD High1', 'JSD High2', 'JSD High3']].max(axis=1, numeric_only=True)
    df["JSD_low_t_max"] = df[['JSD Low', 'JSD Low1', 'JSD Low2', 'JSD Low3']].max(axis=1, numeric_only=True)
    df["JSD_test_max"] = df[['JSD(Extr,Distr)', 'JSD(Extr,Distr)1', 'JSD(Extr,Distr)2', 'JSD(Extr,Distr)3']].max(axis=1, numeric_only=True)
    return df

def prepare_train_test_sets(df):
    '''
    Selects the metrics to be used in the HGBC and separates the input and output columns
    '''
    indep_vars = []#['High Fr. Adj', 'Low Fr. Adj.'] #the quantities for which we have one column only
    if 'Order' in df.columns:
        indep_vars +=['Order']
    for metric in ["mean", "std", 'min','max']:#we add the ones for which we calculated mean, std, max, min, ecc...
        indep_vars += [f"JSD_high_t_{metric}", f"JSD_low_t_{metric}", f"JSD_test_{metric}"]
    dep_var = "Class"
    X_train = df[indep_vars]
    y_train = df[dep_var].astype(int)
    return indep_vars,dep_var,X_train,y_train


def model_algorithm():
    '''
    Sets the algorithm to be used
    '''
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('algorithm', LogisticRegression()) 
        ])
    # Define the parameter grid including both algorithms
    parameters = [
        {
            'algorithm': [LogisticRegression(max_iter=100)],
            'algorithm__C': np.logspace(-1, 5, 20)
        },
        {
            'algorithm': [HistGradientBoostingClassifier()],
            'algorithm__learning_rate': [0.1,1,0.01]#,0.0001,0.001,0.01,1
            #'algorithm__max_depth': [4,6,10,20,30,100],
            #'algorithm__max_iter': [5,10,20,30],
            #'algorithm__min_samples_leaf': [2,3,4]

        }
        ]
    # Perform hyperparameter tuning using cross-validation  
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, scoring='f1')
    return grid_search


def lenght_order(order):
    '''
    Defines and returns the order of the input path
    '''
    if type(order)==str:
        thing=order
        thing=re.sub('\(', '',thing)
        thing=re.sub('\)', '',thing)
        thing = thing.split(',')
        lenght = 0
        for i in range(len(thing)):
            if len(thing[i])>0:
                lenght += 1
        return lenght+1
    else:
        return len(order)+1




