
# Currently reads in the names of the training files, test file and output file

import os
import sys
import numpy as np 
import pandas as pd 
from IPython.display import display 
from collections import Counter 
from viterbi import *
from probabilities import * 
EPSILON = 0.001
def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file. Store information about examples seen in training files.
    

    for txt in training_list: 
        tags = set()  
        words = set() 
        word_tag_freq = {}
        tag_pairs = [] 
        file = open(txt, "r")
        prev = None 
        for line in file.readlines():
            info = line.strip().split()
            word, pos = info[0].lower(), info[2]
            tags.add(pos)
            words.add(word)
            if (word, pos) in word_tag_freq:
                word_tag_freq[(word, pos)] += 1
            else:
                word_tag_freq[(word,pos)] = 1
            curr = pos
            if prev is not None:
                tag_pairs.append((prev, curr))
            else:
                tag_pairs.append(("<S>", curr))
            if word == "!" or word == "." or word == "?":
                prev = None # split up file by sentence 
            else:
                prev = curr
        
        initial_probs = []
        trans_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
        for i, tag1 in enumerate(list(tags)):
            initial_probs.append(prob_t2_given_t1(tag1, "<S>", tag_pairs, tags))
            #initial_probs[0][i] = prob_t2_given_t1(tag1, "<S>", tag_pairs)
            for j, tag2 in enumerate(list(tags)):
                trans_matrix[i][j] = prob_t2_given_t1(tag2, tag1, tag_pairs, tags) 
        file.close()
        
        em_matrix = np.zeros((len(words), len(tags)), dtype='float32')
        for i, word in enumerate(list(words)):
            for j, tag in enumerate(list(tags)):
                em_matrix[i][j] = prob_word_given_tag(word, tag, word_tag_freq, tags)
        em_df = pd.DataFrame(em_matrix, columns = list(tags), index=list(words))
        display(em_df) 
        trans_df = pd.DataFrame(trans_matrix, columns = list(tags), index=list(tags))
        display(trans_df)
    # read test_file
    input_file = open(test_file, "r") 
    output = open(output_file, "w")
    sentence = []
   
    for line in input_file.readlines():
        info = line.strip().split()
        word = info[0]
        sentence.append(word)
        if word == "." or word == "!" or word == "?":
            prob_trellis, path_trellis = viterbi_algorithm(sentence, list(tags), initial_probs, trans_matrix, em_matrix)
            prob_df = pd.DataFrame(prob_trellis, columns = sentence, index=list(tags))
            display(prob_df) 
            path_df = pd.DataFrame(path_trellis, columns = sentence, index=list(tags))
            display(path_df)
            tag_seq = backward_pass(prob_trellis, path_trellis, list(tags), sentence)
            print(tag_seq, sentence)
            tags = list(tags)
            for i in range(len(sentence)):
                output.write("{} : {}\n".format(sentence[i], tags[int(tag_seq[i])] ))
            sentence = [] 
    input_file.close()
    output.close()




if __name__ == '__main__':
    
    # Tagger expects "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Ouptut file: " + output_file)

    # Start the training and tagging operation.
    tag (training_list, test_file, output_file)
