
# Currently reads in the names of the training files, test file and output file

import os
import sys
import numpy as np 
import pandas as pd 
from IPython.display import display 
from collections import Counter 
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
        #initial_probs = np.zeros((1, len(tags)), dtype='float32')
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
        display(em_df) #TODO: REMOVE IMPORT 
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
            prob_trellis, path_trellis = viterbi(sentence, list(tags), initial_probs, trans_matrix, em_matrix)
            prob_df = pd.DataFrame(prob_trellis, columns = sentence, index=list(tags))
            display(prob_df) #TODO REMOVE
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


def prob_t2_given_t1(t2, t1, tag_pairs, tags):
    pair_occurrences = Counter(tag_pairs)
    tag_1_pairs = 0
    for pair in pair_occurrences:
	    if pair[0] == t1:
		    tag_1_pairs += pair_occurrences[pair]
    
    pairs_of_tag1_tag2 = pair_occurrences[(t1, t2)]
    #if t1 != "<S>":
    # smoothing avoids 0 probabilities
    return (pairs_of_tag1_tag2 + EPSILON) / (tag_1_pairs + (EPSILON * len(tags)))
    """
    else:
        # no smoothing for initial probabilities.
        # We need 0 probabilities to show a sentence can't begin with punctuation
        if tag_1_pairs > 0:
            return pairs_of_tag1_tag2 / tag_1_pairs
        else:
             return 0
    """

def prob_word_given_tag(word, tag, word_tag_frequencies, tags):
    if (word,tag) in word_tag_frequencies:
        word_tag_count = word_tag_frequencies[(word,tag)]
    else:
        word_tag_count = 0
    tag_count = 0
    for pair in word_tag_frequencies:
        if pair[1] == tag:
            tag_count += word_tag_frequencies[pair]
    
    return  (word_tag_count + EPSILON) / (tag_count + (EPSILON * len(tags)))

def viterbi(words, tags, pi, A, B):
    """ The viterbi algorithm finds the most likely sequence of part-of-speech tags for a sentence.
    words : a list of words that form one sentence
    tags : a list of unique tags in training files
    pi: the initial state probabilities 
    A : transition matrix
    B: emission matrix  
    """
    prob_trellis = np.zeros((len(tags), len(words)), dtype='float32') # a probability table for words and their tags
    path_trellis = np.zeros((len(tags), len(words)), dtype='float32')
    emission_df = pd.DataFrame(B, columns = list(tags), index=list(words))
    # initialize
    sum = 0
    for t in range(len(tags)):
        #TODO: never before seen words, normalize probs??
        prob_trellis[t][0] = pi[t] * emission_df.loc[words[0], tags[t]]
        sum += prob_trellis[t][0]
        path_trellis[t][0] = -1
    if sum != 0:
        for t in range(len(tags)):
            # normalize first column
            prob_trellis[t][0] /= sum 

    # for states X_2 to X_T, find each current state's most likely prior state x
    for w in range(1, len(words)):
        sum = 0
        #TODO: what if there was a tag that we never saw in the training?
        for t in range(len(tags)):
            em_prob = emission_df.loc[words[w], tags[t]]
            # find the most likely tag for the previous word 
            x, max_prod = None, float('-inf') 

            for tag in range(len(tags)):
                # recall: log(a*b) = log a + log b
                #print(prob_trellis[tag][w-1],A[tag][t], em_prob )
                prod = prob_trellis[tag][w-1] * A[tag][t] * em_prob
                #prod =  np.log(prob_trellis[tag][w-1]) + np.log(A[tag][t]) + np.log(em_prob)
                print("product is: {}\n".format(prod))
                if prod > max_prod:
                    max_prod = prod
                    x = tag
            print("Tag_x is {}".format(tags[x]))

            prob_trellis[t][w] = max_prod
            path_trellis[t][w] = x
            sum += prob_trellis[t][w]
        if sum != 0:
            for t in range(len(tags)):
                # normalize 
                prob_trellis[t][w] /= sum 
    
    return prob_trellis, path_trellis 

def backward_pass(prob_trellis, path_trellis, tags, words):
    """ Returns the most likely sequence of tags for a sentence.
        Ex: [tag0, tag1, tag0, tag2] for the sentence ["Mary", "sees", "Will", "."]
    """
    
    # find the most likely tag for the last word in the sentence 
    tag_for_last_word = np.argmax(prob_trellis, axis=0)[-1]
    tag_sequence = [tag_for_last_word]
    curr_tag = tag_for_last_word 
    # go through path_trellis to find tags for previous words 
    for word_i in range(len(words)-1, 0, -1):
        prev_tag = path_trellis[int(curr_tag)][word_i]
        tag_sequence.insert(0, prev_tag)
        curr_tag = prev_tag

    return tag_sequence # a sequence of tag numbers


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Ouptut file: " + output_file)

    # Start the training and tagging operation.
    tag (training_list, test_file, output_file)
