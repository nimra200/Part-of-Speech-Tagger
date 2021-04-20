def viterbi_algorithm(words, tags, pi, A, B):
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
        
        for t in range(len(tags)):
            em_prob = emission_df.loc[words[w], tags[t]]
            # find the most likely tag for the previous word 
            x, max_prod = None, float('-inf') 

            for tag in range(len(tags)):
                # recall: log(a*b) = log a + log b
                #print(prob_trellis[tag][w-1],A[tag][t], em_prob )
                prod = prob_trellis[tag][w-1] * A[tag][t] * em_prob
                
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
        where tag_i represents noun, verb, etc. 
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
