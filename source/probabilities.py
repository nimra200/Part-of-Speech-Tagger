EPSILON = 0.001
def prob_t2_given_t1(t2, t1, tag_pairs, tags):
    """ Determine the transition probability of tag2, given that tag1 was seen. 
    For example, if the previous tag <t1> was a noun, determine the probability that the tag <t2> is a verb.
    Parameters:
    t2: a Part of Speech (POS) tag (noun, verb, adverb, etc.)
    t1: a POS tag 
    tag_pairs: a list of all the tags that have appeared side-by-side in the training examples 
    tags: all the POS tags seen in the training examples
    """
    pair_occurrences = Counter(tag_pairs)
    tag_1_pairs = 0
    for pair in pair_occurrences:
	    if pair[0] == t1:
		    tag_1_pairs += pair_occurrences[pair]
    
    pairs_of_tag1_tag2 = pair_occurrences[(t1, t2)]
    
    # smoothing avoids 0 probabilities
    return (pairs_of_tag1_tag2 + EPSILON) / (tag_1_pairs + (EPSILON * len(tags)))


def prob_word_given_tag(word, tag, word_tag_frequencies, tags):
    """Determine the emission probability of a word, given that a tag was seen. 
    For example, if we know that a word is a noun, what is the probability that the word is "dog"? 
    Parameters:
    word: a word in the English language
    tag: a POS tag
    word_tag_frequencies: a dictionary storing how frequently a word appeared with a POS tag in training files. 
   			  For example, the word "dog" appeared as a noun 5 times. 
    tags: a list of all tags seen in the training. 
    """
    if (word,tag) in word_tag_frequencies:
        word_tag_count = word_tag_frequencies[(word,tag)]
    else:
        word_tag_count = 0
    tag_count = 0
    for pair in word_tag_frequencies:
        if pair[1] == tag:
            tag_count += word_tag_frequencies[pair]
    
    return  (word_tag_count + EPSILON) / (tag_count + (EPSILON * len(tags)))
