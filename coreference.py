import sys, nltk
'''
Coreference Resolution
Created on Oct 25, 2012

@author: John Wells
@author: Joel Hough

'''
def input_listfile(listfile):
    raw_inputs = []
    
    if(listfile is None):
        sys.exit("No listfile detected. Exiting.")
    else:
        fin = open(listfile)
    for line in fin.readlines():
        # Throw all text into key/value pairs? Key being the filename, since it's used later.
        pass
    
    return raw_inputs
    
def np_chunker(raw_inputs):
    """
    Given raw inputs, break up the input into NP chunks
    """
    pass

def hobbs_distance():
    pass

def sentence_distance():
    pass

def gender_agreement():
    pass

def number_agreement():
    pass

def output_response():
    pass

def edit_distance(anaphor, antecedent):
    """
    Defined as the minimum number of character editing ops necessary to turn one string into another. 
    TODO: Is this the same as the Levenshtein function Joel wrote last spring?
    """
    pass

def is_appositive(anaphor, antecedent):
    """
    
    """
    pass

def linguistic_form(anaphor):
    """
    Returns the form of the potential anaphor NP_j.
    @param anaphor: Targeted potential anaphor of known antecedent. 
    @return form: [proper_name, definite_description, indefinite_NP, pronoun]
    """ 
    pass

def hobbs_algorithm():
    """
    1. Begin at the NP mode immediately dominating the pronoun.
    2. Go up the tree to the first NP or S node encountered. Call this node X and call the path used to reach it P.
    3. Traverse all branches below node X to the left of path p in a left-to-right BFS fashion. Propose as the antecedent any encountered NP node that has an NP or S node between it and X.
    4. If node X is the highest S node in the sentence, traverse the surface parse trees of previous sentences in the text in order of recency, most recent first. 
        Each tree is traversed in a left-to-right BFS and when an NP node is encountered it is proposed as antecedent. 
        If X is not the highest S node in the sentence, continue to step 5.
    5. From node X, go up the tree to the first NP or S node encountered. Call this new node X and the path traversed to reach it P.
    6. If X is an NP node and if the path P to X did not pass through the Nominal node that X immediately dominates, propose X as the antecedent.
    7. Traverse all branches below node X to the left of path P in a left to right BFS. Propose any NP node encountered as the antecedent.
    8. If X is an S node, traverse all branches of node X to the right of path P in a left-to-right BFS, but do not go below any NP or S node encountered. Propose any NP node encountered as the antecedent.
    9. Go to step 4.
    """
    pass

def centering_algorithm():
    """
    Reference: Centering theory, entity-based coherence. Page 706.
    
    RULES: 
        1. If any element of Cf(Un) is realized by a pronoun in utterance Un+1 then Cb(Un+1) must be realized as a pronoun also.
        2. Transition states are ordered. Continue is preferred to Retain is preferred to Smooth-Shift is preferred to Rough-Shift.
        
    Algorithm:
        1. Generate possible (Cb,Cf) combinations for each possible set of reference assignments.
        2. Filter by constraints. For example, syntactic coreference constraints, selectional restrictions, centering rules, and constraints.
        3. Rank by transition orderings.
    """
    pass

def log_linear():
    """
    Supervised machine learning approach. 
    Train a log-linear classifier on a corpus in which the antecedents are marked for each pronoun.
    NOTE: Probably shouldn't implement this one.
    """
    pass


def main():
    list_file = None
    output_dir = None
    if(len(sys.argv) == 3):
        list_file = sys.argv[2]
        output_dir = sys.argv[3]
#    else:
#        sys.exit("Incorrect number of input files specified. Aborting")
    tagged_stories = input_listfile(list_file)

if __name__ == '__main__':
    main()
