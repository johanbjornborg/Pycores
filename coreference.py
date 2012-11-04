import sys, nltk, re
'''
Coreference Resolution
Created on Oct 25, 2012

# REQUIRED NLTK PACKAGES:
# punkt
# maxent_treebank_pos_tagger

@author: John Wells
@author: Joel Hough

'''
def input_listfile(listfile):
    
    file_list = []
    
    if(listfile is None):
        sys.exit("No listfile detected. Exiting.")
    else:
        fin = open(listfile)
    for line in fin.readlines():
        file_list.append(line)
    
    return file_list

def get_anaphora(crf_file):
    """
    Obtains all valid XML-tagged anaphora from a given input file.
    @param crf_file: Valid XML-tagged (<COREF ID=\d> </COREF>) file.
    """
    return [({"ID":int(m[0]), "ana":m[1]}) for m in re.findall(r"<COREF ID=\"(\d+)\">(.*?)</COREF>", crf_file)]

def strip_xml(crf_file):
    """
    Strips the XML-style tags from a given .crf input crf_file path.
    @param crf_file: A valid .crf crf_file path.
    """
    #rawtext = open(crf_file).read()
    return re.sub(r"<COREF ID=\"\d+\">|</COREF>|<.*>", '', open(crf_file).read()) # Let's be concise, shall we?

def tagger(chunked, anaphora):
    """
    Comprehensive tagging function. Does all kinds of neat stuff.
    Tagged antecedents should be mapped with respect to their coreferent anaphora ID (number), a unique alpha character, and the word.
    {'id' = <anaphor_id>, 'uid' = [A-Z], 'antecedent' = <word>}
    """
    pass

def np_chunker(clean_text):
    """
    Given an XML-free string, break up the input into NP chunks.
    @param clean_text: 
    """
    
    sentences = nltk.sent_tokenize(clean_text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    grammar = r"""
NP: {<DT|PP\$>?<JJ>*<NN|NNS>} # chunk determiner/possessive, adjectives and nouns
{<NNP>+} # chunk sequences of proper nouns
"""
    chunker = nltk.RegexpParser(grammar)
    return chunker.batch_parse(sentences)
#    for sent in sentences:
#        t = chunker.parse(sent)
#        traverse(t)

def pronoun_matcher():
    """
    Relevant factors for pronoun resolution:
        Recency: (Anaphor should always point to the most recent applicable antecedant.)
        Grammatical Role: Entities introduced in the subject are more salient than those in the object, and so forth.
            (This could be applied to pronouns over multiple sentences.)
            (Bill went to the bar with Jim. He called for a glass of rum.) <- He refers to Bill.
        Repeated mention: Ideas focused on previously are more likely to be focused on again.
            (Bill had done some stuff earlier. He walked over to the bar. Jim went with him. He ordered a glass of rum.)
            While either is valid, it's more likely that He referes to Bill.
        Parallelism: A little too complex.
        Verb Semantics: Certain verbs place a semantically oriented emphasis on one of their argument positions. This can have the
            effect of biasing the manner in which the subsequent pronouns are interpreted.
            (John telephoned Bill. He lost the laptop) (John criticized Bill. He lost the laptop)
            This is defined as "implicit causality"
    """
    pass

def string_matcher(chunked_text, anaphora):
    """
    Possible uses for string matching:
    Exact matches of Proper Names
    Common NPs can be matched, but can be risky.
    Partial String matching is more risky.
    Titles, abbreviations, and acronym recognition.
    """
    
    #Proper names:
    
    #Common NPs
    
    #Partial Matches
    
    #Advanced:
    pass

def hobbs_distance():
    pass

def sentence_distance():
    """
    """
    pass

def gender_agreement():
    """
    Referents must agree with the gender specified by the referring expression. 
    Animate entities are always related to male and female pronouns.
    Inanimate entities are nonpersonal.
    """
    pass

def number_agreement():
    """
    Generally, the number of items in the referent must agree with the referring expression.
    John has a thing. They are red. # Does not agree.
    John has a thing. It is red. # Does agree.
    Plural = Plural, Singular = Singular.
    """
    pass

def output_response(anaphora, tagged_antecedents, output_filename):
    """
    Given a list of Tagged Anaphora and antecedents, and an original input file, create a tagged output file.
    Note: The references do not need to be placed back into the original story. They can be placed side-by-side.
    The grading program only looks at XML tags. It doesn't care about the background story.
    """
    # Example output: <COREF ID="A">antecedent</COREF> <COREF ID="1" REF="A">anaphora</COREF>
    fout = open(output_filename, "rw+")
    # Do something else.
    
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
    
def traverse(t, anaphora):
    """
    11/2: This concatenates each NP in the tree, and attempts to find a match in the given anaphora hash.
    Tree traversal function for extracting NP chunks in a RegexParse'd tree.
    @param t: POS tagged and parsed chunk.
    """
    try:
        t.node
    except AttributeError:
        return
    else:
        if t.node == 'NP': 
#            print ' '.join([(leaf[0]) for leaf in t.leaves()])
            l = t.leaves()  # or do something else
            leaf = [(x[0]) for x in l]
            __list = ' '.join(leaf)
            anas = [(anaphor) for anaphor in anaphora if anaphor['ana'] in leaf]
            if anas:
                print anas, __list
            
                
               
        else:
            for child in t:
                traverse(child, anaphora)
        
                
#==============================================================================
# Test Functions
#==============================================================================
def test_xml():
    res = []
    rawtext = open("devset/input/1.crf").read() 
    mat = re.findall(r"<COREF ID=\"(\d+)\">(.*?)</COREF>", rawtext)
    #for m in mat:
    res = [({"ID":int(m[0]), "ana":m[1]}) for m in re.findall(r"<COREF ID=\"(\d+)\">(.*?)</COREF>", rawtext)]
    return res


def test_nltk():
    t = nltk.Tree
    anaphora = test_xml()
#    print anaphora
    rawtext = open("devset/input/1.crf").read()
    sentences = nltk.sent_tokenize(rawtext)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    for s in sentences:
        for a in anaphora:
            if a['ana'] in s[0]:
                print a['ana'], ":\t", s
    grammar = r"""
NP: {<DT|PP\$>?<JJ>*<NN|NNS>} # chunk determiner/possessive, adjectives and nouns
{<NNP>+} # chunk sequences of proper nouns

"""
    np_list = []
    chunker = nltk.RegexpParser(grammar)
    parsed = chunker.batch_parse(sentences)
    print parsed

    
    for sent in parsed:
        traverse(sent, anaphora)



        
    
#===============================================================================
# Main
#===============================================================================
def main():
    list_file = None
    output_dir = None
    
    #Rudimentary arg parsing. Since it's defined in the spec I didn't see a need to get fancy.
    if(len(sys.argv) == 3):
        list_file = sys.argv[2]
        output_dir = sys.argv[3]
#    else:
#        sys.exit("Incorrect number of input files specified. Aborting")

    file_list = input_listfile(list_file) # Obtain the list of filenames to coreference-ate 

    for crf_file in file_list:
        tagged_anaphora = get_anaphora(crf_file) # Get the anaphora and ID's from file.
        clean_text = strip_xml(crf_file) # Remove XML tagging
        chunked = np_chunker(clean_text) # Chunk the text into Trees.
        tagged_antecedents = tagger(chunked) # Start the tagger. 
        
    

if __name__ == '__main__':
#    main()
    test_nltk()
#    test_xml()


