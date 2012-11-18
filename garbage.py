import os
from pprint import pprint as pp
import sys, nltk, re

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

def traverse(t, anaphora):
    try:
        t.node
    except AttributeError:
        return
    else:
        if t.node == 'NP': 
            print ' '.join([(leaf[0]) for leaf in t.leaves()])
#            l = t.leaves()  # or do something else
#            leaf = [(x[0]) for x in l]
#            __list = ' '.join(leaf)
#            anas = [(anaphor) for anaphor in anaphora if anaphor['value'] in leaf]
#            if anas:
#                print anas, __list
        else:
            for child in t:
                traverse(child, anaphora)
                
def test_xml():
    rawtext = open("devset/input/1.crf").read() 
    res = [({"ID":int(m[0]), 'value':m[1]}) for m in re.findall(r"<COREF ID=\"(\d+)\">(.*?)</COREF>", rawtext)]
    return res

def test_nltk(anaphora):
    """
    Given: Parsed text, optionally POS tagged.
    """
    rawtext = open("devset/raw/1.txt").read()
    sentences = nltk.sent_tokenize(rawtext)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
#    for s in sentences:
#        for a in anaphora:
#            if a['value'] in s[0]:
#                print a['value'], ":\t", s
    grammar = r"""
NP: {<DT|PP\$>?<JJ>*<NN|NNS>} # chunk determiner/possessive, adjectives and nouns
{<NNP>+} # chunk sequences of proper nouns

"""
    chunker = nltk.RegexpParser(grammar)
    parsed = chunker.batch_parse(sentences)

    for sent in parsed:
        t = chunker.parse(sent)
        t = t.flatten()
        u = [(i, val) for i, val in enumerate(t) if "PRP" in val]
        for i,_u in u:
            for nn in t[:i]:
                if "NN" in nn or "NNS" in nn or "NNP" in nn or "NNPS" in nn:
                    for a in anaphora:
                        if a['value'] in _u[0]: # Check for gender/number agreement here.
                            a['pro_ants'] = nn[0]
                           
    pp(anaphora)
    return anaphora

if __name__ == '__main__':
    anaphora = test_xml()
    test_nltk(anaphora)