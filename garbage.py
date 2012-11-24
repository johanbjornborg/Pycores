import os
from pprint import pprint as pp
import sys, nltk, re
import nltk.chunk
import itertools
from inspect import Attribute
from pickle import FALSE

def test_xml():
    rawtext = open("devset/input/1.crf").read() 
    res = [({"ID":int(m[0]), 'value':m[1]}) for m in re.findall(r"<COREF ID=\"(\d+)\">(.*?)</COREF>", rawtext)]
    return res

def test_nltk(anaphora):
    rawtext = open("devset/input/1.crf").read()
    sentences = nltk.sent_tokenize(rawtext)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
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
        for i, _u in u:
            for nn in t[:i]:
                if "NN" in nn or "NNS" in nn or "NNP" in nn or "NNPS" in nn:
                    for a in anaphora:
                        if a['value'] in _u[0]: # Check for gender/number agreement here.
                            a['pro_ants'] = nn[0]
                           
    pp(anaphora)
    return anaphora

def dummy():
    rawtext = open("devset/input/1.crf").read() 
    res = [({"ID":int(m[0]), 'value':m[1]}) for m in re.findall(r"<COREF ID=\"(\d+)\">(.*?)</COREF>", rawtext)]
    return res

def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag == tag_prefix)  
    return dict((tag, cfd[tag].keys()[:]) for tag in cfd.conditions())

def appositive(sentences, anaphor):

    t = chunker.parse(sentences)
    a = [(a[0]) for a in anaphor if 'PRP' in a[1]]
    if a: # If there exists a pronoun.
        for u in reversed(t):
            try:
                if u.node == 'NP': #If we have an NP, check gender/number agreement.
                    if a[0] in ['he', 'his', 'him']:
                        male = [n for n in names.words('male.txt') if n in ''.join([_u[0] for _u in u])]
                        if male:
                            return ' '.join([_u[0] for _u in u.leaves() if 'NNP' in _u])
                    elif a[0] in ['she', 'hers', 'her']:
                        female = [n for n in names.words('female.txt') if n in ''.join([_u[0] for _u in u])]
                        if female:
                            return ' '.join([_u[0] for _u in u.leaves() if 'NNP' in _u])
                    elif a[0] in ['it', 'its', 'itself']:
                        neuter = [_u[0] for _u in u.leaves() if 'NNP' not in _u]
                        if neuter:
                            return ' '.join(neuter)               
            except AttributeError:
                continue

        
    
           
if __name__ == '__main__':
#    anaphora = test_xml()
#    test_nltk(anaphora)
    names = nltk.corpus.names
    grammar = r"""
NP: {<DT|PP\$>?<JJ>*<NN|NNS>} # chunk determiner/possessive, adjectives and nouns
{<NNP>+} # chunk sequences of proper nouns

"""
    chunker = nltk.RegexpParser(grammar)
    sentence = "As senior director of UAL, and a member of the executive committee of its board, I am appalled at the inaccuracies and anti-management bias in the Journal's April 17 article about Richard Ferris, "
    ana = "he"
    sentences = nltk.pos_tag(sentence.split())
    anaphor = nltk.pos_tag(ana.split())
    print appositive(sentences, anaphor)
