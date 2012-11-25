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

def get_anaphora_with_periods(anaphora):
    return [(a['value']) for a in anaphora if '.' in a['value']]        
    
def fix_span_breaks(str_idx, pairs):
    span = []
    i = 0
    j = 0
    low = 0
    hi = str_idx[-1][1]
    p = pairs[j]
    s = str_idx[i]
    low,hi = set(s)
    for p in pairs:
        while max(s) < max(p):
            i += 1
            s = str_idx[i]
            hi = s[1]
        inter = list(set(range(low,hi)) & set(p))
        print inter
        if len(inter) == 2: 
            span.append((low,hi))
    print span

def edit_distance(anaphor, potential_antecedent):
    ana = anaphor['value'].split()
    ant = potential_antecedent['value'].split()
    dist = 0
    if len(ana) < len(ant):
        return edit_distance(potential_antecedent, anaphor)
   
    for i in range(0,len(ana)):
        if i >= len(ant):
            print "empty"
            dist += levenshtein(ana[i], "")
        else:
            dist += levenshtein(ana[i], ant[i])
    return dist 

def levenshtein(s1, s2):
    if len(s2) == 0:
        return len(s1)
    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

if __name__ == '__main__':
    s1 = {'value' : "Creutzfeldt-Jakob"}
    s2 = {'value' : "Creutzfeldt-Jacob Disease"}
    print edit_distance(s1, s2)
#    anaphora = test_xml()
#    get_anaphora_with_periods(anaphora)
#    test_nltk(anaphora)
#    names = nltk.corpus.names
#    grammar = r"""
#NP: {<DT|PP\$>?<JJ>*<NN|NNS>} # chunk determiner/possessive, adjectives and nouns
#{<NNP>+} # chunk sequences of proper nouns
#
#"""
#    chunker = nltk.RegexpParser(grammar)
#    sentence = "As senior director of UAL, and a member of the executive committee of its board, I am appalled at the inaccuracies and anti-management bias in the Journal's April 17 article about Richard Ferris, "
#    ana = "he"
#    sentences = nltk.pos_tag(sentence.split())
#    anaphor = nltk.pos_tag(ana.split())
#    print appositive(sentences, anaphor)
