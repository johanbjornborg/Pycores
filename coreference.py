import os
from pprint import pprint as pp
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
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.punkt.PunktWordTokenizer()
lemmatizer = nltk.WordNetLemmatizer()
tagger = nltk.RegexpTagger([(r'coref_tag_beg_.*', 'CRB'), 
                            (r'coref_tag_end_.*', 'CRE'),
                            (r'\$[0-9]+(.[0-9]+)?', 'NN')], backoff=nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle'))
names = nltk.corpus.names
coref_tag_re = r'(?is)<COREF ID="(\w+)">(.*?)</COREF>'
coref_token_re = r'(?is)coref_tag_beg_(\w+)(.*?)coref_tag_end_\1'
chunker_grammar = r"""
NP:
{<CRB>(<.*>+?)<CRE>}
{<DT|PP\$>?<JJ>*<NN.*>+} # chunk determiner/possessive, adjectives and nouns
"""
chunker = nltk.RegexpParser(chunker_grammar)

def chunk(sentence):
    return chunker.parse(sentence)
    
def lemmatize(word):
    return lemmatizer.lemmatize(word)

def pos_tag(text):
    return tagger.tag(text)

def word_tokenize(text):
    return word_tokenizer.tokenize(text)

def sentence_tokenize(text, no_break_zones=[]):
    spans = sentence_tokenizer.span_tokenize(text)
    return sentence_tokenizer._realign_boundaries(text[slice(start, end)] for start, end in adjust_spans(spans, no_break_zones))

def adjust_spans(spans, no_break_zones):
    def valid_break(pos):
        for (start, end) in no_break_zones:
            if pos >= start and pos <= end:
                return False
        return True
    
    new_spans = []
    new_start = 0
    for start, end in spans:
        if valid_break(end):
            yield (new_start, end)
            new_start = end

def input_listfile(listfile):
    
    file_list = []
    
    if(listfile is None):
        sys.exit("No listfile detected. Exiting.")
    else:
        fin = open(listfile)
    for line in fin.readlines():
        file_list.append(line)
    
    return file_list

def count_sentences(text):
    """
    An _approximate_ count of sentences in the text.
    """
    global sentence_tokenizer
    return len(sentence_tokenizer.tokenize(text.strip()))

def get_anaphora(text):
    """
    Get a list of information about anaphora in the given text
    @param text: text with anaphors marked with coref_tag_* tokens.
    """
    return [{'ID':m.groups()[0],
             'value':m.groups()[1],
             'position': m.start()} for m in re.finditer(coref_token_re, text)]

def get_anaphora_with_periods(anaphora):
    return [(a['value']) for a in anaphora if '.' in a['value']]    

def strip_xml(crf_file):
    """
    Strips the XML-style tags from a given .crf input crf_file path.
    @param crf_file: A valid .crf crf_file path.
    """
    #rawtext = open(crf_file).read()
    return re.sub(r"<COREF ID=\"\d+\">|</COREF>|<.*>", '', open(crf_file).read()) 


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

def pronoun_matcher(potential_antecedent, anaphor):
    sentence = anaphor['sentence']
#    print potential_antecedent['value'], sentence
    global names
#    t = chunker.parse(sentences)
    a = [(a[0]) for a in anaphor['value'] if 'PRP' in a[1]]
    if a: # If there exists a pronoun.
        for u in reversed(sentence):
            try:
                if u.node == 'NP': #If we have an NP, check gender/number agreement.
                    # Male pronoun agreement. Returns first agreement found.
                    if a[0] in ['he', 'his', 'him']:
                        male = [n for n in names.words('male.txt') if n in ''.join([_u[0] for _u in u])]
                        if male:
                            return ' '.join([_u[0] for _u in u.leaves() if 'NNP' in _u])
                    # Female pronoun agreement. Returns first agreement found.
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

def is_appositive(potential_antecedent, anaphor):
    
    try:
        sentence = anaphor['sentence']
        #If the chunk prior to the anaphor location is a NP, verify that it also contains a comma.
        #        print anaphor
        if sentence[-1].node == 'NP': # Probably should check the anaphor to ensure that is in fact a NP as well.
            appos = [ap for ap in sentence[-1].leaves() if ',' in ap[0]]
            if appos:
                return True
        else:
            return False
    except AttributeError:
        return False
    except KeyError:
        return False

    

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

   
def traverse(t, anaphora):
    """
    Tree traversal function for extracting NP chunks in a RegexParsed tree.
    @param t: POS tagged and parsed chunk.
    """
    try:
        t.node
    except AttributeError:
        return
    else:
        if t.node == 'NP': 
            l = t.leaves()  # or do something else
            leaf = [(x[0]) for x in l]
            __list = ' '.join(leaf)
            anas = [(anaphor) for anaphor in anaphora if anaphor['value'] in leaf]
            if anas:
                print anas, __list     

        else:
            for child in t:
                traverse(child, anaphora)

def each_with_tail(seq):
    i = 0
    l = list(seq)
    while (l[i:]):
        i += 1
        yield (l[i - 1], l[i:])

def any_word_matches_p(anaphor, potential_antecedent):
    return any(word for word in anaphor['value'].split() if lemmatize(word.lower()) in map(lambda w: lemmatize(w.lower()), potential_antecedent['value'].split()))

def sentence_distance(anaphor, potential_antecedent):
    return anaphor['position'][0] - potential_antecedent['position'][0]

def distance(anaphor, potential_antecedent):
    a_sent, a_phrase = anaphor['position']
    b_sent, b_phrase = potential_antecedent['position']
    return (a_sent - b_sent, a_phrase - b_phrase)

def features(anaphor, potential_antecedent):
    return {
        'REF': potential_antecedent['ID'],
        'word_match': any_word_matches_p(anaphor, potential_antecedent),
        'sentence_distance': sentence_distance(anaphor, potential_antecedent),
        'distance': distance(anaphor, potential_antecedent),
        'is_appositive' : is_appositive(potential_antecedent, anaphor),
#        'pronoun' : pronoun_matcher(potential_antecedent, anaphor)
        }

def coreferent_pairs_features(anaphora):
    refs = dict()
    for anaphor, potential_antecedents in each_with_tail(sorted(anaphora, key=lambda a:a['position'], reverse=True)):
        if not anaphor['is_anaphor']:
            continue
        refs[anaphor['ID']] = [features(anaphor, potential_antecedent) for potential_antecedent in potential_antecedents]
    return refs

def feature_resolver(anaphora):
    features = coreferent_pairs_features(anaphora)
    for id in features:
        matches = filter(lambda f: f['word_match'], features[id])
        if matches:
            yield {'ID': id, 'REF': min(matches, key=lambda f: f['distance'])['REF']}

def update_refs(text, refs):
    """
    Given a list of Tagged Anaphora and antecedents, and an original input file, create a tagged output file.
    """
    new_text = text
    for ref in refs:
        new_text = new_text.replace('<COREF ID="{0[ID]}">'.format(ref), '<COREF ID="{0[ID]}" REF="{0[REF]}">'.format(ref))

    return new_text

def replace_coref_tags_with_tokens(text):
    return re.sub(coref_tag_re, r' coref_tag_beg_\1 \2 coref_tag_end_\1 ', text)

def replace_coref_tokens_with_tags(text):
    return re.sub(coref_token_re, r'<COREF ID="\1">\2</COREF>', text)

def no_break_zones(text):
    for match in re.finditer(coref_token_re, text):
        yield match.span()

def coref_abbrs(text):
    for match in re.finditer(coref_token_re, text):
        for word in word_tokenize(match.groups()[1]):
            if word[-1] == '.':
                yield word

def read_text(file):
    return open(file, 'r').read()

def filename(file):
    return os.path.splitext(os.path.basename(file))[0]

def resolve_files(files, response_dir_path):
    # extract text, preprocess, and train sentence_tokenizer
    to_resolve = []
    for file in files:
        name = filename(file)
        text = read_text(file)
        detagged_text = re.sub(r'(?is)</?TXT>', '', replace_coref_tags_with_tokens(text))
        #print 'TEXT'
        #print detagged_text
        to_resolve.append((name, detagged_text))
        abbrs = coref_abbrs(detagged_text)
        #print 'ABBRS'
        #print list(abbrs)
        # add abbrs to sentence_tokenizer

    for name, text in to_resolve:
        class Gensym:
            i = 0
            def __call__(self):
                self.i += 1
                return 'X{0}'.format(self.i)

        gensym = Gensym()
        coref_zones = no_break_zones(text)
        #print 'COREF ZONES'
        #print list(coref_zones)
        corefs = []
        np_tagged_sentences = []
        sentences = sentence_tokenize(text, coref_zones)
        for i_sentence, sentence in enumerate(sentences):
            tokenized_sentence = word_tokenize(sentence)
            tagged_sentence = pos_tag(tokenized_sentence)
            #            print 'CHUNKED'
            chunked_sentence = chunk(tagged_sentence)
            #            print chunked_sentence
            
            for i_subtree, subtree in enumerate(chunked_sentence.subtrees(filter=lambda s: s.node == 'NP')):
                word, tag = subtree[0]
                if tag == 'CRB':
                    ident = re.match(r'coref_tag_beg_(\w+)', word).group(1)
                    anaphor = True
                    del subtree[0]
                    del subtree[-1]
                else:
                    ident = gensym()
                    anaphor = False
                tagged_value = subtree[:]
                value = ' '.join(word for word, tag in tagged_value)
                subtree.insert(0, ('coref_tag_beg_{0}'.format(ident), 'CRB'))
                subtree.append(('coref_tag_end_{0}'.format(ident), 'CRE'))

                def move_last_period_out_of_coref_tag(tree):
                    tag = tree[-1]
                    last_word = tree[-2]
                    #print 'TAG:{0}, WORD:{1}'.format(tag, last_word)
                    if last_word[0][-1] == '.':
                        tree[-2] = (last_word[0][:-1], last_word[1])
                        tree[-1] = (tag[0] + '.', tag[1])
                move_last_period_out_of_coref_tag(subtree)
                corefs.append({'ID': ident,
                               'position': (i_sentence, i_subtree),
                               'is_anaphor': anaphor,
                               'value': value,
                               'tagged_value': tagged_value,
                               'tokenized_sentence': tokenized_sentence,
                               'tagged_sentence': tagged_sentence,
                               'chunked_sentence': chunked_sentence,
                               'sentence': sentence
                               })
            #print 'TREE'
            #print chunked_sentence
            #print 'FLAT'
            #print chunked_sentence.leaves()
            np_tagged_sentences.append(' '.join(word for word, tag in chunked_sentence.leaves()))
        #print 'TAGGED'
        tagged_text = '<TXT>' + replace_coref_tokens_with_tags("\n".join(np_tagged_sentences)) + '</TXT>'
        refs = feature_resolver(corefs)
        resolved_text = update_refs(tagged_text, refs)
        #print resolved_text
        #            print list(refs)
        #     anaphora.append(anaphor)
        output_path = os.path.join(response_dir_path, name + '.response')
        open(output_path, 'w').write(resolved_text)
                
#==============================================================================
# Test Functions
#==============================================================================
def test_xml():
    res = []
    rawtext = open("devset/input/1.crf").read() 
    mat = re.findall(r"<COREF ID=\"(\d+)\">(.*?)</COREF>", rawtext)
    #for m in mat:
    res = [({"ID":int(m[0]), 'value':m[1]}) for m in re.findall(r"<COREF ID=\"(\d+)\">(.*?)</COREF>", rawtext)]
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
            if a['value'] in s[0]:
                print a['value'], ":\t", s
    grammar = r"""
NP: {<DT|PP\$>?<JJ>*<NN|NNS>} # chunk determiner/possessive, adjectives and nouns
{<NNP>+} # chunk sequences of proper nouns

"""

    np_list = []
    chunker = nltk.RegexpParser(grammar)
    parsed = chunker.batch_parse(sentences)
    print parsed    
    for sent in parsed:
        t = chunker.parse(sent)
        pp(t)
        traverse(sent, anaphora)

       
#===============================================================================
# Main
#===============================================================================
def main():
    listfile_path = sys.argv[1]
    response_dir_path = sys.argv[2]

    files = [l.strip() for l in open(listfile_path, 'r').readlines()]

    resolve_files(files, response_dir_path)
    # file_list = input_listfile(list_file) # Obtain the list of filenames to coreference-ate 

    # for crf_file in file_list:
    #     tagged_anaphora = get_anaphora(crf_file) # Get the anaphora and ID's from file.
    #     clean_text = strip_xml(crf_file) # Remove XML tagging
    #     chunked = np_chunker(clean_text) # Chunk the text into Trees.
    #     tagged_antecedents = tagger(chunked) # Start the tagger. 
        
    

if __name__ == '__main__':
    main()
#    test_nltk()
#    test_xml()


