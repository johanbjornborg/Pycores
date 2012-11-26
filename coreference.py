import os
from pprint import pprint as pp
from math import fabs
import sys, nltk, re
import xml.sax.saxutils as saxutils

'''
Coreference Resolution
Created on Oct 25, 2012

# REQUIRED NLTK PACKAGES:
# punkt
# maxent_treebank_pos_tagger

@author: John Wells
@author: Joel Hough

'''
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.punkt.PunktWordTokenizer()
lemmatizer = nltk.WordNetLemmatizer()
tagger = nltk.RegexpTagger([(r'.*coref_tag_beg_.*', 'CRB'), 
                            (r'.*coref_tag_end_.*', 'CRE'),
                            (r'\$[0-9]+(.[0-9]+)?', 'NN')], backoff=nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle'))
names = nltk.corpus.names
coref_tag_re = r'(?is)<COREF ID="(\w+)">(.*?)</COREF>'
coref_token_re = r'(?is)coref_tag_beg_(\w+)_(.*?)coref_tag_end_\1_'
chunker_grammar = r"""
NP:
{<CRB>(<.*>+?)<CRE>}
{<DT|PP\$>?<JJ>*<NN.*>+} # chunk determiner/possessive, adjectives and nouns
#{<WP.*>}
{<PRP.*>}
"""
chunker = nltk.RegexpParser(chunker_grammar)
titles = ['Mrs', 'Mr', 'Ms', 'Prof', 'Dr', 'Gen', 'Rep', 'Sen', 'St', 'Sr', 'Jr', 'PhD', 'MD', 'BA', 'MA', 'DDS']
stopwords = nltk.corpus.stopwords.words('english') + titles
male_pronouns = ['he', 'him', 'himself']
female_pronouns = ['she', 'her', 'herself']
neuter_pronouns = ['it', 'itself', 'they']
plural_pronouns = ['we']

def chunk(sentence):
    return chunker.parse(sentence)

@Memoize
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
        #print list(no_break_zones)
        for start, end in no_break_zones:
            if pos >= start and pos <= end:
                return False
        return True
    
    new_start = -1
    for start, end in spans:
        if new_start == -1:
            new_start = start
        if valid_break(end):
            yield (new_start, end)
            new_start = -1

def get_anaphora(text):
    """
    Get a list of information about anaphora in the given text
    @param text: text with anaphors marked with coref_tag_* tokens.
    """
    return [{'ID':m.groups()[0],
             'value':m.groups()[1],
             'position': m.start()} for m in re.finditer(coref_token_re, text)]

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
        ana_i, ana_j = anaphor['position']
        ant_i, ant_j = potential_antecedent['position']
        if (ana_i == ant_i) and fabs(ana_j - ant_j) == 1:
            male = [n for n in names.words('male.txt')]
            female = [n for n in names.words('female.txt')]
            _names = male + female
            appos = []
            for ana in anaphor['value'].split():
                appos = [ana for m in _names if m in ana]
                if appos:
                    print appos
                    return True
            for ant in potential_antecedent['value'].split():
                appos = [ant for m in _names if m in ant]
                if appos:
                    print appos
                    return True
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

def edit_distance(anaphor, potential_antecedent):
    ana = anaphor['value'].split()
    ant = potential_antecedent['value'].split()
    dist = 0
    if len(ana) < len(ant):
        return edit_distance(potential_antecedent, anaphor)

    for i in range(0, len(ana)):
        if i >= len(ant):
            dist += levenshtein(ana[i], "")
        else:
            dist += levenshtein(ana[i], ant[i])
    return dist 

##
def levenshtein(s1, s2):
    if len(s2) == 0:
        return len(s1)
    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 
            deletions = current_row[j] + 1       
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def each_with_tail(seq):
    i = 0
    l = list(seq)
    while (l[i:]):
        i += 1
        yield (l[i - 1], l[i:])

def split_and_strip(text):
    return re.findall(r'\w+', text)

def without_stop_words(words):
    def not_stop_word(word):
        return not word.lower() in ' '.join(stopwords).lower()
    return filter(not_stop_word, words)

def important_words(text):
    return set(without_stop_words(split_and_strip(text)))

def all_words_in_antecedent_p(anaphor, potential_antecedent):
    ana_set = important_words(anaphor['value'])
    return len(ana_set) > 0 and len(ana_set - important_words(potential_antecedent['value'])) == 0

def stupid_pronoun_match_p(anaphor, potential_antecedent):
    ana = split_and_strip(anaphor['value'].lower())
    ant = split_and_strip(potential_antecedent['value'].lower())
    if len(ana) == 0 or len(ant) == 0:
        return False
    for ps in [male_pronouns, female_pronouns, neuter_pronouns, plural_pronouns]:
        if ana[0] in ps and ant[0] in ps:
            return True
    return False

def string_match_p(anaphor, potential_antecedent):
    return anaphor['value'] == potential_antecedent['value']

def any_word_matches_p(anaphor, potential_antecedent):
    return any(word for word in important_words(anaphor['value']) if lemmatize(word.lower()) in map(lambda w: lemmatize(w.lower()), important_words(potential_antecedent['value'])))

def it_referring_the_p(anaphor, potential_antecedent):
    return potential_antecedent['value'].lower().startswith('the') and anaphor['value'].lower() == 'it'

def sentence_distance(anaphor, potential_antecedent):
    return anaphor['position'][0] - potential_antecedent['position'][0]

def distance(anaphor, potential_antecedent):
    a_sent, a_phrase = anaphor['position']
    b_sent, b_phrase = potential_antecedent['position']
    sentence_dist = a_sent - b_sent
    if sentence_dist == 0:
        phrase_dist = -(a_phrase - b_phrase)
    else:
        phrase_dist = -b_phrase
    return (sentence_dist, phrase_dist)

def features(anaphor, potential_antecedent):
    return {
        'REF': potential_antecedent['ID'],
        'value': potential_antecedent['value'],
        'string_match': string_match_p(anaphor, potential_antecedent),
        'all_words_in_antecedent': all_words_in_antecedent_p(anaphor, potential_antecedent),
        'it_referring_the': it_referring_the_p(anaphor, potential_antecedent),
        'stupid_prounoun_match': stupid_pronoun_match_p(anaphor, potential_antecedent),
        'word_match': any_word_matches_p(anaphor, potential_antecedent),
        'sentence_distance': sentence_distance(anaphor, potential_antecedent),
        'distance': distance(anaphor, potential_antecedent),
        'is_appositive' : is_appositive(potential_antecedent, anaphor),
        'edit_distance' : edit_distance(anaphor, potential_antecedent),
#        'pronoun' : pronoun_matcher(potential_antecedent, anaphor)
        }

def coreferent_pairs_features(corefs):
    refs = dict()
    for coref, potential_antecedents in each_with_tail(sorted(corefs, key=lambda a:a['position'], reverse=True)):
        if not coref['is_anaphor']:
            continue
        refs[coref['ID']] = [features(coref, potential_antecedent) for potential_antecedent in potential_antecedents]
    return refs

def feature_resolver(corefs):
    coref_dict = dict([(c['ID'], c) for c in corefs])
    features_of_coref_antecedents = coreferent_pairs_features(corefs)
    for coref_id, antecedents in features_of_coref_antecedents.items():
        potential_resolutions = []
        resolution = None
        for antecedent in antecedents:
            if antecedent['string_match']:
                potential_resolutions.append(((0, antecedent['distance']), antecedent, 'string_match'))
            if antecedent['all_words_in_antecedent']:
                potential_resolutions.append(((1, antecedent['distance']), antecedent, 'all_words'))
            if antecedent['word_match']:
                potential_resolutions.append(((2, antecedent['distance']), antecedent, 'word_match'))
            if antecedent['it_referring_the']:
                potential_resolutions.append(((3, antecedent['distance']), antecedent, 'it_referring_the'))
            if antecedent['stupid_prounoun_match']:
                potential_resolutions.append(((4, antecedent['distance']), antecedent, 'stupid_prounoun_match'))
            if antecedent['is_appositive']:
                potential_resolutions.append(((5, antecedent['distance']), antecedent, 'appositive'))

        if not resolution:
            if potential_resolutions:
                priority, resolution, method = min(potential_resolutions, key=lambda p: p[0])
                #print method
                #print "{0}@{1}: '{2[value]}' in '{2[sentence]}' matched {3[value]}".format(method, priority, coref_dict[coref_id], resolution)
            #if word_matches:
            #    resolution = min(word_matches, key=lambda f: f['distance'])
            else:
                if antecedents:
                    resolution = antecedents[0]
                    #if not 'the' in coref_dict[coref_id]['value'].lower():
                    #                    print "PUNT '{0[value]}'".format(coref_dict[coref_id])
                        #                    else:
                        #print "PUNT '{0[value]}' in {0[chunked_sentence]}".format(coref_dict[coref_id])

        if resolution:
            yield {'ID': coref_id, 'REF': resolution['REF']}

def update_refs(text, refs):
    """
    Given a list of Tagged Anaphora and antecedents, and an original input file, create a tagged output file.
    """
    new_text = text
    for ref in refs:
        new_text = new_text.replace('<COREF ID="{0[ID]}">'.format(ref), '<COREF ID="{0[ID]}" REF="{0[REF]}">'.format(ref))

    return new_text

def replace_coref_tags_with_tokens(text):
    return re.sub(coref_tag_re, r' coref_tag_beg_\1_ \2 coref_tag_end_\1_ ', text)

def replace_coref_tokens_with_tags(text):
    return re.sub(coref_token_re, r'<COREF ID="\1">\2</COREF>', text)

def no_break_zones(text):
    return [match.span() for match in re.finditer(coref_token_re, text)]

def coref_abbrs(text):
    return [word 
            for match in re.finditer(coref_token_re, text)
            for word in word_tokenize(match.groups()[1])
            if word[-1] == '.']

def read_text(file):
    return open(file, 'r').read()

def filename(file):
    return os.path.splitext(os.path.basename(file))[0]

def teach_abbreviations_to_tokenizer(abbrs):
    global sentence_tokenizer
    sentence_tokenizer._params.abbrev_types |= set(abbrs)

def get_text_from_files(files):
    to_resolve = []
    for file in files:
        name = filename(file)
        text = read_text(file)
        to_resolve.append((name, text))
    return to_resolve

class Gensym:
    i = 0
    def reset(self):
        self.i = 0

    def __call__(self):
        self.i += 1
        return 'X{0}'.format(self.i)

def resolve_files(files, response_dir_path):
    gensym = Gensym()
    def write_response(name, text):
        output_path = os.path.join(response_dir_path, name + '.response')
        open(output_path, 'w').write(text)

    def untagged_phrase(tagged_tokens):
        return ' '.join(word for word, tag in tagged_tokens)

    def tag_as_new_coref(noun_phrase):
        def surround_with_coref_tokens(tokens, ident):
            tokens.insert(0, ('coref_tag_beg_{0}_'.format(ident), 'CRB'))
            tokens.append(('coref_tag_end_{0}_'.format(ident), 'CRE'))
            
        def move_last_period_out_of_coref_tag(tokens):
            tag = tokens[-1]
            last_word = tokens[-2]
            if last_word[0][-1] == '.':
                tokens[-2] = (last_word[0][:-1], last_word[1])
                tokens[-1] = (tag[0] + '.', tag[1])

        surround_with_coref_tokens(noun_phrase, gensym())
        #move_last_period_out_of_coref_tag(noun_phrase)

    def is_anaphor(tokens):
        word, tag = tokens[0]
        return tag == 'CRB'

    def coref_from_noun_phrase(noun_phrase):
        def phrase_without_coref_tokens(noun_phrase):
            return noun_phrase[1:-1]

        def get_id(tokens):
            word, tag = tokens[0]
            return re.match(r'coref_tag_beg_(\w+)_', word).group(1)

        tagged_value = phrase_without_coref_tokens(noun_phrase)

        return {
            'ID': get_id(noun_phrase),
            'value': untagged_phrase(tagged_value),
            'tagged_value': tagged_value
            }
    
    def add_coref_data(coref, data):
        coref.update(data)

    documents_to_resolve = []
    for name, text in get_text_from_files(files):
        detagged_text = text
        detagged_text = re.sub(r'(?is)</?TXT>', '', detagged_text)
        detagged_text = replace_coref_tags_with_tokens(detagged_text)
        detagged_text = saxutils.unescape(detagged_text)
        documents_to_resolve.append((name, detagged_text))

    for _, text in documents_to_resolve:
        teach_abbreviations_to_tokenizer(coref_abbrs(text))
        
    for document_name, text in documents_to_resolve:
        gensym.reset()
        corefs = []
        np_tagged_sentences = []
        sentences = sentence_tokenize(text, no_break_zones(text))
        for i_sentence, sentence in enumerate(sentences):
            #            print sentence
            tokenized_sentence = word_tokenize(sentence)
            tagged_sentence = pos_tag(tokenized_sentence)
            #if 'we' in tokenized_sentence:
            #    print tagged_sentence
            chunked_sentence = chunk(tagged_sentence)

            noun_phrases = list(chunked_sentence.subtrees(filter=lambda s: s.node == 'NP'))
            for i_noun_phrase, noun_phrase in enumerate(noun_phrases):
                #                print noun_phrase
                was_an_anaphor = is_anaphor(noun_phrase)
                if not was_an_anaphor:
                    tag_as_new_coref(noun_phrase)
                    
                coref = coref_from_noun_phrase(noun_phrase)
                
                add_coref_data(coref, {
                    'is_anaphor': was_an_anaphor,
                    'position': (i_sentence, i_noun_phrase),
                    'phrases_in_sentence': len(noun_phrases),
                    'tokenized_sentence': tokenized_sentence,
                    'tagged_sentence': tagged_sentence,
                    'chunked_sentence': chunked_sentence,
                    'sentence': sentence
                })
                corefs.append(coref)
            np_tagged_sentences.append(untagged_phrase(chunked_sentence.leaves()))

        tagged_text = "\n".join(np_tagged_sentences)
        tagged_text = saxutils.escape(tagged_text)
        tagged_text = replace_coref_tokens_with_tags(tagged_text)
        tagged_text = '<TXT>' + tagged_text + '</TXT>\n'
        refs = feature_resolver(corefs)
        resolved_text = update_refs(tagged_text, refs)
        
        write_response(document_name, resolved_text)
        
#===============================================================================
# Main
#===============================================================================
def main():
    listfile_path = sys.argv[1]
    response_dir_path = sys.argv[2]

    files = [l.strip() for l in open(listfile_path, 'r').readlines()]

    resolve_files(files, response_dir_path)

if __name__ == '__main__':
    main()
    #cProfile.run('main()')



