#!/usr/bin/env python3
import sys
import string
from enum import Enum

import nltk
from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.corpus import treebank
from nltk.tag import UnigramTagger
from nltk.tag.sequential import ClassifierBasedPOSTagger
from selenium import webdriver

from chunker import get_trained_classifier

# DATA STRUCTURES USED FOR NATURAL LANGUAGE PROCESSING:

ADDITIONAL_STOPS = ["it's", "i'm", "i'll", "i'd", "i've",
                    "it’s", "i’m", "i’ll", "i’d", "i’ve",
                    "he's", "he'd", "he'll",
                    "he’s", "he’d", "he’ll",
                    "she's", "she'd", "she'll",
                    "she’s", "she’d", "she’ll",
                    "you're", "you'd", "you've",
                    "you’re", "you’d", "you’ve"]

CHARS_TO_DELETE = '"'',.?:()[]<>~!–-•—“”“”…_'

# The Dict keys in POS_TAGS match those used for
# selecting/storing parts-of-speech in the functions below
POS_TAGS = {'nouns':['NP', 'NX', 'NN', 'NNS', 'NNP', 'NNPS'],
            'pronouns' : ['PRP', 'PRP$', 'WP', ' WP$'],
            'verbs' : ['VP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adverbs':['ADVP', ' WHADJP', ' WHAVP', 'RB', 'RBR', 'RBS', 'ADV',
                      '-BNF', '-DIR', '-EXT', '-LOC', '-MNR', '-PRP', '-TMP'],
            'adjectives' : ['ADJP', ' WHADJP', 'JJ', 'JJR', 'JJS'],
            'prepositions' : ['PP', 'WHPP', 'IN'],
            'untagged words' : [None],
            'miscellaneous words' : []} # The inclusion of 'misc words' is an
                                        # inelegant side-effect of an otherwise
                                        # elegant solution

class TokenisationMode(Enum):
    WORDS = 1
    CHUNKS = 2
    NGRAMS = 3

def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def int_input_prompt(message):
    value_set = False
    while not value_set:
        new_value = input(message)
        if is_int(new_value):
            return int(new_value)
        else:
            print('Please enter an integer.')

def string_from_text_file(path):
    try:
        with open(path, 'r') as textfile:
            return textfile.read()
    except FileNotFoundError:
        print('No file exists matching', path)

def string_to_text_file(path, string):
    with open(path, "w") as text_file:
        text_file.write(string)

def output_command_arguments(arg_descriptions):
    print('These are the recognised command arguments:')
    print(' {0:10}{1}'.format('NAME', 'DESCRIPTION'))
    for a in arg_descriptions.keys():
        print(' {0:10}{1}'.format(a, arg_descriptions[a]['description']))
        print('   Required arguments:')
        print(arg_descriptions[a]['required args'] + '\n')

def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
    return backoff

def get_chunks(text_string):
    # tokenization
    print('Tokenising text...')
    sentences = sent_tokenize(text_string)
    tokenized_sentences = []
    for s in sentences:
        tokenized_sentences.append(word_tokenize(s))
    # PoS tagging
    train_sents = treebank.tagged_sents()
    print('Training PoS tagger...')
    tagger = ClassifierBasedPOSTagger(train=train_sents)
    tagged_sentences = []
    print('Tagging sentences...')
    for s in tokenized_sentences:
        tagged_sentences.append(tagger.tag(s))
    # chunking
    print('Getting trained chunk classifier...')
    chunk_classifier = get_trained_classifier()
    chunked_sentences = []
    print('Chunking sentences...')
    for s in tagged_sentences:
        chunked_sentences.append(chunk_classifier.parse(s))
    return chunked_sentences

def get_words_simple(text_string):
    """
    Gets a list of tagged words from an input string
    using whitespace-based tokenisation and a unigram PoS tagger
    """
    # get trained Unigram tagger
    print('Loading unigram tagger...')
    train_sents = treebank.tagged_sents()
    unigram_tagger = UnigramTagger(train_sents)
    # stripping punctuation
    # string.translate() takes a dictionary as input.
    # The dictionary mapping ordinal chars to None is created in place:
    text_string = text_string.translate(
                  {ord(c): None for c in CHARS_TO_DELETE})
    words = text_string.split() # crude tokenisation, keeps contractions
    english_stops = stopwords.words('english')
    stops_set = set(english_stops + ADDITIONAL_STOPS)
    cleaned_words = []
    for w in words:
        if w not in stops_set and w not in string.punctuation:
            cleaned_words.append(w)
    return unigram_tagger.tag(cleaned_words)

def get_plain_chunks(chunked_sentences):
    plain_chunks = []
    for s in chunked_sentences:
        for chunk in s:
            words = []
            for token in chunk:
                words.append(token[0])
            if len(words) > 0:
                plain_chunks.append(' '.join(words))
    return plain_chunks
    
def get_ngrams(text_string, n):
    # stripping punctuation
    # string.translate() takes a dictionary as input.
    # The dictionary mapping ordinal chars to None is created in place:
    text_string = text_string.translate(
                  {ord(c): None for c in CHARS_TO_DELETE})
    words  = text_string.split()
    ngram_lists = ngrams(words, n)
    ngram_strings = []
    for n in ngram_lists:
        ngram_strings.append(' '.join(n))
    return ngram_strings
   
def choose_mode():
    text_string = string_from_text_file(sys.argv[2])
    print('\nHow do you want the text to be broken up for analysis?\n')
    chosen = False
    while not chosen:
        print('Enter W for single words.')
        print('Enter C for chunks.')
        print('Enter N for ngrams.')
        choice = input()
        if choice.lower() == 'w':
            words = get_words_simple(text_string.lower())
            frequency_analysis(TokenisationMode.WORDS, words)
        elif choice.lower() == 'c':
            chunked_sents = get_chunks(text_string)
            plain_chunks = get_plain_chunks(chunked_sents)
            frequency_analysis(TokenisationMode.CHUNKS, plain_chunks)
        elif choice.lower() == 'n':
            n = int_input_prompt('How many words should make up each ngram?\n')
            ngram_strings = get_ngrams(text_string, n)
            frequency_analysis(TokenisationMode.NGRAMS, ngram_strings)
        else:
            print('Input not recognised, please try again.')

def frequency_analysis(mode, tokens):
    """
    Performs simple frequency analysis with options for
    minimum word length, number of words and parts-of-speech to be included.
    """
    # Variables for word selection
    num_tokens = 100
    min_token_length = 3
    max_token_length = 16
    all_pos_tags_included = True
    pos_tags_included =     {'untagged words' : True,
                            'nouns' : True,
                            'pronouns' : True,
                            'verbs' : True,
                            'adverbs' : True,
                            'adjectives' : True,
                            'prepositions' : True,
                            'miscellaneous words' : True}

    committed = False
    while not committed:
        #Determining words to use in new FreqDist
        new_tokens = []
        working_tokens = []
        # Only choose words with POS tags
        # matching the classes in pos_tags_included:
        if not all_pos_tags_included:
            for w in tokens:
                word_included = False
                for tag in pos_tags_included:
                    if pos_tags_included[tag]:
                        for t in POS_TAGS[tag]:
                            if w[1] == t:
                                #print(w[0] + ' matches ' + str(w[1]))
                                working_tokens.append(w[0])
                                word_included = True
                if not word_included:
                    if pos_tags_included['miscellaneous words']:
                        working_tokens.append(w[0])
        else:
            working_tokens = tokens.copy()
        print(working_tokens)
        for w in working_tokens:
            if mode == TokenisationMode.CHUNKS:
                token = w
            elif mode == TokenisationMode.NGRAMS:
                token = w
            elif mode == TokenisationMode.WORDS:
                token = w[0]
            if  min_token_length <= len(token) <= max_token_length:
                new_tokens.append(token)

        fdist = FreqDist(new_tokens)

        prelim_results = ['\nFrequency analysis had found ']
        prelim_results.append(str(fdist.B()))
        prelim_results.append(' unique token of potential interest\n')
        prelim_results.append('out of a total of ')
        prelim_results.append(str(fdist.N()))
        prelim_results.append('.')
        print(''.join(prelim_results))
        
        intro = ['The ']
        intro.append(str(num_tokens))
        intro.append(' most frequent tokens are currently selected.\n')
        intro.append('Selected words are currently between ')
        intro.append(str(min_token_length))
        intro.append(' and ')
        intro.append(str(max_token_length))
        intro.append(' characters in length.\n')
        if not mode == TokenisationMode.NGRAMS:
            if all_pos_tags_included:
                intro.append('All parts-of-speech are included in the selection.\n')
            else:
                intro.append('Parts-of-speech included in the selection:\n')
                for tag in pos_tags_included:
                    if pos_tags_included[tag]:
                        intro.append(tag)
                        intro.append(', ')
                intro[len(intro) - 1] = '\n'# replace trailing comma with linebreak
                intro.append('Parts-of-speech excluded from the selection:\n')
                for tag in pos_tags_included:
                    if not pos_tags_included[tag]:
                        intro.append(tag)
                        intro.append(', ')
                intro[len(intro) - 1] = '\n'# replace trailing comma with linebreak

        intro.append('Below are the selected words, most frequent first:\n')
        print(''.join(intro))
        selected_tokens = fdist.most_common(num_tokens)
        word_string = []
        charcount = 0
        print(selected_tokens)
        for w in selected_tokens:            
            if mode == TokenisationMode.CHUNKS:
                token = w[0]
            elif mode == TokenisationMode.NGRAMS:
                token = w[0]
            elif mode == TokenisationMode.WORDS:
                token = w[0]
            charcount += len(token) + 2
            if charcount > 80:
                word_string.append('\n')
                charcount = 0
            word_string.append(token)
            word_string.append(', ')
        word_string[len(word_string) - 1] = '\n' # strip comma, add linebreak
        print(''.join(word_string))
        chosen = False
        while not chosen:
            print('Enter M below to change the minimum token length.')
            print('Enter X below to change the maximum token length.')
            print('Enter N to change the total number of tokens selected.')
            if not mode == TokenisationMode.NGRAMS:
                print('Enter P to restrict selection with PoS tagging.')
            print('Enter A to accept the current list of tokens and continue.')
            user_input = input()
            if user_input.lower()  == 'n':
                num_tokens = int_input_prompt(
                                  '\nHow many words do you want selected?\n')
                chosen = True
            elif user_input.lower()  == 'm':
                validated = False
                while not validated:
                    min_token_length = int_input_prompt(
                                    '\nEnter a new minimum word length...\n')
                    if min_token_length > max_token_length:
                        print("Minimum word length can't exceed maximum!")
                    else:
                        validated = True
                chosen = True
            elif user_input.lower()  == 'x':
                validated = False
                while not validated:
                    max_token_length = int_input_prompt(
                                    '\nEnter a new maximum word length...\n')
                    if mamin_token_length > max_token_length:
                        print("Maximum word length cannot be less than"
                                                        +   "minimum!")
                    else:
                        validated = True
                chosen = True
            elif user_input.lower()  == 'p':
                if not mode == TokenisationMode.NGRAMS:
                    nothing_selected = True
                    while nothing_selected:
                        for tag in pos_tags_included:
                            print('Do you want to include ' + tag + '?')
                            pos_tag_chosen = yes_no_input_prompt()
                            pos_tags_included[tag] = pos_tag_chosen
                            if pos_tag_chosen:
                                chosen = True
                                nothing_selected = False
                            else:
                                all_pos_tags_included = False
                        if nothing_selected:
                            print('Error: you must include at least ' +
                                        'one class of POS tags.')
                            print('Restarting selection...\n')
                else:
                    print('PoS tagging not applicable to ngrams.')
                    print('Doing nothing...')
            elif user_input.lower() == 'a' :
                chosen = True
                committed = True
            else:
                print('Input not recognised')

    words_string =  ', '.join([w[0][0] for w in selected_tokens])
    string_to_text_file(sys.argv[3], words_string)
    print('Words/phrases sucessfully saved to ' + sys.argv[3])
    quit()

# Dictionary of dictionaries defining command arguments accepted by snu-snu
ARGS = {'freq':
    {'description':'carries out various forms of frequency analysis on texts.',
    'required arg count' : 4,
    'required args' :
        '   1. The command (i.e. "freq") 2. path to source file '
        + '(e.g. "in.txt")\n   3. path to destination file (eg. "out.txt")',
    'function' : choose_mode}}

def initialise():
    """
    Checks arguments and calls appropriate functions.
    """
    print("""This is text-process: a utility for carring out NLP on texts.\n""")

    # Ensures that NLTK data can be stored in the same directory as code
    nltk.data.path.append('./nltk_data/')

    proceed_with_args = False
    if len(sys.argv) > 1:
        includes_recognised_arg = False
        for recognised_arg in ARGS.keys():
            if sys.argv[1] == recognised_arg:
                includes_recognised_arg = True
        if includes_recognised_arg:
            print('You ran text-process with the command argument: '
                                                        + sys.argv[1])
            print('This ' + ARGS[sys.argv[1]]['description'])
            if len(sys.argv) == ARGS[sys.argv[1]]['required arg count']:
                proceed_with_args = True
            else:
                error = ['Error: this command will only work with a total of ']
                error.append(str(ARGS[sys.argv[1]]['required arg count'] - 1))
                error.append(' arguments.')
                print(''.join(error))
                print('See "' + sys.argv[1] + '" in the below list...\n')
                output_command_arguments(ARGS)
        else:
            print('Command argument "' + sys.argv[1] + '" not recognised\n')
            output_command_arguments(ARGS)
    else:
        print('Error: text-process requires terminal arguments to run.\n')
        output_command_arguments(ARGS)
    if proceed_with_args:
        ARGS[sys.argv[1]]['function']()
    else:
        print('Quitting...')
        quit()

initialise()


