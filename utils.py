'''
Adapted from MACS40400 course material - week4, week6
help functions in nlp.py, music.py
Created by Prof Jon Clindaniel
'''

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import string
from gensim import corpora, models
import demoji
from gensim.utils import effective_n_jobs


# Define stop words + punctuation
STOP = set(nltk.corpus.stopwords.words('english')
       + list(string.punctuation)
       )

# Define words with the following focus
WEIGHT = ['weight loss', 'pound', 'lb', 'kg', 'size']        # weight monitering
DIET = ['diet', 'eat', 'fast', 'cal', 'starve', 'nutrition', 'meal', 'food', 
        'flavo', 'protein', 'nutri', 'crave', 'meal']        # dietary restraints
EXERCISE = ['exercise', 'injur', 'pain', 'lift', 'workout', 'cardio', 'train', 
            'squat', 'burpee', 'jog', 'gym']        # physical exercise
APPEARANCE = ['sexy', 'sexi', 'beauty', 'hot', 'look', 'shape', 'bodycheck', 
              'waist', 'thigh', 'tummy', 'booty']     # appearance focus
ED = ['anor', 'bulim', 'edtwt', 'binge', 'ed', 'ana', 
      'purge', 'vomit']     # eating disorder
HEALTH = ['health', 'mindful']      # health maintenance
PROMO = ['product', 'subscribe', 'shipping', 'coupon', 'promo', 'ads', 'free', 
         'bio', 'link', 'discount', 'bonus', 'offer', 'code']     # promotions

demoji.download_codes()
    
def remove_em(text):
    '''
    removes all emojis from a string
    '''

    dem = demoji.findall(text)
    for item in dem.keys():
        text = text.replace(item, '')
    return text


def clean_tweet_texts(df):
    '''
    Removes the strings '@<username>', '#<hashtag>' and 'https:<link>' 
    from textual data, seperates all hashtags from the texts,
    and removes duplicates of captions (keeping the first one)

    Input: pandas dataframe containing data scraped from Twitter
    Output: pandas dataframe with clean textual data
    '''

    text_clean = []     # list of clean texts
    hashtags = []       # list of hashtags

    for txt in df['text']:
        ht = []
        clean_words = []

        txt = remove_em(txt)
        txt_lst = txt.replace('#', ' #').strip().split()

        for word in txt_lst:
            if '#' in word:
               ht.append(word)
            elif '@' not in word and 'http' not in word:
                clean_words.append(word)

        hashtags.append(' '.join(ht))
        text_clean.append(' '.join(clean_words))
    
    df['hashtags'] = hashtags
    df['text_clean'] = text_clean

    '''
    # remove repetitvie texts
    df = df.drop_duplicates(subset=["text_clean"], keep='first') \
           .sort_values(by=["created_at"]) \
           .reset_index(drop=True)
    '''

    print("The sample contains {} posts from {} to {}."
        .format(df.shape[0], 
                df.iloc[0]["created_at"], 
                df.iloc[-1]["created_at"]))         

    return df


def tag_text_focus(text, word_lst):
    '''
    Tags a text 1 if item in word_lst appears in text, 0 otherwise
    '''
    
    for target in word_lst:
        if target in text:
            return 1
        else:
            for word in text.split():
                if target in word:
                    return 1
    return 0


def pos_tag(text):
    '''
    Tags each word in a string with its part-of-speech indicator,
    excluding stop-words and words <= 3 characters
    '''
    # Tokenize words using nltk.word_tokenize, keeping only those tokens that do
    # not appear in the stop words we defined
    tokens = [i for i in nltk.word_tokenize(text.lower())
                 if (i not in STOP) and (len(i) > 3)]

    # Label parts of speech automatically using NLTK
    pos_tagged = nltk.pos_tag(tokens)
    return pos_tagged


def pos_tagged_full(series):
    '''
    Returns a list of tuples of word and tag.
    '''
    # Apply part of Speech tagger that we wrote above to any Pandas series that
    # pass into the function
    pos_tagged = series.apply(pos_tag)

    pos_tagged_full = []
    
    for i in pos_tagged:
        pos_tagged_full.extend(i)
    return pos_tagged_full


def plot_top_words(series, data_description, n = 10):
    '''
    Plots the top `n` words in a Pandas series of strings.
    '''

    # Create Frequency Distribution of diff words and plot distribution
    fd = nltk.FreqDist(word + "/" + tag for (word, tag) in pos_tagged_full(series))
    fd.plot(n, title='Top {} Words for '.format(n) + data_description);
    return

def plot_top_adj(series, data_description, n = 10):
    '''
    Plots the top `n` adjectives in a Pandas series of strings.
    '''

    # Create Frequency Distribution of diff adjectives and plot distribution
    fd = nltk.FreqDist(word + "/" + tag for (word, tag) in pos_tagged_full(series)
                           if tag[:2] == 'JJ')
    fd.plot(n, title='Top {} Adjectives for '.format(n) + data_description);
    return


def plot_top_n(series, data_description, n = 10):
    '''
    Plots the top `n` nouns in a Pandas series of strings.
    '''
    
    # Create Frequency Distribution of diff nouns and plot distribution
    fd = nltk.FreqDist(word + "/" + tag for (word, tag) in pos_tagged_full(series)
                           if (tag[:2] == 'NN') or (tag[:3] == 'NNS'))
    fd.plot(n, title='Top {} Nouns for '.format(n) + data_description);
    return


def plot_top_vb(series, data_description, n = 10):
    '''
    Plots the top `n` verbs in a Pandas series of strings.
    '''

    # Create Frequency Distribution of diff adjectives and plot distribution
    fd = nltk.FreqDist(word + "/" + tag for (word, tag) in pos_tagged_full(series)
                           if tag[:2] == 'VB')
    fd.plot(n, title='Top {} Verbs for '.format(n) + data_description);
    return


def get_wordnet_pos(word):
    '''
    Tags each word with its Part-of-speech indicator -- specifically used for
    lemmatization in the get_lemmas function
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


def plot_top_tfidf(series, data_description, n=20):
    '''
    Plots the top `n` lemmas (in terms of average TFIDF)
    across a Pandas series (corpus) of strings (documents).
    '''
    # Get lemmas for each row in the input Series
    lemmas = series.apply(get_lemmas)

    # Initialize Series of lemmas as Gensim Dictionary for further processing
    dictionary = corpora.Dictionary(lemmas)

    # Convert dictionary into bag of words format: list of
    # (token_id, token_count) tuples
    bow_corpus = [dictionary.doc2bow(text) for text in lemmas]

    # Calculate TFIDF for each word in a document,
    # and compute total TFIDF sum across all documents:
    tfidf = models.TfidfModel(bow_corpus, normalize=True)
    tfidf_weights = {}
    for doc in tfidf[bow_corpus]:
        for ID, freq in doc:
            tfidf_weights[dictionary[ID]] = np.around(freq, decimals=2) \
                                          + tfidf_weights.get(dictionary[ID], 0)

    # highest (average) TF-IDF:
    top_n = pd.Series(tfidf_weights).nlargest(n) / len(lemmas)

    # Plot the top n weighted words:
    plt.plot(top_n.index, top_n.values)
    plt.xticks(rotation='vertical')
    plt.title('Top {} Lemmas (TFIDF) for {}'.format(n, data_description));


def hashtag_pooling_lda(df_input):
    '''
    Pooling the tweets based on hashtags
    Input: 
        dataframe with the columns "hashtags" and "text_clean"
    Output:
        dataframe for lda with the column "tweets_concatenated" 
        indexed with each hashtag
    '''

    # initiates a dataframe
    df = pd.DataFrame(columns=["hashtag", "tweets_concatenated"])

    # maps the fitspiration hashtags to the number of tweets with the hashtag
    fit_ht_dict = {}
    for h_lst in df_input["hashtags"].str.split():
        for h in h_lst:
            fit_ht_dict[h] = fit_ht_dict.get(h, 0) + 1

    # add rows in df with index being individual hashtags
    for ht, count in fit_ht_dict.items():
        if count >= 5:
            new_row = [ht, df_input.loc[df_input["hashtags"] > ht] \
                                                    ["text_clean"].str.cat()]
            df.loc[len(df.index)] = new_row
   
    return df


def get_lemmas(text):
    '''
    Gets lemmas for a string input, excluding stop words, punctuation, as well
    as a set of study-specific stop-words

    Only returns lemmas that are greater 3 characters long.
    '''
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t))
              for t in nltk.word_tokenize(text.lower()) if t not in STOP]
    return [l for l in lemmas if len(l) > 3]


def make_bigrams(lemmas):
    '''
    Make bigrams for words within a given document
    '''
    bigram = models.Phrases(lemmas, min_count=5)
    bigram_mod = bigram.freeze()
    return [bigram_mod[doc] for doc in lemmas]


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    '''
    Computes Coherence values for LDA models with differing numbers of topics.

    Returns list of models along with their respective coherence values (pick
    models with the highest coherence)
    '''
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.ldamulticore.LdaMulticore(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=num_topics,
                                                 workers=effective_n_jobs(-1))
        model_list.append(model)
        coherence_model = models.coherencemodel.CoherenceModel(model=model,
                                                          corpus=corpus,
                                                          dictionary=dictionary,
                                                          coherence='u_mass')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values


def top_tweets_by_topic(filepath, df, ldamodel, corpus, ntop=1):
    '''
    Finds the top "n" tweets by topic, which we can use for
    understanding the types of tweets included in a topic.
    '''
    topn_texts_by_topic = {}
        
    with open(filepath, 'w') as f:
        
        for i in range(len(ldamodel.print_topics())):
            # For each topic, collect the most representative tweet(s)
            # (i.e. highest probability containing words belonging to topic):
            top = sorted(zip(range(len(corpus)), ldamodel[corpus]),
                        reverse=True,
                        key=lambda x: abs(dict(x[1]).get(i, 0.0)))
            topn_texts_by_topic[i] = [j[0] for j in top[:ntop]]

            # Print out the topn tweets for each topic and return their indices as a
            # dictionary for further analysis:
            f.write("Topic " + str(i))
            f.write(df.loc[topn_texts_by_topic[i]].to_string())
            f.write("\n")
            f.write("*******************************")
            f.write("\n")

    return topn_texts_by_topic