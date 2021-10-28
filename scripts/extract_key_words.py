import pandas as pd
import os
import nltk
nltk.download()
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk import word_tokenize
from collections import Counter
from scipy.sparse import coo_matrix
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns


# Most frequently occuring words
def get_word_freq(corpus):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                       reverse=True)
    words_freq = pd.DataFrame(words_freq)
    words_freq.columns = ["Word", "Freq"]
    return words_freq


# Convert most freq words to dataframe for plotting bar plot
def plot_most_freq_words(dataframe, n):
    top_df = dataframe.head(n)
    #Barplot of most freq words
    sns.set(rc={'figure.figsize':(13,8)})
    g = sns.barplot(x="Word", y="Freq", data=top_df)
    g.set_xticklabels(g.get_xticklabels(), rotation=30)


# Function for sorting tf_idf in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def feature_score_tag(feature_names, sorted_items):
    score_vals = []
    feature_vals = []
    tags = []
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
        tags.append(pos_tag([feature_names[idx]])[0][1])
    results = pd.DataFrame(list(zip(feature_vals, score_vals, tags)),
               columns =['word', 'score', 'tag'])
    return results


def extract_topn_from_vector(feature_score_tag_results, tags, topn, threshold):
    # use only topn items from vector
    temp = feature_score_tag_results[feature_score_tag_results['tag'].isin(tags)][:topn].reset_index(drop=True)
    # only keep items pass the score threshold
    temp = temp[temp["score"] >= threshold]
    results = {}
    for idx in range(len(temp['word'])):
        results[temp['word'][idx]] = temp['score'][idx]
    return results


def process_by_num_words(corpus, stop_words, num_words, topn, threshold):
    # get feature names
    cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(num_words, num_words))
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    x = cv.fit_transform(corpus)
    tfidf_transformer.fit(x)
    # list(cv.vocabulary_.keys())[:10]
    feature_names = cv.get_feature_names()

    # extract key words
    id = []
    nn_keywords_keys = []
    nn_keywords_values = []
    vb_keywords_keys = []
    vb_keywords_values = []
    for idx in range(len(corpus)):
        # fetch document for which keywords needs to be extracted
        id.append(df['ID'][idx])
        doc = corpus[idx]
        # generate tf-idf for the given document
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        # sort the tf-idf vectors by descending order of scores
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        feature_score_tag_results = feature_score_tag(feature_names, sorted_items)
        # extract only the top n
        nn_keywords = extract_topn_from_vector(
            feature_score_tag_results, ['NN', 'NNP', 'NNS'], topn, threshold
        )
        nn_keywords_keys = list(nn_keywords.keys())
        nn_keywords_values = list(nn_keywords.values())
        vb_keywords = extract_topn_from_vector(
            feature_score_tag_results, ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'], topn, threshold
        )
        vb_keywords_keys = list(vb_keywords.keys())
        vb_keywords_values = list(vb_keywords.values())

        nn_keywords_keys.append(nn_keywords_keys)
        nn_keywords_values.append(nn_keywords_values)
        vb_keywords_keys.append(vb_keywords_keys)
        vb_keywords_values.append(vb_keywords_values)

    keyword_results = pd.DataFrame(
        list(zip(id, nn_keywords_keys, nn_keywords_values, vb_keywords_keys, vb_keywords_values)),
        columns=['ID', 'NN_keywords', 'NN_scores', 'VB_keywords', 'VB_scores'])
    return keyword_results


# root directory
root_directory = '/Users/melissapanggwugmail.com/Desktop/CIA_POC/Key_words_extraction'
# read in raw article data
input_file = 'raw_data.csv'
df = pd.read_csv(os.path.join(root_directory, input_file))
# Some exploration on the most frequent words, not neccessary in the pipeline
# Fetch wordcount for each abstract
df['word_count'] = df['Article'].apply(lambda x: len(str(x).split(" ")))
# Descriptive statistics of word counts
df.word_count.describe()
# Identify common words
# freq = pd.Series(' '.join(df['Article']).split()).value_counts().reset_index().rename({"index": "word", 0: "count"}, axis=1)

# Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
# Creating a list of custom stopwords
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]
stop_words = stop_words.union(new_words)

# clean and normalize the text data
corpus = []
for i in range(0, df.shape[0]):
    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', df['Article'][i])
    # Convert to lowercase
    text = text.lower()
    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # remove special characters and digits
    # shall we keep the digits?
    text = re.sub("(\\d|\\W)+", " ", text)
    # Convert to list from string
    text = text.split()
    # Stemming
    # ps = PorterStemmer()
    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text = " ".join(text)
    corpus.append(text)

# exploration: plot the most frequent n words
# words_freq = get_word_freq(corpus)
# plot_most_freq_words(words_freq, 20)


# extract keywords
key_single_word = process_by_num_words(corpus, stop_words, 1, 5, 0.1)
key_single_word = key_single_word.rename(
    {"NN_keywords": "NN_keywords_single_word",
     "NN_scores": "NN_scores_single_word",
     "VB_keywords": "VB_keywords_single_word",
     "VB_scores": "VB_scores_single_word"},
    axis=1
)
key_bi_gram = process_by_num_words(corpus, stop_words, 2, 5, 0.1)
key_bi_gram = key_bi_gram.rename(
    {"NN_keywords": "NN_keywords_bi_gram",
     "NN_scores": "NN_scores_bi_gram",
     "VB_keywords": "VB_keywords_bi_gram",
     "VB_scores": "VB_scores_bi_gram"}, axis=1
)
key_tri_gram = process_by_num_words(corpus, stop_words, 3, 5, 0.1)
key_tri_gram = key_tri_gram.rename(
    {"NN_keywords": "NN_keywords_tri_gram",
     "NN_scores": "NN_scores_tri_gram",
     "VB_keywords": "VB_keywords_tri_gram",
     "VB_scores": "VB_scores_tri_gram"}, axis=1
)
final_df = df.merge(key_single_word, left_on='ID', right_on='ID', how='left')
final_df = final_df.merge(key_bi_gram, left_on='ID', right_on='ID', how='left')
final_df = final_df.merge(key_tri_gram, left_on='ID', right_on='ID', how='left')

# output
output_filename = 'processed_input_data'
final_df.to_csv(os.path.join(root_directory, output_filename), index=False)