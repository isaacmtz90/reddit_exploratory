import pandas as pd
import json
from pandas.io.json import json_normalize
import re
import nltk as nlp
from nltk.collocations import *
from nltk.corpus import stopwords # Import the stop word list



nlp.download('punkt')

blacklist =['travel','go','trip','good','went', 'like', 'get', 'day', 'days', 'got', 'also', 'would', 'time', 'www', 'reddit', 'com']
basename= 'traveldata.csv'
data =pd.read_csv(basename)


bigram_measures = nlp.collocations.BigramAssocMeasures()
trigram_measures = nlp.collocations.TrigramAssocMeasures()


data['comms_body'] = data['comms_body'].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))
data['comms_body'] = data['comms_body'].apply(lambda x: x.lower())
query_data = data[data['comms_body'].str.contains('vietnam')]

comments_list = query_data['comms_body'].tolist()
alltext = " ".join(comments_list)
tokenized = nlp.word_tokenize(alltext)
original =tokenized
tokenized = [w for w in tokenized if not w in stopwords.words("english")]
tokenized = [w for w in tokenized if not w in blacklist]
text = nlp.Text(tokenized)
text_original = nlp.Text(original)

fdlist = nlp.FreqDist(text)

finder = BigramCollocationFinder.from_words(text)
finder.apply_freq_filter(3)
print finder.nbest(bigram_measures.pmi, 5)
print text_original.concordance("food")
# print text.collocations(4)
print fdlist.most_common(50)
with open('viet.txt', 'a') as the_file:
     the_file.write(str(" ".join(text.tokens)))
