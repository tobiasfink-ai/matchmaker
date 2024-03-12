import numpy as np
from matplotlib import pyplot
from tqdm import tqdm

#bins = [i/100 for i in range(30,101, 5)]
f_bert= "/data/tfink/kodicare/trec-2019-dl/doc_ret/triplets_bert.train.id"
f_tfidf= "/data/tfink/kodicare/trec-2019-dl/doc_ret/triplets_tfidf.train.id"

x_bert = []
x_tfidf = []
with open(f_bert, "r") as fp:
    for line in tqdm(fp):
        q_id, positive, negative, score = line.split()
        score = float(score)
        x_bert.append(score)


with open(f_tfidf, "r") as (fp):
    for line in tqdm(fp):
        q_id, positive, negative, score = line.split()
        score = float(score)
        x_tfidf.append(score)

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec='k', range=(0.4,1.0))

#pyplot.hist(x_tfidf, bins, label='TF-IDF', **kwargs)
#pyplot.hist(x_bert, bins, label='SentenceBERT', **kwargs)
pyplot.hist(x_tfidf, bins=40, label='TF-IDF', **kwargs)
pyplot.hist(x_bert, bins=40, label='SentenceBERT', **kwargs)
pyplot.legend(loc='upper right')
pyplot.savefig('kd_histogram.png')