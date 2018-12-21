# coding=utf-8
import sys
import os
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import jieba
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def preprocess_wiki_data(wiki_file, output_file):
    with open(output_file, 'w') as f:
        wiki = WikiCorpus(wiki_file, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            f.write(' '.join(jieba.cut(''.join(text))) + '\n')


def word2vec(inp, model_path):
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=20)
    model.save(model_path)


if __name__ == '__main__':
    # wiki_file = 'zhwiki-latest-pages-articles.xml.bz2'
    wiki_file = 'zhwiki-latest-pages-articles4.xml-p2654618p2771086.bz2'
    output_file = 'wiki.zh.text'
    model_path = 'wiki.zh.text.model'
    preprocess_wiki_data(wiki_file, output_file)
    word2vec(output_file, model_path)
    model = Word2Vec.load(model_path)
    print(model.most_similar(u"青蛙"))