from datetime import datetime
import json
import multiprocessing
import pathlib
import re
import time

import dask.dataframe as dd
from gensim.models import word2vec
import MeCab
import numpy as np
import pandas as pd
from tqdm import tqdm

import api
import preprocessor
import utils

tqdm.pandas()


def extract_product_name(x: str) -> list:
    x = re.sub(r'\【.+?\】', '', x)
    x = re.sub(r'\「.+?\」', '', x)
    x = preprocessor.clean_number(x)
    prod_name = x.split('◆')[0]
    result = []
    for w in utils.NounTokenizer.tokenize(prod_name):
        if w not in ['今治', '産', 'ディズニー']:
            result.append(w)
    return result


def extract_text(x: str) -> str:
    x = preprocessor.clean_tag(x)
    x = preprocessor.clean_number(x)
    x = preprocessor.clean_symbol(x)
    x = preprocessor.clean_space(x)
    return x


def extract_suggests(x: list) -> list:
    result = []
    for i in x:
        result += api.fetch_suggest_words(i)
        time.sleep(1)
    return list(set(result))


def get_unique_names(ser: pd.Series) -> list:
    names = []
    for i in range(len(ser)):
        for name in np.unique(ser.iloc[i]):
            if name not in names:
                names.append(name)
    return names


class SuggestsExtractor:
    """
    Parameters
    ----------
    path: str
        The path to json.
        ex) {'タオル': ['suggest1', ...], '素麺': ['suggest1', ...]}
    """

    def __init__(self, path: str=None) -> None:
        self.word_to_suggests = {}
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                self.word_to_suggests = json.load(f)

    def extract(self, x: list) -> list:
        """
        Parameters
        ----------
        x: list
            The list of product names.
            ex) ['素麺', '彩', '大量']
        
        Returns
        -------
        result: list
            The list of unique suggest words.
        """
        result = []
        for w in x:
            if w in self.word_to_suggests:
                result += self.word_to_suggests[w]
            else:
                suggests = api.fetch_suggest_words(w)
                time.sleep(1.0)
                result += suggests
                self.word_to_suggests[w] = suggests
        return list(set(result))


def similars_in_text(row: pd.Series) -> list:
    text = row['text']
    similar_dic = row['similar']
    if not similar_dic:
        return []
    try:
        if np.isnan(text):
            return []
    except:
        pass

    res = []
    for w in similar_dic.keys():
        if w in text:
            res.append(w)
    return res


if __name__ == '__main__':
    df = pd.read_csv('./inputs/dl-item201907041607-1.csv', encoding='shift-jis')

    df['prod_name'] = df['商品名'].apply(extract_product_name)
    df['text'] = df['PC用商品説明文'].astype(str).apply(extract_text)

    drop_cols = ['コントロールカラム', '商品管理番号（商品URL）', 'PC用商品説明文',
                     'スマートフォン用商品説明文', 'PC用販売説明文']
    df.drop(drop_cols, axis=1, inplace=True)

    df['prod_name'] = df['prod_name'].apply(preprocessor.clean_stopwords)
    uniques = get_unique_names(df['prod_name'])

    json_path = pathlib.PurePath('./dicts/name_to_suggests.json')
    extractor = SuggestsExtractor(json_path)
    print('Fetching suggest words.')
    df['suggest'] = df['prod_name'].progress_apply(extractor.extract)

    json_path = pathlib.PurePath(f'./dicts/name_to_suggests_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    utils.save_json(json_path, extractor.word_to_suggests)

    df = utils.expand_list_in_df(df[['商品名', 'text', 'suggest', 'prod_name']], 'suggest')

    model = utils.Word2Vec(path='./word2vec/wiki_mecab-ipadic-neologd.model')
    ddf = dd.from_pandas(df, npartitions=multiprocessing.cpu_count()-1)
    ddf['similar'] = ddf['suggest'].apply(model.predict, meta=('suggest', 'str'))
    df = ddf.compute()

    # meta = df.head(1).apply(similars_in_text, axis=1)
    df['similar_words_in_text'] = df.apply(similars_in_text, axis=1)

    csv_path = pathlib.PurePath(f'./outputs/nouhin_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(csv_path, index=False)