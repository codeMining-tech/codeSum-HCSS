from torch.utils.data import Dataset
import utils
from utils import load
from graph import pre_traverse, get_graph_seqs, parse_single, extractAPI


class CodePtrDataset(Dataset):

    def __init__(self, code_path, tree_path,type_path, nl_path):
        # get lines
        codes = utils.load_dataset(code_path)  #
        trees = utils.load_dataset(tree_path)  #
        types = utils.load_dataset(type_path)  #
        nls = utils.load_dataset(nl_path)

        if len(codes) != len(nls) or len(types) != len(codes) or len(codes)!=len(trees):
            raise Exception('The lengths of three dataset do not match.')

        self.codes, self.trees, self.types, self.nls = utils.filter_data(codes, trees, types, nls)  # 最大长度限制

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.codes[index], self.trees[index], self.types[index], self.nls[index]

    def get_dataset(self):
        return self.codes, self.trees, self.types, self.nls


# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:51:52 2019

@author: Zhou
"""

from javalang import tokenizer
from torchtext.legacy.data import Field

from strudata import node_filter, get_ast, convert_tree2lstm_input, get_tree_dict
from utils import save
from math import ceil
from itertools import chain
import pyastyle as style
import re
import random
import torch
#from ApiSlice import findStatementwithAPI, getFuncName

def getFuncName(code):# 第一次匹配
    """
    :param code: code snippet;
    :return: function name in input code snippet
    """
    obj = re.search(r'\s?(\w*?)\s?\(',code)
    return obj.group(1)


def findStatementwithAPI(code,apistr,dropped,idx):
    result = []
    for api in apistr:

        item = re.search('\n(.*?'+api+'.*?)\n', code)
        if item:
            result.append(item.group().strip())
            code.replace(api,' ')
            #codelines.remove(line)
    if 0 == len(result):
        #result.extend([code])
        dropped.add(idx)
        return 0
        # odd sample
    return result
# from torchtext.legacy.data import Field  # 版本问题！

from tqdm import tqdm

subtoken = True
node_maxlen = 15
text_minlen = 1
text_maxlen = 30
code_maxlen = 200
# max_statm = 60
max_statm = 40
# statm_maxlen = 100
statm_maxlen = 60  #
code_vocab_size = 35000
nl_vocab_size = 25000
# batch_size = 64
batch_size = 32
# special tokens
bos = '<s>'
eos = '</s>'
pad = '<pad>'
unk = '<unk>'

_NUM = {tokenizer.Integer,
        tokenizer.BinaryInteger,
        tokenizer.DecimalInteger,
        tokenizer.DecimalFloatingPoint,
        tokenizer.FloatingPoint,
        tokenizer.HexFloatingPoint,
        tokenizer.HexInteger,
        tokenizer.OctalInteger}

_LITERAL = {tokenizer.Character,
            tokenizer.Literal,
            tokenizer.String,
            tokenizer.Identifier}

def code_filter(code, subtoken=subtoken, max_tokens=code_maxlen):
    tokens = list(tokenizer.tokenize(code))
    result = []
    for token in tokens:
        if type(token) in _NUM:
            result.append('num')
        elif type(token) in _LITERAL:
            value = node_filter(token.value, subtoken)
            result.extend(value)
        else:
            result.append(token.value)
    return result[:max_tokens]

import config
def tokenize_code(codes, subtoken=subtoken, max_tokens=config.max_code_length, save_path=None):
    results = []
    for code in tqdm(codes, desc='Tokenizing codes...'):
        code = code.strip()
        result = code_filter(code, subtoken, max_tokens)
        results.append(result)

    if save_path is not None:
        save(results, save_path, is_json=True)
    return results

def get_ACG_nodes(codes,api_seqs,matrix_results=[]):
    results = []
    results_type = []

    for code, api_seq in zip(codes, api_seqs):
        result, types, matrix = get_graph_seqs(code, api_seq)
        results.append(result)
        results_type.append(types)
        matrix_results.append(matrix)
    return results

def split_code(codes, subtoken=subtoken, max_statm=max_statm, statm_maxlen=statm_maxlen,
               save_path=None):
    results = []
    for code in tqdm(codes, desc='Splitting codes into statements & tokenizing... '):
        code += '\n'  # ensure that the formatter will work
        code = style.format(code, '--style=allman --delete-empty-lines --add-brackets')  # 这里有格式化代码欸
        statements = code.strip().split('\n')

        result = []
        for statement in statements:
            statement = statement.strip()
            if statement == 'else {':  # a bug of old AStyle
                result.extend([['else'], ['{']])
            else:
                tokens = code_filter(statement, subtoken, statm_maxlen)
                result.append(tokens)
        results.append(result[:max_statm])
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results


def split_codeby_API(codes, api_seqs, dropped=set(), subtoken=subtoken, max_statm=max_statm, statm_maxlen=statm_maxlen,
                     save_path=None):
    results = []
    for idx, (code, api_seq) in tqdm(enumerate(zip(codes, api_seqs)),
                                     desc='Splitting codes into API based statements & tokenizing... '):
        code += '\n'  # ensure that the formatter will work
        code = style.format(code, '--style=allman --delete-empty-lines --add-brackets')
        statements = findStatementwithAPI(code, api_seq, dropped, idx)
        result = []
        if 0 != statements:
            for statement in statements:
                statement = statement.strip()
                if statement == 'else {':  # a bug of old AStyle
                    result.extend([['else'], ['{']])
                else:
                    try:
                        tokens = code_filter(statement + ';', subtoken, statm_maxlen)
                    except:
                        tokens = nl_filter(statement, min_tokens=0)
                    result.append(tokens)
        results.append(result[:max_statm])
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results

def nl_filter(s, subtoken=subtoken, min_tokens=text_minlen, max_tokens=text_maxlen):
    s = re.sub(r"\([^\)]*\)|(([eE]\.[gG])|([iI]\.[eE]))\..+|<\S[^>]*>", " ", s)
    #    brackets; html labels; e.g.; i.e.
    s = re.sub(r"\d+\.\d+\S*|0[box]\w*|\b\d+[lLfF]\b", " num ", s)
    first_p = re.search(r"[\.\?\!]+(\s|$)", s)
    if first_p is not None:
        s = s[:first_p.start()]
    s = re.sub(r"https:\S*|http:\S*|www\.\S*", " url ", s)
    s = re.sub(r"\b(todo|TODO)\b.*|[^A-Za-z0-9\.,\s]|\.{2,}", " ", s)
    s = re.sub(r"\b\d+\b", " num ", s)
    s = re.sub(r"([\.,]\s*)+", lambda x: " " + x.group()[0] + " ", s)

    if subtoken:
        s = re.sub(r"[a-z][A-Z]", lambda x: x.group()[0] + " " + x.group()[1], s)
        s = re.sub(r"[A-Z]{2}[a-z]", lambda x: x.group()[0] + " " + x.group()[1:], s)
        s = re.sub(r"\w{32,}", " ", s)  # MD5
        s = re.sub(r"[A-Za-z]\d+", lambda x: x.group()[0] + " ", s)
    s = re.sub(r"\s(num\s+){2,}", " num ", s)
    s = s.lower().split()
    return 0 if len(s) < min_tokens else s[:max_tokens]


def tokenize_nl(texts, subtoken=subtoken, min_tokens=text_minlen,
                max_tokens=text_maxlen, save_path=None):
    drop_list = set()
    results = []
    for idx, text in enumerate(tqdm(texts, desc='Tokenizing texts...')):
        tok_text = nl_filter(text, subtoken, min_tokens, max_tokens)
        if tok_text:
            results.append(tok_text)
        else:
            results.append([])
            drop_list.add(idx)

    if save_path is not None:
        save(results, save_path, is_json=True)
    print('number of dropped texts:', len(drop_list))
    return results, drop_list


def get_txt_vocab(texts, max_vocab_size=nl_vocab_size, is_tgt=True, save_path=None):
    if is_tgt:
        field = Field(init_token=bos, eos_token=eos, batch_first=True, pad_token=pad, unk_token=unk)
    else:
        field = Field(batch_first=True, pad_token=pad, unk_token=unk)
    field.build_vocab(texts, max_size=max_vocab_size)
    if save_path:
        print('Saving...')
        torch.save(field, save_path)
    return field


def id2word(word_ids, field, source=None, remove_eos=True, remove_unk=False, replace_unk=False):
    if replace_unk:
        assert type(source) is tuple and not remove_unk
        raw_src, alignments = source
    eos_id = field.vocab.stoi[eos]
    unk_id = field.vocab.stoi[unk]

    if remove_eos:
        word_ids = [s[:-1] if s[-1] == eos_id else s for s in word_ids]
    if remove_unk:
        word_ids = [filter(lambda x: x != unk_id, s) for s in word_ids]
    if not replace_unk:
        return [[field.vocab.itos[w] for w in s] for s in word_ids]
    else:
        return [[field.vocab.itos[w] if w != unk_id else rs[a[i].argmax()] \
                 for i, w in enumerate(s)] for s, rs, a in zip(word_ids, raw_src, alignments)]


def singlesample(code):
    api_seqs =extractAPI(parse_single(code))
    codeTokens = tokenize_code([code])
    matrix_results=[]
    ACGNodes = get_ACG_nodes([code],[api_seqs],matrix_results)
    ACGMatrixs = matrix_results
    ASTMatrixs = get_ast([code],save_path='data/' +'preprocess_filterapi' + '/nodes.json')
    ASTNodes = load('data/' + 'preprocess_filterapi' + '/' + 'nodes' + '.json', is_json=True, drop_list=[])
    return codeTokens[0],ACGNodes[0],ACGMatrixs,ASTNodes[0],ASTMatrixs[0]
    # AST_matrixs

