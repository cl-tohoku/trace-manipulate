from ast import operator
from urllib import parse
import re
from typing import Tuple
import uuid, random, jsonlines, logging, argparse
from datetime import datetime, date, timedelta
from dateutil import relativedelta
import ujson as json
from tqdm import tqdm
import numpy as np
import random
import math
from collections import defaultdict
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nltk.corpus import words, wordnet
import nltk
from pcfg import PCFG
import itertools
#nltk.download('words')
# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# words with <= 2 wordpieces
#nltk_words = [w.lower() for w in words.words() if len(bert_tokenizer.tokenize(w)) <= 2]

class Leaf(object):
    def __init__(self, number, depth):
        self.number = number
        self.depth = depth
    def __str__(self):
        return str(self.number)
class Node(object):
    def __init__(self, operator, children, depth):
        self.operator = operator
        self.children = children
        self.depth = depth
    def __str__(self):
        a, b = self.children
        if self.operator in ["min","max"]:
            return f"{self.operator}({a},{b})"
        return f"({a} {self.operator} {b})"

def generate(depth, pargs,max_depth =3,min_num=1,max_num = 100):
    
    if (not pargs.sub) and (not pargs.complete_binary_tree):
        if random.random() <= pargs.leaf_p:  # randomに終端を生成
            N = random.randrange(min_num, max_num)
            if pargs.float:
                N = rand_float(N)
            return Leaf(N, 1)  # depth 1

    if depth >= max_depth: #指定の深さまでいったら終端を生成
        N = random.randrange(min_num, max_num)
        if pargs.float:
            N = rand_float(N)
        return Leaf(N, 1)  # depth 1
    if pargs.short:
        N = random.randrange(min_num, max_num)
        children = [Leaf(N,1),
            generate(depth+1,pargs,max_depth=max_depth,min_num=min_num,max_num=max_num)]
    else:    
        children = [generate(depth+1,pargs,max_depth=max_depth,min_num=min_num,max_num=max_num),
            generate(depth+1,pargs,max_depth=max_depth,min_num=min_num,max_num=max_num)]
    
    depth = max(child.depth for child in children)
    operator = np.random.choice(['+', '-'],p=[0.5,0.5])
    if pargs.ast:
        operator = np.random.choice(['+', '-','*'],p=[0.45,0.45,0.1])
    if pargs.sub:
        operator="-"
    if pargs.mm:
        operator = random.choice(['max','min'])
    return Node(operator, children, depth + 1)
def replace_num_to_char(fomula:str):
    repl_char = chr(ord("a") - 1)

    def repl(matchobj):
        nonlocal repl_char
        repl_char = chr(ord(repl_char) + 1)
        return repl_char
        
    return re.sub("\d+", repl, fomula)

def rand_float(x):
    # randomly add upto 2 decimal places
    precision = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])
    #if pargs.align_digit:
        # precision = 0 #毎回小数点２桁を追加→桁数を揃えるタイプの実験のため
    fractional_part = {0: 0, 1: random.randint(0, 9)*0.1, 2: random.randint(0, 99)*0.01}[precision]
    return x + fractional_part
def sample(min_value, max_value) -> Tuple[int, ...]:

    min_digit = (
        0 if min_value == 0 else math.floor(
            math.log10(min_value))
    )
    max_digit = math.floor(math.log10(max_value))

    while True:
        digit = random.randrange(min_digit, max_digit)
        value = random.randrange(10 ** digit, 10 ** (digit + 1))
        if min_value <= value <= max_value:
            yield value
def formula_type(tree:Node)->str:
    tree_str = str(tree)
    tree_str = tree_str.replace(".","")
    return re.sub(r"\d+","x",tree_str)
def dfs(tree,intermed_values):
    if tree.depth!=1:
#         print(f"深さ{tree.depth}")
        intermed_values[tree.depth].append(round(eval(str(tree)),2))
        dfs(tree.children[0],intermed_values)
        dfs(tree.children[1],intermed_values)
        
    else:
        intermed_values[tree.depth].append(round(eval(str(tree)),2))
        return tree.number
def signed_expression_tree(pargs):
    while(1):
        tree = generate(0,pargs,max_depth = pargs.depth,min_num=pargs.num_limit_min, max_num = pargs.num_limit)
        if pargs.ood_train:
            if str(tree).count("-")<=pargs.minus_num:
                break
        elif pargs.ood_test:
            if str(tree).count("-")>pargs.minus_num:
                break
        else:
            break
    # calc_type = formula_type(tree)
    calc_type = replace_num_to_char(str(tree))
    intermed_values = defaultdict(list)
    dfs(tree,intermed_values)
    args = intermed_values[1]
    expr = str(tree).strip()
    if pargs.nooperation:
        expr = expr.replace("+","?").replace("-","?")
    return expr, round(eval(str(tree)),2),args,calc_type,intermed_values

def main():
    parser = argparse.ArgumentParser(description='For generating synthetic numeric data.')
    parser.add_argument("--num_samples", default=1e6, type=float, help="Total number of samples to generate.")
    parser.add_argument("--num_dev_samples", default=1e4, type=float, help="Num of samples to keep aside for dev set.")
    parser.add_argument("--num_limit", default=1000, type=int, 
                        help="max Num of args.")
    parser.add_argument("--num_limit_min", default=1, type=int, 
                        help="min Num of args.")
    # parser.add_argument("--align_digit",action="store_true",
    #                     help = "小数点(桁数)を固定")
    parser.add_argument("--focus_type",type=int,
                        default = -1,
                        help = "特定の計算タイプだけのデータを作る") #未実装
    parser.add_argument("--depth",type=int,
                        default = 1,
                        help = "計算木の深さを指定")  
    parser.add_argument("--float",action = "store_true",
                        help = "項に少数を使用")
    parser.add_argument("--sub",action="store_true",
                        help = "引き算データのみを作成")
    parser.add_argument("-cbt","--complete_binary_tree",action="store_true",
                        help="指定された深さの完全二分木のみからなるデータを作成")
    parser.add_argument("--leaf_p",
                        type = float,
                        default=0.2,
                        help ="数式の生成途中で葉が来る確率")
    parser.add_argument("--mm",
                        action="store_true",
                        help = "演算子をminとmaxにする")
    parser.add_argument("--short",
                        action="store_true",
                        help = "常に左側は葉を生成")
    parser.add_argument("--ood_train",
                        action="store_true",
                        help="マイナスがプラスの数以下のデータのみを生成")
    parser.add_argument("--ood_test",
                        action="store_true",
                        help="マイナスがプラスより多い外挿データのみを生成")
    parser.add_argument("--nooperation","-nop",
                        action="store_true",
                        help="演算子を全て?文字に置換(strong baseline)")
    parser.add_argument("--minus_num",
                        type=int,
                        default = 1,
                        help="o.o.d.テストを行う際の訓練データに含まれるマイナスの数")       
    parser.add_argument("--ast",
                        action="store_true",
                        help = "足し算引き算掛け算")
    pargs = parser.parse_args()
    assert not( pargs.complete_binary_tree and pargs.sub), "subとcbtのどちらもTrueになっています"
    # split the domain
    #domain, train_number_range, dev_number_range = int(2e4), [], []
    domain, train_number_range, dev_number_range = pargs.num_limit, [], []
    #範囲をpargs.num_limitでにする

    for i in range(pargs.num_limit_min,domain):
        x = train_number_range if random.random() < 0.8 else dev_number_range
        x.append(i)

    n_examples, n_dev, q_types = int(pargs.num_samples), int(pargs.num_dev_samples), 1
    discrete_ops_data, n_iters = [], n_examples // q_types
    train_args, dev_args = set(), set()
    sample_nums=defaultdict(int)
    calc_data_set = set()
    logger.info(f"Creating {n_examples} samples...")
    for i_s in tqdm(range(n_iters)):
        # decide train/dev split
        split = 'train' if i_s < n_iters - (n_dev // q_types) else 'dev'
        rng = {'train': train_number_range, 'dev': dev_number_range}[split]

        offset=0
        # with 50% prob add rand fraction
        # args = list(map(rand_float, args)) if random.randint(0,1) else args

        

        expr ,val ,args,oper_type, intermediate_values = signed_expression_tree(pargs)

        plus_num = expr.count("+")
        minus_num = expr.count("-")
        train_args.update(args) if split == 'train' else dev_args.update(args)
        calc_data_set.add(oper_type)
        d1 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 
            'type': oper_type, 'check_domain':True, 'split': split, 'intermed_values':intermediate_values}
        sample_nums[split]+=1
        discrete_ops_data += [d1]
    #print(f"train:{list(train_args)[0]},dev:{list(dev_args)[0]}")
    # assert train_args.isdisjoint(dev_args) # trn, dev args are disjoint
    logger.info(f"{len(calc_data_set)}タイプの計算データが生成されました")
    # print(calc_data_set)
    operator_type = "mm" if pargs.mm else "as"
    if pargs.ast:
        operator_type ="ast"
    # focus_type = f"_focus_{pargs.focus_type}" if pargs.focus_type!=-1 else ""
    args_type = "float" if pargs.float else "int"
    # samples = f"n_{int(pargs.num_samples)}_dev_{int(pargs.num_dev_samples)}"
    samples = f"n_{sample_nums['train']+sample_nums['dev']}_dev_{sample_nums['dev']}"
    leaf = f"_leaf{pargs.leaf_p}"
    if pargs.sub or pargs.complete_binary_tree:
        leaf=""
    sub = "_sub" if pargs.sub else ""
    cbt = "_cbt" if pargs.complete_binary_tree else ""
    short = "_short" if pargs.short else ""
    nop = "_no_operation" if pargs.nooperation else ""
    ood = ""
    if pargs.ood_test:
        ood=f"_ood_test_minus{pargs.minus_num}"
    elif pargs.ood_train:
        ood=f"_iid_minus{pargs.minus_num}"
    output_jsonl = f"./data/numeric_{samples}_min_{pargs.num_limit_min}_max_{pargs.num_limit}"\
        f"_depth_{pargs.depth}{leaf}_{operator_type}_{args_type}{sub}{cbt}{short}{ood}{nop}.jsonl"
    with jsonlines.open(output_jsonl, mode='w') as writer:
        writer.write_all(discrete_ops_data)
    print(f"{output_jsonl}に保存")
    #print(discrete_ops_data)

if __name__ == "__main__":
    main()
    
    
'''
python gen_tree_arithmetic_data.py --num_samples 1e4 --num_dev_samples 1e3 --num_limit 1000 --num_limit_min 1 --depth 2
'''
