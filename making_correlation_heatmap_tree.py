import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertModel,BertConfig
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

from tqdm import tqdm
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib
import re
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import os
import pandas as pd
# from probing_intermediate_value import create_dataset_for_visualize_and_probe_all_layer
from utils import create_dataset_for_visualize_and_probe_all_layer,dimention_reduction,compile_med_val,decompose_intermed_results
import seaborn as sns
import pickle

# import modeling
#from create_examples_n_features import DropExample, DropFeatures, read_file, write_file, split_digits
from create_examples_n_features_with_type import DropExample, DropFeatures, read_file, write_file, split_digits
from finetune_on_drop_me import DropDataset

from sklearn.decomposition import PCA #主成分分析器
from collections import defaultdict,Counter
import argparse
def main(args):
    example_dir_name = args.examples_n_features_dir.split("/")[1]
    os.makedirs("./"+args.model_dir+"/"+example_dir_name,exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    probing_data = DropDataset(args, 'eval')
    num_layers = args.num_layers
    model_dir = args.model_dir
    model = BERT_n_layer(num_hidden_layer=num_layers)
    # model.load_state_dict(torch.load(f"{model_dir}/checkpoint_best.pt"))
    if len(args.model_dir_pt):
        model_pt_dst = f"{args.model_dir_pt}/{model_dir[7:]}"
    else:
        model_pt_dst = model_dir
    model.load_state_dict(torch.load(f"{model_pt_dst}/checkpoint_best.pt")) 
    # model.load_state_dict(torch.load(f"/work01/yuta_m/arithmetic_probing/{model_dir}/checkpoint_best.pt")) 

    type_text_str = np.array([examples.type_text.replace(" ","") for examples in probing_data.examples])
    calc_type_num_dict=dict()
    for i,a in enumerate(set(type_text_str)):
    #     print(i,a)
        calc_type_num_dict[a] = i
    type_text = np.array([calc_type_num_dict[text] for text in type_text_str])
    numbers = np.array([float(feature[-1][0]) for feature in probing_data])
    passage_text = np.array(["".join(examples.passage_tokens) for examples in probing_data.examples])
    # focus_type = calc_type_num_dict[args.focus_type.replace(" ","")]
    # intermed_values = [examples.intermed_values for examples in probing_data.examples if calc_type_num_dict[examples.type_text.replace(" ","")] == focus_type]
    intermed_values = [examples.intermed_values for examples in probing_data.examples ]
    # 特定のタイプだけを取り出すのをやめる(そもそもこのコードは数式ゴチャまぜデータセットには使わないので)
    # print(f"計算式:{passage_text[np.where(type_text==focus_type)][0]}")
    # print(f"計算式:{passage_text[0]}")
    # formula_type = re.sub(r"\d+","x",passage_text[0])
    formula_types = [example.type_text for example in probing_data.examples]
    assert len(set(formula_types))==1, f"計算式のタイプは1通りが想定されますが、{len(set(formula_types))}通りあります。"
    formula_type = probing_data.examples[0].type_text
    print(f"計算式タイプ:{formula_type}")

    X = create_dataset_for_visualize_and_probe_all_layer(model,probing_data,args,device,
    x_dim=768,concat=True)

    
    num_feature = args.num_features
    features_pca, explained_values,pcas = dimention_reduction(X,num_layers,num_feature,output_model=True)
    with open(f"{model_dir}/{example_dir_name}/pcas_{num_feature}.model","wb") as fi:
        pickle.dump(pcas,fi)
    med_dict, max_depth = compile_med_val(intermed_values)
    middle_values_list, indexs = decompose_intermed_results(med_dict,max_depth)
    print(f"intermediate results={args.index}")
    assert len(args.index)==0 or len(args.index)==len(indexs), f"len of args.index has to be equal to len(indexs),{len(args.index)}!={len(indexs)}"
    if len(args.index)==len(indexs):
        indexs = args.index


    if args.together:
        fig = plt.figure(figsize=(20,9))
    for layer in tqdm(range(1,num_layers+1)):
        if args.together:
            ax1 = fig.add_subplot(2,3,layer)
        else:
            fig,ax1 = plt.subplots(figsize=(12,7))
        df = middle_values2correlation(middle_values_list,features_pca,layer,num_feature,
            indexs=indexs)
        #g = sns.heatmap(df,cmap='bwr',vmax=1,vmin=-1,square=True,cbar_kws={"shrink": 0.5})
        if args.together:
            g = sns.heatmap(df,cmap='Blues',vmax=1,vmin=0,ax=ax1,annot=args.annot,fmt=".2f") #abs
        else:
            # g = sns.heatmap(df,cmap='Blues',vmax=1,vmin=0,square=True,cbar_kws={"shrink": 0.6},annot=True,fmt=".2f",ax=ax1) #abs
            g = sns.heatmap(df,cmap='Blues',vmax=1,vmin=0,ax=ax1,annot=args.annot,fmt=".2f") #abs
        if args.variance:
            ax2 = ax1.twinx()
            ax2.plot(np.arange(num_feature)+0.5,explained_values[layer],c="orange",marker=".",label="contribution ratio")
            handles, labels = ax2.get_legend_handles_labels()
            ax2.set_ylim(0,1)
            if not args.together:
                ax2.legend()
        ax1.set_title(f"Layer{layer}",fontsize=18)
        g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right',fontsize=15)
        ax1.set_xticklabels(range(1,num_feature+1),fontsize=20)
    #     plt.title(f"{layer}層目の主成分と各値との相関関係")
        if not args.together:
            plt.savefig(f"{model_dir}/{example_dir_name}/pca_relation_heatmap_layer{layer}_{num_feature}_square_abs_{formula_type}",bbox_inches="tight")
            plt.show()
    if args.together:
        variance =""
        if args.variance:
            variance="with_var"
            fig.legend(handles,labels,bbox_to_anchor=(0.1,1.0),loc='upper left',fontsize=20)
        plt.savefig(f"{model_dir}/{example_dir_name}/pca_relation_heatmap_alllayer_{num_feature}_abs_{variance}_{formula_type}",bbox_inches="tight")
        plt.show()
class BERT_n_layer(nn.Module):
    def __init__(self, drop_rate=0.4, output_size=1,num_hidden_layer=2):
        super().__init__()
        self.poor_bert = BertModel.from_pretrained('bert-base-uncased')
        for i in range(12-num_hidden_layer):
            #self.poor_bert.encoder.layer = self.poor_bert.encoder.layer[:num_hidden_layer]
            del(self.poor_bert.encoder.layer[11-i])
        self.poor_bert.config.num_hidden_layers = num_hidden_layer
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定

    def forward(self,**kwargs):
        ids,mask = kwargs["input_ids"],kwargs["input_mask"]        
        output = self.poor_bert(ids, attention_mask=mask,output_hidden_states=True)
        #print(output)
        self.encoder_output =  output["hidden_states"]
        # if self.cls:
        y = output["pooler_output"]
        # else:
        #     y = output["last_hidden_state"]
        #     self.last_hidden_state=y
        #     #print(y.shape)
        #     y = torch.mean(y,1)
        self.hidden_state=y
        #print(y.shape)

        y = self.fc(self.drop(y))
        return y

def abs_list(l:list):
    l = np.array(l)
    return abs(l)
def middle_values2correlation(middle_values_list,features_pca,layer,num_feature,
    indexs = ["a","b","c","d","a+b","c+d","(a+b)×(c+d)"])->pd.DataFrame:
    assert len(middle_values_list)==len(indexs), f"The length of middle_values_list({len(middle_values_list)}) must match that of index({len(indexs)})"
    co_list = []
    for middle_values in middle_values_list:
        co_list.append(abs_list([np.corrcoef(features_pca[layer][:,dim],middle_values)[0,1] for dim in range(num_feature)]))
    return pd.DataFrame(co_list,index=indexs,columns=range(1,num_feature+1))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_n_features_dir",
                        default='data/examples_n_features/',
                        type=str,
                        help="Dir containing drop examples and features.入力タイプは一つだと仮定")
    parser.add_argument("--model_dir",
                    type=str)
    parser.add_argument("--model_dir_pt",
                        type=str,
                        default="",
                        help="チェックポイントとその他を分けて保存する場合のチェックポイント保存ディレクトリの場所。指定しなかったら--model_dirが使用される。")
    # parser.add_argument("--architecture",
    #                     default = "bert",
    #                     type=str,
    #                     help = "using model architecture. ['bert','rnn','cnn','mlp','bert_cls','transformer','poor_bert'] ")                               
    parser.add_argument("--num_layers",
                        default=6,
                        type=int,
                        help="num layers of poor bert.") 
    parser.add_argument("--num_features",
                        default=10,
                        type=int,
                        help="PCAを行う際の主成分数.") 
    # parser.add_argument("--cls", action="store_true")
    parser.add_argument("--fixed_start",
                        default=0,
                        type=int,
                        help="start of fixed idx") 
    parser.add_argument("--fixed_end",
                        default=30,
                        type=int,
                        help="end of fixed idx")                      
    # parser.add_argument("--focus_type",
    #                     # default=8,
    #                     # type=int,
    #                     help="分析対象となる計算タイプ")  
    parser.add_argument("--together",
                        action="store_true",
                        help="全ての層の結果を1枚の画像で表示")  
    parser.add_argument("--variance",
                        action="store_true",
                        help="PCAの寄与率を表示")      
    parser.add_argument("--annot",
                        action="store_true",
                        help = "ヒートマップにアノテーションを付加")
    parser.add_argument("--index",
                        type=str,
                        nargs="*",
                        help = "ヒートマップのインデックスを指定")
    args = parser.parse_args()
    main(args)