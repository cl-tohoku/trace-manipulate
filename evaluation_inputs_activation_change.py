import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertModel,BertConfig
import transformers
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

from create_examples_n_features_with_type import DropExample, DropFeatures, read_file, write_file, split_digits
from finetune_on_drop_me import DropDataset
import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from umap import UMAP
from collections import defaultdict,Counter
import pandas as pd
import seaborn as sns
from sympy import *
import re #ここの順番大事(sympyにreがある)
import argparse
import numpy as np
from tqdm.auto import tqdm
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import os
import pickle

from plotly.subplots import make_subplots
from change_component import BertModelv2,BertEncoderv2,BERT_n_layer,create_dataset_for_visualize_and_probe_all_layer,decide_ratio_boundary
from utils import dimention_reduction,compile_med_val,decompose_intermed_results,min_max,norm_to_original,read_pickle

def main(args):


    # transformers.logging.set_verbosity_info()
    example_dir_name = args.examples_n_features_dir.split("/")[1]
    os.makedirs("./"+args.model_dir+"/"+example_dir_name,exist_ok=True)
    device_ = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    print(device_)
    probing_data = DropDataset(args, 'eval')
    num_layers = args.num_layers
    model_dir = args.model_dir
    target_arg = args.target_arg
    with open(f"./{args.model_dir}/min_max.txt") as fi:
        x_min,x_max = fi.read().strip().split("\n")
        x_min,x_max = float(x_min),float(x_max)
        print(f"x_min:{x_min},x_max:{x_max}")
    numbers = np.array([float(feature[-1][0]) for feature in probing_data])
    intermed_values = [examples.intermed_values for examples in probing_data.examples ]
    med_dict, max_depth = compile_med_val(intermed_values)

    model_vanilla = BERT_n_layer(num_hidden_layer=num_layers,cls=True,
        eliminate_pc=False)
    # model_vanilla.load_state_dict(torch.load(f"/work01/yuta_m/arithmetic_probing/{model_dir}/checkpoint_best.pt"))
    if len(args.model_dir_pt):
        model_pt_dst = f"{args.model_dir_pt}/{model_dir[7:]}"
    else:
        model_pt_dst = model_dir
    model_vanilla.load_state_dict(torch.load(f"{model_pt_dst}/checkpoint_best.pt")) 
    X = create_dataset_for_visualize_and_probe_all_layer(model_vanilla,probing_data,args,device_,
        x_dim=768,concat=True)
    pcas =read_pickle(f"{model_dir}/{example_dir_name}/pcas_{args.num_features}.model")
    eliminate_layer = args.eliminate_layer
    eliminate_k= args.eliminate_k
    features_layer = pcas[eliminate_layer-1].transform(X[eliminate_layer])
    
    instance_ids = range(1000) #先頭1000個のデータ
    # instance_ids = range(10) #テスト用
    component_min = min(features_layer[:,eliminate_k-1])
    component_max = max(features_layer[:,eliminate_k-1])
    loss_array = np.zeros((2*len(instance_ids),900))
    loss_array[:,:] = np.nan
    errors_data = []
    model = BERT_n_layer(num_hidden_layer=num_layers,cls=True,
    eliminate_pc=True,eliminate_layer= eliminate_layer ,eliminate_k= eliminate_k ,pcas=pcas,
    eliminate_rand=False)
    model.load_state_dict(torch.load(f"{model_pt_dst}/checkpoint_best.pt")) 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for i,idx in tqdm(enumerate(instance_ids),total=len(instance_ids)):
        
        example = probing_data.examples[idx]
        orig_answer = int(example.answer_texts[0].replace(" ",""))


        target_component = features_layer[idx,eliminate_k-1]
        max_boundary, min_boundary = decide_ratio_boundary(features_layer[idx,eliminate_k-1], component_max,component_min)

        # manipulate
        manipulated_components_ratio,result_manipulate_components,orig_prediction = manipulate(probing_data,idx,model,min_boundary,max_boundary,device_,x_min,x_max)
        manipulated_components = manipulated_components_ratio*target_component

        # 途中結果操作の仮定
        f, variable, ans_symbol, orig_answer = solve_equation(example,target_arg)
        sub_list,target_number = get_variable(variable, target_arg,med_dict,idx)
        orig_intermed = int(f[0].subs(sub_list+[(ans_symbol,orig_answer)]))
        orig_intermed_prediction = int(f[0].subs(sub_list+[(ans_symbol,orig_prediction)]))
        intermed_result_hypo = get_manipulated_intermed(result_manipulate_components, f, sub_list, ans_symbol)

        # 入力を変えた時との比較
        target = str(target_number)
        if example.passage_tokens.count(target)!=1:
            print(f"{target}が複数あります。{example.passage_tokens}")
            continue
        # assert example.passage_tokens.count(target)==1, f"{target}が複数あります。{example.passage_tokens}"
        target_idx = example.passage_tokens.index(target)

        features, passages = inputs_change(example, target_idx,tokenizer,args.fixed_end-args.fixed_start)
        X_locality = create_dataset_diy(model_vanilla,features,args,device_,\
            x_dim=768,cutting=False,concat=True,output_attention=False,disable=True)
        features_layer_locality = pcas[eliminate_layer-1].transform(X_locality[eliminate_layer])
        components_ratio_actual =  features_layer_locality[:,eliminate_k-1]/target_component
        actual_weight = features_layer_locality[:,eliminate_k-1]
        
        min_args=100
        max_args=1000
        #定量評価
        x = intermed_result_hypo
        y = manipulated_components
        intermed_x_for_predict = np.array(range(min_args,max_args))
        res4=np.polyfit(x, y, 4) #4次式で近似
        predicted_weight = np.poly1d(res4)(intermed_x_for_predict) 
        errors = abs(predicted_weight-actual_weight)
        errors_data.extend([{'dist_to_original':(im-orig_intermed),'error':error,'idx':idx} for error,im in zip(errors,intermed_x_for_predict)])
    errors_data = pd.DataFrame(errors_data)

    save_dir = f"{args.model_dir}/{example_dir_name}"
    # np.save(f"{save_dir}/eval_{target_arg}_l{args.eliminate_layer}_k{eliminate_k}_loss.npy",loss_array)
    with open(f"{save_dir}/eval_{target_arg}_l{args.eliminate_layer}_k{eliminate_k}_error.pkl","wb") as f:
        pickle.dump(errors_data,f)
    print(f"誤差データを{save_dir}/eval_{target_arg}_l{args.eliminate_layer}_k{eliminate_k}_error.pklに保存")
    sns.set()
    plt.rcParams["figure.figsize"] = (8, 5)
    japanize_matplotlib.japanize()
    sns.lineplot(data=errors_data,x="dist_to_original",y="error",estimator=np.median)
    plt.xlabel("元データからの距離",fontsize=15)
    plt.ylabel("誤差",fontsize=15)
    plt.savefig(f"{save_dir}/eval_{target_arg}_l{args.eliminate_layer}_k{eliminate_k}_error_median.png",bbox_inches="tight")
def manipulate(probing_data,idx,model,min_boundary,max_boundary,device_,x_min,x_max,verbose=False):
    model.to(device_)
    model.eval()
    with torch.no_grad():
        data = probing_data[idx]
        if verbose:
            print("問題:","".join(probing_data.examples[idx].passage_tokens))
            print("答え:",probing_data.examples[idx].answer_texts[0].replace(" ",""))
        data = tuple(t.to(device_) for t in data)
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,label = data
        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        label = min_max(label,x_min,x_max)
        x_len = torch.sum(input_mask,dim=1)
        kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len,"magnitude":1}
        result = model(**kwargs)
        orig_prediction = norm_to_original(result,x_max=x_max,x_min=x_min )[0][0].cpu().detach().numpy().item()
    #     components_ratio = np.arange(min_boundary*1.5,max_boundary*1.5,0.2) #arange
        manipulated_components_ratio = np.linspace(min_boundary*1.2,max_boundary*1.2,20) #linspace
        result_manipulate_components=[]
        for magnitude in manipulated_components_ratio:
            kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len,"magnitude":magnitude}
            result = model(**kwargs)
    #         print(f"主成分を{magnitude:.2f}倍:{norm_to_original(result,x_max=x_max,x_min=x_min )[0][0]}")
            result_manipulate_components.append(norm_to_original(result,x_max=x_max,x_min=x_min )[0][0].cpu().detach().numpy().item())
    result_manipulate_components = np.array(result_manipulate_components)
    return manipulated_components_ratio,result_manipulate_components,orig_prediction
def solve_equation(example, target_arg):
    
    variable = re.findall("(\w)",example.type_text)
    var(",".join(variable))
    ans_symbol = symbols('ans')
    orig_answer = int(example.answer_texts[0].replace(" ","")) #元の答え
    eq = Eq(ans_symbol ,eval(example.type_text) ) #方程式を定義
    f = solve(eq,eval(target_arg)) #target_arg = ?の形にする
    return f,variable,ans_symbol,orig_answer

def get_variable(variable, target_arg,med_dict,idx):
    sub_list=[]
    for i,v in enumerate(variable):
        if v == target_arg:
            target_number = med_dict[1][i][idx]
        sub_list.append((v,med_dict[1][i][idx]))
    return sub_list,target_number
def get_manipulated_intermed(result_manipulate_components, f, sub_list, ans_symbol):
    intermed_result_hypo = []
    for result in result_manipulate_components:
        intermed_result_hypo.append(f[0].subs(sub_list+[(ans_symbol,result)]))
    intermed_result_hypo = np.array(intermed_result_hypo,dtype=float)  
    return intermed_result_hypo
def inputs_change(example, target_idx,tokenizer,max_seq_length):
    min_args=100
    max_args=1000
    # START_TOK, END_TOK, SPAN_SEP, IGNORE_IDX, MAX_DECODING_STEPS = '@', '\\', ';', 0, 20 
    # max_seq_length=25
    indiv_digits=True
    logger=None
#     logger.info('creating features')
    
    tokenize = (lambda s: split_digits(tokenizer.tokenize(s))) if indiv_digits else tokenizer.tokenize

    features, all_qp_lengths = [], []
    passages = []
    for arg in range(min_args,max_args):
        passage = ""
        doc_tok_to_orig_index = []
        doc_orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.passage_tokens):
            if i==target_idx:
                token = str(arg)
            doc_orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenize(token)
            doc_tok_to_orig_index += [i]*len(sub_tokens)
            all_doc_tokens += sub_tokens
            passage += token


        tokens, segment_ids = [], []
        tokens.append("[CLS]")
        tokens.append("[SEP]")
        segment_ids += [0]*len(tokens)

        for i in range(len(all_doc_tokens)):
            tokens.append(all_doc_tokens[i])
        tokens.append("[SEP]")
        len_tokens = len(tokens)

        segment_ids += [1]*(len(tokens) - len(segment_ids))
        tokens += ["[PAD]"]*(max_seq_length-len(tokens))
        segment_ids += [0]*(len(tokens)-len(segment_ids))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len_tokens
        input_mask += [0] * (max_seq_length-len_tokens)
        # segment_ids are 0 for toks in [CLS] Q [SEP] and 1 for P [SEP]
        assert len(segment_ids) == len(input_ids)

        answer = float(eval(passage))
        features.append((torch.tensor(input_ids),torch.tensor(input_mask),torch.tensor(answer)))
        passages.append(passage)
    return features, passages
def create_dataset_diy(model,probing_data,args,device,\
    x_dim=768,cutting=False,concat=False,output_attention=False,disable=False):
    fixed_start=args.fixed_start
    fixed_end=args.fixed_end
    head_num=12
    if output_attention:
        attns = np.zeros([args.num_layers+1,len(probing_data),head_num,(fixed_end-fixed_start),(fixed_end-fixed_start)])
    if cutting:
        X=np.zeros([args.num_layers+1,cutting,x_dim])
        y=np.zeros([cutting]) 
        
    else:
        if concat:
            X=np.zeros([args.num_layers+1,len(probing_data),x_dim*(fixed_end-fixed_start)])
            y=np.zeros([len(probing_data)])    
            
        else:
            X=np.zeros([args.num_layers+1,len(probing_data),x_dim])
            y=np.zeros([len(probing_data)]) 
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i,data in tqdm(enumerate(probing_data),total=len(probing_data),disable=disable):
            if cutting:
                if cutting==i:
                    break
            data = tuple(t.to(device) for t in data)

            input_ids, input_mask, label = data
            input_ids = torch.unsqueeze(input_ids,0)
            input_mask = torch.unsqueeze(input_mask,0)
#             label = min_max(label,x_min,x_max)
            x_len = torch.sum(input_mask,dim=1)
            kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len,"magnitude":1}
            result = model(**kwargs)
            for layer in range(args.num_layers+1):
                if layer < args.num_layers and output_attention:
                    attns[layer][i] = model.encoder_attention[layer][0, :, fixed_start:fixed_end, fixed_start:fixed_end].cpu()
                if concat:
                    X[layer][i] = model.encoder_output[layer][0][fixed_start:fixed_end].view(-1).cpu() # 結合

                else:
                    X[layer][i] = torch.mean( model.encoder_output[layer][0][fixed_start:fixed_end] , 0).cpu() #平均
                #X[layer][i] = model.encoder_output[layer][0][0].cpu() #CLS

#             y[i] = probing_data.examples[i].type_text
    if output_attention:
        return X,attns
    return X

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
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--fixed_start",
                        default=0,
                        type=int,
                        help="start of fixed idx") 
    parser.add_argument("--fixed_end",
                        default=30,
                        type=int,
                        help="end of fixed idx")      
    parser.add_argument("--eliminate_k",
                        "-ek",
                        default=1,
                        type=int,
                        help="除去する主成分")      
    parser.add_argument("--eliminate_layer",
                        "-el",
                        default=1,
                        type=int,
                        help="主成分の除去を行う層")
    parser.add_argument("--target_arg",
                        type=str,
                        help = "途中結果 (e.g. 'b','a','a-c') ") #現在は項にのみ対応
    parser.add_argument("--paper",
                        action="store_true",
                        help = "論文用の図を出力")                         
    args = parser.parse_args()
    main(args)