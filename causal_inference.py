import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertModel,BertConfig
from transformers.models.bert.modeling_bert import *
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence


import pickle
import logging
from tqdm import tqdm
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import os
import pandas as pd
# from probing_intermediate_value import create_dataset_for_visualize_and_probe_all_layer
from utils import dimention_reduction,compile_med_val,decompose_intermed_results,min_max,norm_to_original,read_pickle
import seaborn as sns

# import modeling
#from create_examples_n_features import DropExample, DropFeatures, read_file, write_file, split_digits
from create_examples_n_features_with_type import DropExample, DropFeatures, read_file, write_file, split_digits
from finetune_on_drop_me import DropDataset

from sklearn.decomposition import PCA #主成分分析器
from collections import defaultdict,Counter
import argparse
from sympy import *
import re #sympyとreの順番を入れ替えない
from change_component import BertModelv2,BertEncoderv2,BERT_n_layer,create_dataset_for_visualize_and_probe_all_layer,decide_ratio_boundary,plot_result

def main(args):

    #諸々の準備
    target_arg = args.target_arg
    example_dir_name = args.examples_n_features_dir.split("/")[1]
    os.makedirs(f"./{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}",exist_ok=True)
    device_ = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    probing_data = DropDataset(args, 'eval')
    example = probing_data.examples[args.idx]
    origin_passage = "".join(example.passage_tokens)
    orig_answer = int(example.answer_texts[0].replace(" ",""))
    
    print(f"{example.type_text}: {origin_passage}={orig_answer}の途中結果{target_arg}に着目")
    num_layers = args.num_layers
    model_dir = args.model_dir
    with open(f"./{args.model_dir}/min_max.txt") as fi:
        x_min,x_max = fi.read().strip().split("\n")
        x_min,x_max = float(x_min),float(x_max)
    numbers = np.array([float(feature[-1][0]) for feature in probing_data])
    intermed_values = [examples.intermed_values for examples in probing_data.examples ]
    med_dict, max_depth = compile_med_val(intermed_values)
    model_vanilla = BERT_n_layer(num_hidden_layer=num_layers,cls=True,
    eliminate_pc=False)
    if len(args.model_dir_pt):
        model_pt_dst = f"{args.model_dir_pt}/{model_dir[7:]}"
    else:
        model_pt_dst = model_dir
    model_vanilla.load_state_dict(torch.load(f"{model_pt_dst}/checkpoint_best.pt")) 
    # model_vanilla.load_state_dict(torch.load(f"/work01/yuta_m/arithmetic_probing/{model_dir}/checkpoint_best.pt"))
    X = create_dataset_for_visualize_and_probe_all_layer(model_vanilla,probing_data,args,device_,
        x_dim=768,concat=True)
    # with open(f"{model_dir}/{example_dir_name}/pcas_{args.num_features}.model","rb") as fi:
    #     pcas = pickle.load(fi)
    pcas = read_pickle(f"{model_dir}/{example_dir_name}/pcas_{args.num_features}.model")
    eliminate_layer = args.eliminate_layer
    eliminate_k= args.eliminate_k
    features_layer = pcas[eliminate_layer-1].transform(X[eliminate_layer])

    # change_component
    component_min,component_max = min(features_layer[:,eliminate_k-1]),max(features_layer[:,eliminate_k-1])
    # target_component = features_layer[args.idx,eliminate_k-1]
    original_weight = features_layer[args.idx,eliminate_k-1]
    max_boundary, min_boundary = decide_ratio_boundary(original_weight, component_max,component_min)
    model = BERT_n_layer(num_hidden_layer=num_layers,cls=True,
    eliminate_pc=True,eliminate_layer= eliminate_layer ,eliminate_k= eliminate_k ,pcas=pcas,
    eliminate_rand=False)
    # model.load_state_dict(torch.load(f"/work01/yuta_m/arithmetic_probing/{model_dir}/checkpoint_best.pt"))
    model.load_state_dict(torch.load(f"{model_pt_dst}/checkpoint_best.pt")) 
    model.to(device_)
    model.eval()
    # components_ratio, result_manipulate_components = move_component(probing_data,model,max_boundary,min_boundary,device_,x_max,x_min)
    manipulated_weights_ratio,result_manipulate_weights,orig_prediction = manipulate(probing_data,args.idx,model,min_boundary,max_boundary,device_,x_min,x_max)
    manipulated_weights = manipulated_weights_ratio*original_weight
    # plot_result(components_ratio,result_manipulate_components,probing_data, numbers, min_boundary,max_boundary ,args)
    fig,ax1 = plot_manipulation(manipulated_weights, result_manipulate_weights, original_weight, 
                  orig_prediction, component_min, component_max,example)
    ax1.set_xlabel(rf"Weight of principal component $p^{eliminate_layer}_{{{eliminate_k}}}$",fontsize=25)
    ax1.set_ylabel("Model Prediction",fontsize=20)
    plt.savefig(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}/result_change_layer{args.eliminate_layer}_k{args.eliminate_k}_idx{args.idx}_{args.num_features}features", bbox_inches="tight")
    plt.show()

    with torch.no_grad():
        data = probing_data[args.idx]
        data = tuple(t.to(device_) for t in data)
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,label = data
        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        label = min_max(label,x_min,x_max)
        x_len = torch.sum(input_mask,dim=1)
        kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len,"magnitude":min_boundary}
        result = model(**kwargs)
        result_min = norm_to_original(result,x_max=x_max,x_min=x_min )[0][0].cpu().detach().numpy().item()
        kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len,"magnitude":max_boundary}
        result = model(**kwargs)
        result_max = norm_to_original(result,x_max=x_max,x_min=x_min )[0][0].cpu().detach().numpy().item()
    print("impact of manipulation=",abs(result_max-result_min))

    # 局所性を仮定した時の主成分倍率-途中結果の関係
    f, variable, ans_symbol, orig_answer = solve_equation(example,target_arg)
    print(f"解く方程式:{target_arg}={f}")
    sub_list,target_number = get_variable(variable, target_arg,med_dict,args.idx,depth=1)
    intermed_result_hypo = np.array([f[0].subs(sub_list+[(ans_symbol,result)]) for result in result_manipulate_weights],dtype=float)
    orig_intermed_value = int(f[0].subs(sub_list+[(ans_symbol,orig_answer)]))
    orig_intermed_prediction = int(f[0].subs(sub_list+[(ans_symbol,orig_prediction)]))
    fig,ax1 = plot_manipulation(manipulated_weights, intermed_result_hypo, original_weight, 
                  orig_intermed_prediction, component_min, component_max,example)
    ax1.set_xlabel(rf"Weight of principal component $p^{eliminate_layer}_{{{eliminate_k}}}$",fontsize=25)
    ax1.set_ylabel(f"Intermediate value: ${target_arg}$",fontsize=20)
    plt.savefig(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}/intermediate_result_change_layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features", bbox_inches="tight")
    plt.show()


    # 倍率-推論結果と倍率-途中結果を一つの図にする(論文用)
    if args.paper:
        twin = True
        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_subplot(111)

        ax1.plot(manipulated_weights,result_manipulate_weights,label="操作済みモデル",marker=".")
        # ans = int(probing_data.examples[args.idx].answer_texts[0].replace(" ",""))
        ax1.plot(original_weight,orig_answer,marker=".",label="訓練済みモデル",markersize=20)

        xmin, xmax = ax1.get_xlim()
        ax1.set_xlim(xmin,xmax)
        ax1.axvspan(xmin,component_min,alpha=0.2,color="black")
        ax1.axvspan(component_max,xmax,alpha=0.2,color="black")
        # ax1.set_title(f"{args.eliminate_layer}層目の第{args.eliminate_k}主成分の値を操作した時の推論結果の変化\n {''.join(probing_data.examples[args.idx].passage_tokens)}")
        # ax1.set_title(f"Original formula: {''.join(example.passage_tokens)}",fontsize=20)
        # ax1.set_xlabel(rf"Weight of principal component $p^{eliminate_layer}_{{{eliminate_k}}}$",fontsize=25)
        ax1.set_xlabel(rf"主成分$p^{eliminate_layer}_{{{eliminate_k}}}$の重み",fontsize=25)

        # ax1.set_ylabel("Model prediction",fontsize=20)
        ax1.set_ylabel("モデル予測",fontsize=20)
        # ax1.set_yticks(fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax1.set_ylim(min(result_manipulate_weights)-200,max(result_manipulate_weights)+200)
        ax1.legend(loc='lower right', fontsize=13)
        if twin:
            ax2 = ax1.twinx()
        #     ax2.set_ylabel(f"Intermediate results: {target_arg}",fontsize=20)
            ax2.set_ylabel(f"途中結果: {target_arg}",fontsize=20)
            ax2.set_ylim(min(intermed_result_hypo)-200,max(intermed_result_hypo)+200)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            diff = list(result_manipulate_weights-intermed_result_hypo)
            if diff.count(diff[0])!=len(diff):
                ax2.invert_yaxis()
        plt.savefig(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}/"+
        f"manipulate_result_layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features.pdf", bbox_inches="tight")
    #途中結果-倍率グラフを作る
    min_args, max_args=100, 1000
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


    target = str(target_number)
    
    assert example.passage_tokens.count(target)==1, f"{target}が複数あります。異なるインスタンスを選んでください。{example.passage_tokens}"
    target_idx = example.passage_tokens.index(target)
    print(f"置換項={target},置換位置={target_idx}")
    features,passages = example_to_feature(example,tokenizer,target_idx,min_args,max_args)
    answers = np.array([feature[-1].item() for feature in features])

    results,labels = evaluate(features,model_vanilla,x_max,x_min,device_)
    # 回帰がちゃんとできるかの確認
    print(f"R2={r2_score(labels,results):.3f}")

    X_locality = create_dataset_diy(model_vanilla,features,args,device_,\
    x_dim=768,cutting=False,concat=True,output_attention=False,disable=False)
    features_layer_locality = pcas[eliminate_layer-1].transform(X_locality[eliminate_layer])
    actual_weights_ratio=  features_layer_locality[:,eliminate_k-1]/original_weight
    actual_weights = features_layer_locality[:,eliminate_k-1]
    plot_causal_inference(intermed_result_hypo,manipulated_weights, actual_weights,original_weight,\
        min_args,max_args,orig_intermed_prediction,component_min, component_max,args)

    # plt.savefig(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{args.target_arg}/predict_component_change_layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features_shade.pdf", bbox_inches="tight")
    plt.savefig(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{args.target_arg}/predict_component_change_layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features_shade.png", bbox_inches="tight")
    plt.show()



    save=True
    layout = go.Layout(width=1200,height = 700)
    fig = go.Figure(layout=layout)
    plotly_causal_inference(fig,intermed_result_hypo,result_manipulate_weights,manipulated_weights,actual_weights,\
    min_args,max_args,orig_intermed_value,origin_passage,passages,target_arg)
    fig.update_layout(title = f"{origin_passage}のmanipulation")
    fig.update_xaxes(title_text = "途中結果")
    fig.update_yaxes(title_text = f"{eliminate_layer}層目の第{eliminate_k}主成分の重み")
    if save:
        print(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}/predict_component_change_layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features.html")
        fig.write_html(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}/predict_component_change_layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features.html")


    # 実際に主成分値を変えた時の推論結果
    x = intermed_result_hypo
    y = manipulated_weights
    intermed_x_for_predict = np.array(range(min_args,max_args))
    res4=np.polyfit(x, y, 4) #4次式で近似
    predicted_weights = np.poly1d(res4)(intermed_x_for_predict) 
    predicted_weights_ratio = predicted_weights/original_weight
    plt.figure(figsize=(12,7))
    plt.scatter(x, y, label='元データ')
    plt.xlabel("途中結果",fontsize=20)
    plt.ylabel("主成分倍率",fontsize=20)
    plt.plot(intermed_x_for_predict, predicted_weights ,label="近似結果")
    plt.title("4次式での近似結果",fontsize=20)
    plt.legend()
    plt.savefig(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}/Approximation_results_quadratic equation_layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features", bbox_inches="tight")
    print(f"局所性の仮定の正しさ:R={np.corrcoef(actual_weights, predicted_weights)[0,1]:.3f}")
    print(f"局所性の仮定の正しさ:R^2={r2_score(actual_weights, predicted_weights):.3f}")
    
    result_manipulate_components_predict = move_component_follow_predict(probing_data,model,predicted_weights_ratio,device_,x_max,x_min)
    layout = go.Layout(width=1200,height = 700)
    fig = go.Figure(layout=layout)
    plotly_causal_inference_eval(fig,intermed_x_for_predict,result_manipulate_components_predict,\
    results,passages,actual_weights_ratio,answers,predicted_weights_ratio,
    orig_answer,orig_intermed_value,origin_passage,target_arg)    

    print(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}/causal_inference_"+
        f"layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features.htmlに保存")
    fig.write_html(f"{args.model_dir}/{example_dir_name}/idx_{args.idx}_{target_arg}/causal_inference_"+
        f"layer{eliminate_layer}_k{eliminate_k}_idx{args.idx}_{args.num_features}features.html")


def manipulate(probing_data,idx,model,min_boundary,max_boundary,device_,x_min,x_max,verbose=False):
    """_summary_

    Args:
        probing_data (_type_): _description_
        idx (_type_): _description_
        model (_type_): _description_
        min_boundary (_type_): _description_
        max_boundary (_type_): _description_
        device_ (_type_): _description_
        x_min (_type_): _description_
        x_max (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
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
def plot_manipulation(manipulated_components, result_manipulate_components, original_weight, orig_prediction, 
                      component_min, component_max,example):
    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(111)

    ax1.plot(manipulated_components,result_manipulate_components,label="Manipulated model",marker=".")
    # ans = int(probing_data.examples[args.idx].answer_texts[0].replace(" ",""))
    ax1.plot(original_weight,orig_prediction,marker=".",label="Original model",markersize=20)

    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim(xmin,xmax)
    ax1.axvspan(xmin,component_min,alpha=0.2,color="black")
    ax1.axvspan(component_max,xmax,alpha=0.2,color="black")
    # ax1.set_title(f"{args.eliminate_layer}層目の第{args.eliminate_k}主成分の値を操作した時の推論結果の変化\n {''.join(probing_data.examples[args.idx].passage_tokens)}")
    ax1.set_title(f"Original formula: {''.join(example.passage_tokens)}",fontsize=20)
    # ax1.set_yticks(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax1.set_ylim(min(result_manipulate_components)-200,max(result_manipulate_components)+200)
    ax1.legend(loc='lower right', fontsize=13)
    return fig,ax1
def solve_equation(example, target_arg):
    variable = re.findall("(\w)",example.type_text)
    var(",".join(variable))
    ans_symbol = symbols('ans')
    orig_answer = int(example.answer_texts[0].replace(" ","")) #元の答え
    eq = Eq(ans_symbol ,eval(example.type_text) ) #方程式を定義
    f = solve(eq,eval(target_arg)) #target_arg = ?の形にする
    return f,variable,ans_symbol,orig_answer
def get_variable(variable, target_arg,med_dict,idx,depth=1):
    sub_list=[]
    for i,v in enumerate(variable):
        if v == target_arg:
            target_number = med_dict[depth][i][idx]
        sub_list.append((v,med_dict[depth][i][idx]))
    return sub_list,target_number
def get_manipulated_intermed(result_manipulate_components, f, sub_list, ans_symbol):
    intermed_result_hypo = [f[0].subs(sub_list+[(ans_symbol,result)]) for result in result_manipulate_components]
    intermed_result_hypo = np.array(intermed_result_hypo,dtype=float)
    
    return intermed_result_hypo
def example_to_feature(example,tokenizer,target_idx,min_args=100,max_args=1000):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    START_TOK, END_TOK, SPAN_SEP, IGNORE_IDX, MAX_DECODING_STEPS = '@', '\\', ';', 0, 20 
    max_seq_length=args.fixed_end-args.fixed_start
    max_decoding_steps=11
    indiv_digits=True
    # logger=None
    logger.info('creating features')
    unique_id = 1000000000
    skip_count, truncate_count = 0, 0

    tokenize = (lambda s: split_digits(tokenizer.tokenize(s))) if indiv_digits else tokenizer.tokenize

    features, all_qp_lengths = [], []
    passages = []
    # for (example_index, example) in enumerate(examples):  
    for arg in range(min_args,max_args):
        passage=""
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
        doc_token_to_orig_map = {}
        tokens.append("[CLS]")
        tokens.append("[SEP]")
        segment_ids += [0]*len(tokens)

        for i in range(len(all_doc_tokens)):
            doc_token_to_orig_map[len(tokens)] = doc_tok_to_orig_index[i]
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

        # we expect the generative head to output the wps of the joined answer text
        answer = float(eval(passage))
        features.append((torch.tensor(input_ids),torch.tensor(input_mask),torch.tensor(answer)))
        passages.append(passage)
        unique_id += 1
    return features,passages
def evaluate(features,model_vanilla,x_max,x_min,device_):

    results=[]
    labels=[]
    for data in tqdm(features):
    #     print(data)
        data = tuple(t.to(device_) for t in data)
        input_ids, input_mask,label=data
    #     print(tokenizer.convert_ids_to_tokens(input_ids.cpu().detach().numpy))
        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        label = min_max(label,x_min,x_max)
        x_len = torch.sum(input_mask,dim=1)
        kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len,"magnitude":1}
        result = model_vanilla(**kwargs)
        results.append(norm_to_original(result,x_max=x_max,x_min=x_min )[0][0].cpu().detach().numpy().item())
        labels.append(norm_to_original(label,x_max=x_max,x_min=x_min ).cpu().detach().numpy().item())
    return results,labels

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


def plot_causal_inference(intermed_result_hypo,manipulated_weights, actual_weights,original_weight,\
        min_args,max_args,orig_intermed_prediction,component_min, component_max,args):
    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(111)
    # ax1.plot(intermed_result_hypo,manipulated_components,label="Predicted component",marker=".",)
    # ax1.plot(range(min_args,max_args),features_layer_locality[:,eliminate_k-1],label="Actual component")
    #日本語
    ax1.plot(intermed_result_hypo,manipulated_weights,label=fr"予測関数$f_p({args.target_arg})$",marker=".",)
    ax1.plot(range(min_args,max_args),actual_weights,label=rf"実測関数$f_a({args.target_arg})$")

    ax1.plot(orig_intermed_prediction, original_weight,marker=".",label="元データ",markersize=20)
    # plt.hlines(y=max_boundary,xmax = max_args,xmin = min_args,color="r",linestyle=":",label="Max weight of the component")
    # plt.hlines(y=min_boundary,xmax = max_args,xmin = min_args,color="m",linestyle=":",label="Min weight of the component")
    ymin, ymax = ax1.get_ylim()
    print(ymin,ymax)
    ax1.set_ylim(ymin,ymax)
    ax1.set_xlim(0,1100)
    ax1.axhspan(ymin,component_min,alpha=0.2,color="black")
    ax1.axhspan(component_max,ymax,alpha=0.2,color="black")
    # plt.grid()
    # plt.title(f"Original formula: {''.join(example.passage_tokens)}",fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # ax1.set_title(f"Original formula: {''.join(example.passage_tokens)}",fontsize=20)
    # ax1.set_ylabel(rf"Weight of $p^{eliminate_layer}_{{{eliminate_k}}}$",fontsize=25,labelpad=-10)
    ax1.set_ylabel(rf"主成分$p^{args.eliminate_layer}_{{{args.eliminate_k}}}$の重み",fontsize=25,labelpad=-10)
    # ax1.set_xlabel(rf"Intermediate value: ${target_arg}$",fontsize=25)
    ax1.set_xlabel(rf"途中結果: ${args.target_arg}$",fontsize=25)
    ax1.legend(fontsize=15, loc='upper right')

def plotly_causal_inference(fig,intermed_result_hypo,result_manipulate_components,manipulated_weights,actual_weights,\
    min_args,max_args,orig_intermed_value,origin_passage,passages,target_arg):

        
    fig.add_trace(go.Scatter(
        mode="markers+lines",
            x = intermed_result_hypo,
            y = manipulated_weights,
        name =rf"予測関数$f_p({args.target_arg})$",
        customdata=np.stack((manipulated_weights,result_manipulate_components),axis=-1) ,
    #         hovertext= origin_passage+manipulated_weights+"から予測",
            hovertemplate=origin_passage+"の主成分を%{customdata[0]:.2f}倍した推論結果:%{customdata[1]:.1f}から"+target_arg+"=%{x:.2f}"
        ))
    fig.add_trace(go.Scatter(
            x = np.array(range(min_args,max_args)),
            y = actual_weights,
    #     text= passages,
        customdata=np.stack([passages]).T,
        name =rf"実測関数$f_a({args.target_arg})$",
        hovertemplate= "数式:%{customdata[0]}, 主成分値倍率:%{y:.2f}"
        ))
    fig.add_trace(go.Scatter(
        x = [orig_intermed_value],
        y = [1],
        marker_size=20,
        name = "元データ"
    ))
    # customdata.shape = (len(data),len(customdata))でないといけない
    # fig.update_layout(title = f"{origin_passage}からの主成分予測")
    # fig.update_xaxes(title_text = "途中結果")
    # fig.update_yaxes(title_text = f"{eliminate_layer}層目の第{eliminate_k}主成分倍率")

def move_component_follow_predict(probing_data,model,predicted_weights_ratio,device_,x_max,x_min):
# change_componentを行う
    model.to(device_)
    model.eval()
    with torch.no_grad():
        data = probing_data[args.idx]
        data = tuple(t.to(device_) for t in data)
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,label = data
        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        # label = min_max(label,x_min,x_max)
        x_len = torch.sum(input_mask,dim=1)
        result_manipulate_components=[]
        for magnitude in tqdm(predicted_weights_ratio):
            kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len,"magnitude":magnitude}
            result = model(**kwargs)
    #         print(f"主成分を{magnitude:.2f}倍:{norm_to_original(result,x_max=x_max,x_min=x_min )[0][0]}")
            result_manipulate_components.append(norm_to_original(result,x_max=x_max,x_min=x_min )[0][0].cpu().detach().numpy().item())
    return result_manipulate_components


def plotly_causal_inference_eval(fig,intermed_x_for_predict,result_manipulate_components_predict,\
    results,passages,actual_weights_ratio,answers,predicted_weights_ratio,
    orig_answer,orig_intermed_value,origin_passage,target_arg):
    fig.add_trace(go.Scatter(
        mode="lines",
            x = intermed_x_for_predict,
            y = results,
        name ="主成分に沿って重みを操作しない通常の予測",
    #     customdata=np.stack((predicted_weights,result_manipulate_components),axis=-1) ,
        customdata=np.stack((passages,actual_weights_ratio),axis=-1),
    #         hovertext= origin_passage+manipulated_weights+"から予測",
    #         hovertemplate=origin_passage+"の主成分を%{customdata[0]:.2f}倍した推論結果:%{customdata[1]:.1f}からb=%{x:.2f}"
        hovertemplate= "数式:%{customdata[0]},主成分重み:基準データの%{customdata[1]:.1f}倍 推論結果:%{y:.2f}"
        ))
    fig.add_trace(go.Scatter(
            x = intermed_x_for_predict,
            y = answers,
    #     text= passages,
        customdata=np.stack([passages]).T,
        name ="gold",
        hovertemplate= "数式:%{customdata[0]}, 答え:%{y:.2f}"
        ))
    fig.add_trace(go.Scatter(
            x = intermed_x_for_predict,
            y = result_manipulate_components_predict,
        text= passages,
        name ="途中結果に応じた主成分重みを予測し、変化",
        customdata=np.stack((predicted_weights_ratio,result_manipulate_components_predict),axis=-1) ,
    #     hovertemplate= "数式:%{customdata[0]}, 主成分値:%{y:.2f}"
        hovertemplate=target_arg+"=%{x:.1f}にするため、"+origin_passage+"の主成分の重みを%{customdata[0]:.2f}倍"
        ))
    fig.add_trace(go.Scatter(
        x = [orig_intermed_value],
        y = [orig_answer],
        marker_size=20,
        name = "元データ"
    ))
    # customdata.shape = (len(data),len(customdata))でないといけない
    fig.update_layout(title = f"{origin_passage}から主成分の重みを予測し、推論")
    fig.update_xaxes(title_text = f"途中結果{target_arg}")
    fig.update_yaxes(title_text = f"推論結果")
# fig.update_titles(title_text = f"{origin_passage}からの主成分予測",fontsize=17)
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
                        default=20,
                        type=int,
                        help="PCAを行う際の主成分数.") 
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
    parser.add_argument("--idx",
                        default=1,
                        type=int,
                        help="主成分値を変化させるインスタンス")
    parser.add_argument("--target_arg",
                        type=str,
                        help = "途中結果 (e.g. 'b','a','a-c') ") #現在は項にのみ対応
    parser.add_argument("--paper",
                        action="store_true",
                        help = "論文用の図を出力")                         
    args = parser.parse_args()
    main(args)