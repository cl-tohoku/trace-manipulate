import  numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA
import pickle
def min_max(x:np.array, x_min:float,x_max:float):
    """xの正規化を行う

    Args:
        x (np.array): _description_
        x_min (float): _description_
        x_max (float): _description_

    Returns:
        _type_: _description_
    """    
    result = (x-x_min)/(x_max-x_min)
    return result
def norm_to_original(x_norm:np.array, x_min:float, x_max:float):
    """ 正規化を元に戻す

    Args:
        x_norm (np.array): _description_
        x_min (float): _description_
        x_max (float): _description_

    Returns:
        _type_: _description_
    """    
    return x_norm*(x_max-x_min)+x_min
def create_dataset_for_visualize_and_probe_all_layer(model,probing_data,args,device,x_min=0,x_max=1000,\
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

            input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,label = data
            input_ids = torch.unsqueeze(input_ids,0)
            input_mask = torch.unsqueeze(input_mask,0)
#             label = min_max(label,x_min,x_max)
            x_len = torch.sum(input_mask,dim=1)
            kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len}
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
def dimention_reduction(X:np.array,num_layers:int,num_features:int,output_model=False):
    """PCAによる時限削減を行う

    Args:
        X (np.array): 隠れ層の内部表現。(num_layers+1, num_data, vector_size)
        num_layers (int): モデルの層数
        num_features (int): PCA後の次元数

    Returns:
        features_pca(dict): 主成分

        explained_values(np.array): 主成分の寄与度 
        pcas(list): 各層におけるPCAモデル layer層目のpcaモデル=pcas[layer-1]
    """
    explained_values = np.zeros((num_layers+1,num_features))
    features_pca = dict()
    pcas=[]
    print("PCA開始")

    for layer in tqdm(range(1,num_layers+1)):
        pca = PCA(n_components=num_features)
        feature = pca.fit_transform(X[layer])
        explained_values[layer] = pca.explained_variance_ratio_
        features_pca[layer] = feature
        pcas.append(pca)
    print("PCA終了")
    if output_model:
        return features_pca, explained_values,pcas
    return features_pca, explained_values
def compile_med_val(intermed_values):
    """途中結果の値を計算木の深さごとにまとめる

    Args:
        intermed_values (list): DropExamples.intermed_valuesのリスト

    Returns:
        med_dict(dict): 計算木の深さごとにまとめられた途中結果
        max_depth(int): 計算木の深さの最大値
    """
    med_dict = defaultdict(list)
    max_depth = max(list(map(int, intermed_values[0].keys())))
    print(f"max_depth:{max_depth}")
    for depth in range(1, max_depth+1):
        intermed_num = len(intermed_values[0][str(depth)])
        for idx in range(intermed_num):
            med_dict[depth].append( np.array([med_by_depth[str(depth)][idx] for med_by_depth in intermed_values]) )
            #数式の形が同じじゃないとエラーが出るよ
    return med_dict,max_depth
def decompose_intermed_results(med_dict:dict,max_depth:int):
    """ 
        深さごとにまとめられた途中結果をフラットにする

    Args:
        med_dict(dict): 計算木の深さごとにまとめられた途中結果
        max_depth(int): 計算木の深さの最大値

    Returns:
        middle_values_list(np.array): 途中結果のリスト(num_intermed_values, num_data)
        indexs(list): 途中結果の名前の情報(num_intermed_values)
    """
    middle_values_list = []
    indexs = []
    for depth in range(1,max_depth+1):
        intermed_num = len(med_dict[depth])
        for idx in range(intermed_num):
            middle_values_list.append(med_dict[depth][idx])
            indexs.append(f"dep{depth}_{idx}")
    return np.array(middle_values_list), indexs
def abs_list(l:list):
    l = np.array(l)
    return abs(l)

def read_pickle(file):
    with open(file,"rb") as f:
        obj = pickle.load(f)
    return obj