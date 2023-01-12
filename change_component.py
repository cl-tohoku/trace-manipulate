from cachetools import FIFOCache
from transformers import BertModel
# from transformers import *
from transformers.models.bert.modeling_bert import *
# from transformers import BertModel
import argparse
import torch 
from torch import nn
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import japanize_matplotlib
from utils import min_max, norm_to_original ,dimention_reduction
from finetune_on_drop_me import DropDataset
from create_examples_n_features_with_type import DropExample ,DropFeatures, read_file, write_file, split_digits #直接使っていないが、これがないとデータが読み込めない
def main(args):
    example_dir_name = args.examples_n_features_dir.split("/")[1]
    os.makedirs("./"+args.model_dir+"/"+example_dir_name,exist_ok=True)
    device_ = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    probing_data = DropDataset(args, 'eval')
    num_layers = args.num_layers
    model_dir = args.model_dir
    with open(f"./{args.model_dir}/min_max.txt") as fi:
        x_min,x_max = fi.read().strip().split("\n")
        x_min,x_max = float(x_min),float(x_max)
        print(f"x_min:{x_min},x_max:{x_max}")
    numbers = np.array([float(feature[-1][0]) for feature in probing_data])
    model_vanilla = BERT_n_layer(num_hidden_layer=num_layers,cls=args.cls,
        eliminate_pc=False)
    model_vanilla.load_state_dict(torch.load(f"/work01/yuta_m/arithmetic_probing/{model_dir}/checkpoint_best.pt"))
    X = create_dataset_for_visualize_and_probe_all_layer(model_vanilla,probing_data,args,device_,
        x_dim=768,concat=True)
    num_features = args.num_features
    if os.path.exists(f"{model_dir}/{example_dir_name}/pcas_{num_features}.model"):
        print("pca model already exists.")
        with open(f"{model_dir}/{example_dir_name}/pcas_{num_features}.model","rb") as fi:
            pcas = pickle.load(fi)
    else:
        
        
        _, _,pcas = dimention_reduction(X,num_layers,num_features,output_model=True)
        with open(f"{model_dir}/{example_dir_name}/pcas_{num_features}.model","wb") as fi:
            pickle.dump(pcas,fi)
    
    eliminate_layer = args.eliminate_layer
    eliminate_k= args.eliminate_k
    features_layer = pcas[eliminate_layer-1].transform(X[eliminate_layer])
    component_min = min(features_layer[:,eliminate_k-1])
    component_max = max(features_layer[:,eliminate_k-1])

    # asssertしたい X.shapeとpca.component_の次元数
    assert X.shape[2]==pcas[0].components_.shape[1],f"768*(fixed_end)-(fixed_start) must be equal to {pcas[0].components_.shape[1]}"

    # 主成分値の調整
    model = BERT_n_layer(num_hidden_layer=num_layers,cls=True,
    eliminate_pc=True,eliminate_layer= eliminate_layer ,eliminate_k= eliminate_k ,pcas=pcas,
    eliminate_rand=False)
    model.load_state_dict(torch.load(f"/work01/yuta_m/arithmetic_probing/{model_dir}/checkpoint_best.pt"))

    max_boundary, min_boundary = decide_ratio_boundary(features_layer[args.idx,eliminate_k-1], component_max,component_min)
    components_ratio, result_manipulate_components = move_component(model,probing_data,args.idx, device_, x_min, x_max, min_boundary,max_boundary)
    plot_result(components_ratio,result_manipulate_components,probing_data, numbers, min_boundary,max_boundary ,args)
    plt.savefig(f"{args.model_dir}/{example_dir_name}/result_change_layer{args.eliminate_layer}_k{args.eliminate_k}_idx{args.idx}_{num_features}features", bbox_inches="tight")
    
class BertModelv2(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoderv2(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        trans_layer=0,
        trans_num=0,
        eliminate_pc=False,
        pcas=None,
        eliminate_layer=0,
        eliminate_k = 1,
        eliminate_rand=False,
        magnitude = 0
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            trans_layer = trans_layer,
            trans_num = trans_num,
            eliminate_pc=eliminate_pc,
            pcas=pcas,
            eliminate_layer=eliminate_layer,
            eliminate_k = eliminate_k,
            eliminate_rand = eliminate_rand,
            magnitude = magnitude
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
class BertEncoderv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        trans_layer=0,
        trans_num=0,
        eliminate_pc=False,
        pcas=None,
        eliminate_layer=0,
        eliminate_k = 1,
        eliminate_rand = False,
        magnitude = 0
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if eliminate_pc and eliminate_layer==i+1:

                #Uとhidden_statesのサイズを合わせる処理をする必要がある(足りない分はゼロにすればいいはず)
                batch, seq_len, hidden_size = hidden_states.shape
                # print(f"batch={batch},seq_len={seq_len},hidden_size={hidden_size}")
                U = pcas[i].components_
                component, dim = U.shape
                # print(f"component:{component},dim:{dim}")
                lack_size = seq_len*hidden_size-dim
                # print("lack:",lack_size)
                if lack_size>0:
                    U = np.hstack((U, np.zeros((component, lack_size))))
                U_k = U[eliminate_k-1:eliminate_k]
                # hidden_states_abtt= torch.from_numpy((hidden_states[0].view(-1).cpu().detach().numpy()).dot(U.transpose()).dot(U).astype(np.float32))
                hidden_states_abtt= torch.from_numpy((hidden_states[0].view(-1).cpu().detach().numpy()).dot(U_k.transpose()).dot(U_k).astype(np.float32))
                # print("orig norm=", np.linalg.norm(hidden_states_abtt))
                if eliminate_rand:
                    np.random.seed(seed=32)
                    U_k_rand = np.random.rand(1,dim)
                    norm_orig = np.linalg.norm(U_k)
                    norm_rand = np.linalg.norm(U_k_rand)
                    U_k_rand *= (norm_orig/norm_rand)
                    hidden_states_abtt= torch.from_numpy((hidden_states[0].view(-1).cpu().detach().numpy()).dot(U_k_rand.transpose()).dot(U_k_rand).astype(np.float32))
                    # print("rand norm=", np.linalg.norm(hidden_states_abtt))
                hidden_states[0] = hidden_states[0] + (magnitude-1)*hidden_states_abtt.view(seq_len,hidden_size).to(hidden_states.device)
                # magnituide = 1でそのまま、0で除去,m>1で増幅,0<m<1で減らす
            # print(hidden_states.shape)
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BERT_n_layer(nn.Module):
    def __init__(self, drop_rate=0.4, output_size=1,num_hidden_layer=6,cls=True,eliminate_pc=True,
    eliminate_layer = None, eliminate_k=None,pcas=None,eliminate_rand=None,):
        super().__init__()
        self.cls = cls
        self.eliminate_layer = eliminate_layer
        self.eliminate_k = eliminate_k
        self.eliminate_rand = eliminate_rand
        self.pcas= pcas
        self.poor_bert = BertModelv2.from_pretrained('bert-base-uncased')
        for i in range(12-num_hidden_layer):
            #self.poor_bert.encoder.layer = self.poor_bert.encoder.layer[:num_hidden_layer]
            del(self.poor_bert.encoder.layer[11-i])
        self.poor_bert.config.num_hidden_layers = num_hidden_layer
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定


    def forward(self,**kwargs):
        
        ids,mask,magnitude = kwargs["input_ids"],kwargs["input_mask"],kwargs["magnitude"]      
        output = self.poor_bert(ids, attention_mask=mask, output_hidden_states=True,
        eliminate_pc=True, pcas=self.pcas, eliminate_k=self.eliminate_k, eliminate_layer=self.eliminate_layer,
        eliminate_rand=self.eliminate_rand, magnitude=magnitude)
        #print(output)
        self.encoder_output =  output["hidden_states"]
        if self.cls:
            y = output["pooler_output"]
        else:
            y = output["last_hidden_state"]
            self.last_hidden_state=y
            #print(y.shape)
            y = torch.mean(y,1)
        self.hidden_state=y
        #print(y.shape)

        y = self.fc(self.drop(y))
        return y
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

def decide_ratio_boundary(principal_component, component_max,component_min):
    """ 主成分値を操作する際に、データセット中の同じ層、寄与率の主成分の最大値/最小値から操作範囲を決める

    Args:
        principal_component (float): 主成分値
        component_max (float): データセット中に登場する主成分値の最大値
        component_min (float): データセット中に登場する主成分値の最大値

    Returns:
        max_boundary (float): 主成分値を操作する際のデータセット中で内挿となる最大倍率
        min_boundary (float): 主成分値を操作する際のデータセット中で内挿となる最小倍率
    """    
    if principal_component>0:
        max_boundary = component_max/principal_component
        min_boundary = component_min/principal_component
    else:
        min_boundary = component_max/principal_component
        max_boundary = component_min/principal_component
    
    return max_boundary,min_boundary
    

def move_component(model,probing_data,idx,device_,x_min,x_max,min_boundary,max_boundary):
    """_summary_

    Args:
        model (_type_): モデル
        probing_data (_type_): データセット
        idx (_type_): 主成分値を操作するデータインデックス
        device_ (_type_): device
        x_min (_type_): 訓練データ中の答えの最小値
        x_max (_type_): 訓練データ中の答えの最大値
        min_boundary (_type_): データセット中の主成分値の最小値にするための操作倍率
        max_boundary (_type_): データセット中の主成分値の最大値にするための操作倍率

    Returns:
        components_ratio(np.arrray): 主成分操作倍率の配列
        result_manipulate_components(list): 主成分値を操作した際の推論結果のリスト
    """
    model.to(device_)
    model.eval()
    with torch.no_grad():
        data = probing_data[idx]
        print("問題:","".join(probing_data.examples[idx].passage_tokens))
        print("答え:",probing_data.examples[idx].answer_texts[0].replace(" ",""))
        data = tuple(t.to(device_) for t in data)
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,label = data
        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        label = min_max(label,x_min,x_max)
        x_len = torch.sum(input_mask,dim=1)
        components_ratio = np.arange(min_boundary*1.5,max_boundary*1.5,0.2)
        result_manipulate_components=[]
        for magnitude in components_ratio:
            kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len,"magnitude":magnitude}
            result = model(**kwargs)
    #         print(f"主成分を{magnitude:.2f}倍:{norm_to_original(result,x_max=x_max,x_min=x_min )[0][0]}")
            result_manipulate_components.append(norm_to_original(result,x_max=x_max,x_min=x_min )[0][0].cpu().detach().numpy())
    return components_ratio, result_manipulate_components
def plot_result(components_ratio,result_manipulate_components,probing_data,
    numbers, min_boundary,max_boundary ,args):
    plt.figure(figsize=(15,7))
    plt.plot(components_ratio,result_manipulate_components,label="result changes",marker=".")
    ans = int(probing_data.examples[args.idx].answer_texts[0].replace(" ",""))
    plt.plot(1,ans,marker=".",label="正解",markersize=20)
    plt.vlines(x=max_boundary,ymax = max(numbers),ymin = min(numbers),color="m",linestyle=":",label="max component")
    plt.vlines(x=min_boundary,ymax = max(numbers),ymin = min(numbers),color="m",linestyle=":",label="min component")
    plt.grid()
    plt.title(f"{args.eliminate_layer}層目の第{args.eliminate_k}主成分の値を操作した時の推論結果の変化\n {''.join(probing_data.examples[args.idx].passage_tokens)}")
    plt.xlabel("倍率",fontsize=15)
    plt.ylabel("推論結果",fontsize=15)
    plt.ylim(min(numbers),max(numbers))
    plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_n_features_dir",
                        default='data/examples_n_features/',
                        type=str,
                        help="Dir containing drop examples and features.入力タイプは一つだと仮定")
    parser.add_argument("--model_dir",
                    type=str)
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
    parser.add_argument("--idx",
                        default=1,
                        type=int,
                        help="主成分値を変化させるインスタンス")                           
    args = parser.parse_args()
    main(args)