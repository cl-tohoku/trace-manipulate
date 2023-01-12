from boto3.resources.model import Parameter
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertModel, BertConfig
import numpy as np
import os
from sklearn.metrics import r2_score

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.utils.dummy_pt_objects import LayoutLMForSequenceClassification
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib

from create_examples_n_features_with_type import DropExample, DropFeatures, read_file, write_file, split_digits
from finetune_on_drop_me import DropDataset
# import finetune_on_drop
#import create_examples_n_features

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--examples_n_features_dir",
                    default='data/examples_n_features/',
                    type=str,
                    help="Dir containing drop examples and features.")
parser.add_argument("--eval_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--train_batch_size",
                    default=64,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--num_train_samples",
                    default=-1,
                    type=int,
                    help="Total number of training samples used.")
parser.add_argument("--num_train_epoch",
                    default=10,
                    type=int,
                    help="Total number of training epoch.") 
parser.add_argument("--log_dir",
                    default="log_of_train",
                    type=str,
                    help="directory of tensorboard log.")  #いらない
parser.add_argument("--model_dir",
                    type=str,
                    help = "model directory")  
parser.add_argument("--model_dir_pt",
                    type=str,
                    default="",
                    help="チェックポイントとその他を分けて保存する場合のチェックポイント保存ディレクトリの場所。指定しなかったら--model_dirが使用される。")
parser.add_argument("--architecture",
                    default = "poor_bert",
                    type=str,
                    help = "using model architecture. ['bert','rnn','cnn','mlp','bert_cls','transformer','poor_bert'] ")                               
parser.add_argument("--num_layers",
                    default=6,
                    type=int,
                    help="num layers of poor bert.") 
parser.add_argument("--lr",
                    default=5e-5,
                    type=float,
                    help="learning rate.") 
parser.add_argument("--cls",help="poor bertを使う時clsから回帰を行うか", action="store_true")
parser.add_argument("--myloss",help="ロスに二乗誤差/正解の大きさを使うか", action="store_true")
parser.add_argument("--L1",help="ロスにL1ロスを使うか", action="store_true")
parser.add_argument("--ve",help="数値に対するvalue embeddingを行うか", action="store_true")
parser.add_argument("--freeze",help="word embedding層の重みを固定する", action="store_true")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
os.makedirs("./models/"+args.model_dir,exist_ok=True)
if len(args.model_dir_pt):
    model_dst = f"{args.model_dir_pt}/{args.model_dir}"
    os.makedirs(f"{model_dst}",exist_ok=True)
else:
    model_dst = args.model_dir
print(f"モデルファイルを{model_dst}に保存する。")
# os.makedirs("/work01/yuta_m/arithmetic_probing/models/"+args.model_dir,exist_ok=True)
def min_max(x, x_min,x_max):#正規化
    result = (x-x_min)/(x_max-x_min)
    return result

def norm_to_original(x_norm, x_min, x_max): #正規化を元に戻す
    return x_norm*(x_max-x_min)+x_min

def all_data_answer(train_data,eval_data):
    numbers = []
    for example in train_data:
        numbers.append(example[-1][0])
    for example in eval_data:
        numbers.append(example[-1][0])
    numbers = np.array(numbers)
    return numbers

# def create_dataset(args):
train_data = DropDataset(args, 'train')
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

eval_data = DropDataset(args, 'eval')
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

numbers = all_data_answer(train_data,eval_data)
x_min = min(numbers)
x_max = max(numbers)
print(f"x_min:{x_min}, x_max:{x_max}")
with open("./models/"+args.model_dir+"/min_max.txt","w") as fi:
    print(x_min,file=fi)
    print(x_max,file=fi)
    
def measure_loss(model,data_loader):
    running_loss=0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            #print(label)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,label= batch
            label = min_max(label,x_min,x_max)
            x_len = torch.LongTensor([list(s).count(1) for s in input_mask])
            kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len}
            result = model(**kwargs)
            loss = criterion(result, label)
            running_loss += loss.item()
    return running_loss/len(data_loader)
class BERT_n_layer(nn.Module):
    def __init__(self, drop_rate=0.4, output_size=1,num_hidden_layer=12):
        super().__init__()
        #config = BertConfig(num_hidden_layers=2, num_attention_heads=12)
        self.poor_bert = BertModel.from_pretrained('bert-base-uncased')
        for i in range(12-num_hidden_layer):
            #self.poor_bert.encoder.layer = self.poor_bert.encoder.layer[:num_hidden_layer]
            del(self.poor_bert.encoder.layer[11-i])
            print(f"layer[{11-i}] is removed")
        self.poor_bert.config.num_hidden_layers = num_hidden_layer
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定

    def forward(self,**kwargs):
        ids,mask = kwargs["input_ids"],kwargs["input_mask"]        
        output = self.poor_bert(ids, attention_mask=mask)
        y = output["pooler_output"]
        y = self.fc(self.drop(y))
        return y
class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate=0.4, output_size=1,subset=50):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定
        self.subset=subset
        
    def forward(self,**kwargs):
        ids,mask = kwargs["input_ids"],kwargs["input_mask"]        
        output = self.bert(ids, attention_mask=mask)
        y = output["last_hidden_state"]
        #print(y.shape)
        y = y[:,:self.subset]
        y = torch.mean(y,1)
        self.last_hidden_state=y
        #print(y.shape)
        y = self.fc(self.drop(y))
        return y
class BERTClass_cls(torch.nn.Module):
    def __init__(self, drop_rate=0.4, output_size=1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定

    def forward(self,**kwargs):
        ids,mask = kwargs["input_ids"],kwargs["input_mask"]
        
        output = self.bert(ids, attention_mask=mask)
        y = output["pooler_output"]
        #print(y.shape)
        y = self.fc(self.drop(y))
        return y
class RnnModel(nn.Module):
    """
    単語埋め込み → RNN → 線形層 のモデル。
    forward では，単語のID列とその入力長を受け取り，ロジットを返す。
    """
    def __init__(self, vocab_size, emb_dim=300, hidden_dim=50, output_size=1):   
        super().__init__()
        self.emb = nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, **kwargs):
        """
        Args:
            x: padded ID tensor (batch_size, max_seq_length)
            x_len: tensor (batch_size)
        Return:
            logit: tensor (batch_size)
        """
        x = kwargs["input_ids"]
        x_len = kwargs["x_len"]
        embedding = self.emb(x)
        packed_embedding = pack_padded_sequence(embedding, x_len, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed_embedding)
        logit = self.fc(h_n[-1])
        return logit
class MLP(nn.Module):
    def __init__(self,vocab_size,embedding_size=300,output_size=1):
        super(MLP, self).__init__()
        self.emb = nn.Embedding(vocab_size+1, embedding_size, padding_idx=0)
        self.fc1 = nn.Linear(embedding_size, 512)   
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_size)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        
    def forward(self, **kwargs):
        x = kwargs["input_ids"]
        x = self.emb(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.mean(F.relu(self.fc3(x)),1)
class CNN(nn.Module):
    def __init__(self,vocab_size, hidden_size=50, embedding_size=300, output_size=1,  kernel_size=3):
        super().__init__()
        self.hidden_size = hidden_size  # 問題文のd_hz
        self.kernel_size=kernel_size
        self.embedding_size = embedding_size  # 単語埋め込みの次元数
        self.emb = nn.Embedding(vocab_size+1, embedding_size, padding_idx=0)
        self.cnn = nn.Conv1d(embedding_size, hidden_size, kernel_size, stride=1, padding=1)  
        # nput_channel, output_channel, kernel_size, stride,padding
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, **kwargs):
        x = kwargs["input_ids"]
        emb = self.emb(x)
        # emb.size() = (batch_size, seq_len, embbeding_size)
        conv = self.cnn(emb.transpose(-1, -2)) 
        act = F.relu(conv)
        max_pool = nn.MaxPool1d(kernel_size= act.size()[-1])(act)
        #最大値プーリング
        # max_pool.size() = (batch_size, hidden_size, 1)
        #print("プーリング",max_pool.size())
        #print(max_pool)
        #print(torch.squeeze(max_pool, -1))
        logit = self.fc(torch.squeeze(max_pool,dim=-1))
        #squeezeすると次元1の部分を凝縮してくれる
        #logit = self.fc(max_pool)
        return logit
class Transformer_2layer(nn.Module):
    def __init__(self, drop_rate=0.4, output_size=1):
        super().__init__()
        config = BertConfig(num_hidden_layers=2, num_attention_heads=12)
        self.transformer = BertModel(config)
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定

    def forward(self,**kwargs):
        ids,mask = kwargs["input_ids"],kwargs["input_mask"]        
        output = self.transformer(ids, attention_mask=mask)
        y = output["last_hidden_state"]
        #print(y.shape)
        y = torch.mean(y,1)
        self.last_hidden_state=y
        #print(y.shape)
        y = self.fc(self.drop(y))
        return y

class LossFunction(nn.Module):

    def __init__(self):
        super(LossFunction, self).__init__()
    def forward(self, preds, targets):
        loss = self.mselossdivlabel(preds, targets)
        return loss
    def mselossdivlabel(self,preds,targets):
        eps=1e-10
        result = (preds - targets) ** 2 /norm_to_original(targets,x_min,x_max)
        #print(result)
        # print("最大:",result.max())
        # print("最小:",result.min())
        #result = (preds - targets) ** 2 
        return result.mean()

def train(model,train_dataloader,eval_dataloader,optimizer, criterion,total_epoch=args.num_train_epoch,):
    #writer = SummaryWriter(log_dir="./logs/"+args.log_dir)
    writer = SummaryWriter(log_dir="./models/"+args.model_dir+"/logs/")
    train_loss=[]
    valid_loss=[]
    min_loss=10000
    model.train()
    for epoch in tqdm(range(total_epoch)):
        running_loss = 0
        model.train()
        #print(device)
        model = model.to(device)
        for i,batch in enumerate(train_dataloader):
            #print(label)
            
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,label= batch
            x_len = torch.sum(input_mask,dim=1)
            label = min_max(label,x_min,x_max)
            kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len}
            result = model(**kwargs)

            #loss = result.mean()
            loss = criterion(result, label)
            #print(loss)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            #print(loss)
            running_loss += loss.item()
    
        train_loss.append(running_loss/len(train_dataloader))
        loss_v = measure_loss( model,eval_dataloader )
        writer.add_scalars("loss",{"train":running_loss/len(train_dataloader),"valid":loss_v},epoch)
        if loss_v < min_loss:
            min_loss = loss_v
            torch.save(model.to('cpu').state_dict(), f"{model_dst}/checkpoint_best.pt")
            # torch.save(model.to('cpu').state_dict(), "/work01/yuta_m/arithmetic_probing/models/"+args.model_dir+"/checkpoint_best.pt")
        valid_loss.append ( loss_v)
        print(f"epoch{epoch}:{loss_v}")
        model_path = f"checkpoint_{epoch+1}.pt"
        torch.save(model.to('cpu').state_dict(), f"{model_dst}/{model_path}")
        # torch.save(model.to('cpu').state_dict(), "./models/"+args.model_dir+"/"+model_path) #ホームディレクトリにモデルファイルをおくな
        # torch.save(model.to('cpu').state_dict(), "/work01/yuta_m/arithmetic_probing/models/"+args.model_dir+"/"+model_path) 
    writer.close()
    return train_loss,valid_loss
def evaluate(model,eval_data):
    preds = []
    answer = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i,data, in tqdm(enumerate(eval_data)):
            data = tuple(t.to(device) for t in data)
            input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans,label = data
            input_ids = torch.unsqueeze(input_ids,0)
            input_mask = torch.unsqueeze(input_mask,0)
            label = min_max(label,x_min,x_max)
            x_len = torch.sum(input_mask,dim=1)
            kwargs = {"input_ids":input_ids,"input_mask":input_mask,"x_len":x_len}
            result = model(**kwargs)
            preds.append(norm_to_original(result,x_min,x_max)[0][0].cpu().detach().numpy())
            answer.append(norm_to_original(label,x_min,x_max).cpu().detach().numpy())
    return r2_score(answer,preds)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
vocab_size=len(tokenizer.vocab)

architecture = {"bert":BERTClass(),"cnn":CNN(vocab_size),\
    "rnn":RnnModel(vocab_size),"mlp":MLP(vocab_size),\
        "bert_cls":BERTClass_cls(),"transformer":Transformer_2layer(),
        "poor_bert":BERT_n_layer(num_hidden_layer=args.num_layers)}

model = architecture[args.architecture]
if args.ve:
    num_ids = [tokenizer.convert_tokens_to_ids([str(digit)])[0] for digit in range(10)]
    sub_num_ids = [tokenizer.convert_tokens_to_ids([f"##{digit}"])[0] for digit in range(10)]
    for i in range(10):
        embs = model.poor_bert.embeddings.word_embeddings._parameters["weight"].detach().numpy()
        embs[num_ids[i]] = i/1000
        # embs = model.poor_bert.embeddings.word_embeddings._parameters["weight"].detach().numpy()
        embs[sub_num_ids[i]] = i/1000
    if args.freeze:
        model.poor_bert.embeddings.word_embeddings._parameters["weight"].requires_grad = False
model.to(device)
summary(model)
if args.architecture in ["bert","transformer",'bert_cls','poor_bert']:
    optimizer = BertAdam(model.parameters(), lr = args.lr)
else:
    optimizer = optim.Adam(model.parameters(),lr = args.lr)

print(optimizer)
criterion = nn.MSELoss()
if args.L1:
    criterion = nn.L1Loss()
if args.myloss:
    criterion = LossFunction()
train_loss,valid_loss = train(model,train_dataloader,eval_dataloader,optimizer,criterion)
#print(train_loss)
#print(valid_loss)
epochs = [i+1 for i in range(args.num_train_epoch) ]
plt.plot(epochs,train_loss,label="train")
plt.plot(epochs,valid_loss,label="valid")
plt.grid()
plt.legend()
plt.savefig("./models/"+args.model_dir+"/loss")
R2_score = evaluate(model,eval_data)
print(f"r2_score:{R2_score}")
