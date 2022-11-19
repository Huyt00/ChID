import json
import os
import numpy


from transformers import AutoTokenizer, BertModel
import torch
from tqdm import tqdm, trange

_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(_model_device)

def add_syn_and_exp(dir):
    sys = {}
    exp = {}
    with open('./ChID/idiom_synonyms.txt', 'r') as file:
        line = file.readline()
        while True:
            line = file.readline()
            if not line:
                break
            line = line.split('\t')
            if line[0] in sys:
                sys[line[0]].append(line[1])
            else:
                sys[line[0]] = [line[1]]
                
    with open('./ChID/idiom_exp.txt', 'r') as file:
        line = file.readline()
        while True:
            line = file.readline()
            if not line or line == '"':
                break
            line = line.strip('\" \n').split('\t')
            exp[line[0]] = line[-1]

    
    data_set = []
    with open('./ChID/'+dir, 'r', encoding='utf-8') as file:
        line = file.readline()
        while True:
            line = file.readline()
            if not line:
                break
            line = json.loads(line.strip())
            line['synonyms'] = []
            line['explaination'] = []
            
            idiom = line['groundTruth']
            for i in idiom:
                if i in sys:
                    line['synonyms'].append(sys[i])
                else:
                    line['synonyms'].append([])
                if i in exp:
                    line['explaination'].append(exp[i])
                else:
                    line['explaination'].append("None")
            data_set.append(line)
                    
    # with open('./ChID_new/'+dir, 'w', encoding='utf-8') as file:
    #     for line in data_set:
    #         file.write(json.dumps(line, ensure_ascii=False)+'\n')
    return data_set

def bert(tmp):
    inputs = tokenizer(tmp, return_tensors='pt',padding=True, truncation=True, max_length=512).to(_model_device)
    outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state
    # print('last_hidden_states:' ,last_hidden_states)
    pooler_output = outputs.pooler_output
    # print('---pooler_output: ', pooler_output)
    return pooler_output



# 得到释义的经过bert的cls
# tmp是输入bert前的explanation
# bedding是bert后explanation的cls，以列表保存
# groudtruth里的词语——explanation的cls 两两按字典 保存于bedding文件

def add_exp_cls(dir):
    # exp = {}
    # dirs = os.listdir('./ChID')
    # for dir in dirs:          
    #     if '.json' not in dir:
    #         continue
        
    data_set = add_syn_and_exp(dir)
    explaination =[[]]
    embedding = []
    for json_data in data_set:
        explaination[-1] += json_data['explaination']
        if len(explaination[-1]) >= 64:
            explaination.append([])
        
    for exp in tqdm(explaination):
        embedding += bert(exp).cpu().tolist()
        
    index = 0
    for json_data in data_set:
        json_data['exp embedding'] = []
        for exp in json_data['explaination']:
            # for i in range(len(zipped)):
            #     key = zipped[i][0]       # 取groundtruth
            #     tmp = zipped[i][-1]      # 取groundtruth对应的explanation
            if exp != 'None':           # 判断该groundtruth对应的explanation不为空（explanation部分为空）; 否则赋值为None              
                json_data['exp embedding'].append(embedding[index]) # Tensor转化为list
            else:
                json_data['exp embedding'].append([0] * 768)
            index += 1
    
    assert index == len(embedding)  
    return data_set  

def process():
    dirs = os.listdir('./ChID')
    for dir in dirs:          
        if '.json' not in dir:
            continue
        print(dir)
        data_set = add_exp_cls(dir)
        print("start writing "+dir)
        with open('./ChID_new/'+dir, 'w', encoding='utf-8') as file:
            for line in data_set:
                file.write(json.dumps(line, ensure_ascii=False)+'\n')
        
process()


                
