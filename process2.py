
'''
 zip将两列表打包为元组列表 —— 两两一一对应
 
'''

import json
import os
import numpy


from transformers import BertTokenizer, BertModel
import torch
 
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

def bert(tmp):
    inputs = tokenizer(tmp, return_tensors='pt',padding=True,truncation=True)
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

exp = {}
with open('./Chidnew/dev_data.json','r',encoding='utf-8') as f : # 读取需要embedding的文件
    #while True:
    for j in range(0,5):
        line = f.readline()
        if not line:
            break
        json_data = json.loads(line)
        if json_data['explaination']:
            zipped=list(zip(json_data['groundTruth'],json_data['explaination'])) # 打包为zip,两两对应；转化为列表
            for i in range(len(zipped)):
                key = zipped[i][0]       # 取groundtruth
                tmp = zipped[i][-1]      # 取groundtruth对应的explanation
                if len(tmp)>0:           # 判断该groundtruth对应的explanation不为空（explanation部分为空）; 否则赋值为None
                    bedding = bert(tmp)               
                    exp[key] = bedding.tolist() # Tensor转化为list
                else:
                    exp[key] = ['None']            
        else:
            for key in json_data['groundTruth']:  # 若本行explanation整体为空
                exp[key] = ['None']
                


dict_json=json.dumps(exp, ensure_ascii=False) # 转化为json格式文件

#将json文件保存为.json格式文件
with open('./Chidnew/expbedding.json','w+') as file:
    file.write(dict_json)
file.close()


'''
#验证是否全 (不用管)
with open('./Chidnew/expbedding.json','r+') as file:
    line = file.readline()
    ja = json.loads(line)
    print(ja.keys())
    print(len(ja))
    print(ja['独一无二'])
'''
