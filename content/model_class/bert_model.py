
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt



class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)

        self.fc1 = torch.nn.Linear(768, 320)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)


        self.fc2 = torch.nn.Linear(320, 160)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)

        self.fc3 = torch.nn.Linear(160, 18)


    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        x = self.dropout(output_1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

class CustomDataset(Dataset):
    
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.title
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            # padding= "max_length",
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
        
class BertModel:
    def __init__(self, weight_path= 'weight\model-fine-tune2.pth', max_len = 7):
        self.weight_path = weight_path
        self.max_len= max_len
        self.model = BERTClass()
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')),strict=False)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def predict(self, data: pd.DataFrame):
        '''
        data: dataframe'''
        def normalize_title(title: str):
                #remove chars after '('
            idx = title.find('(')
            title = title[:idx].strip()
            
            # reposition word
            parts = title.split(',')
            if len(parts) > 1:
                # Move the word after ',' to the start of the title
                title = parts[1].strip() + ' ' + parts[0].strip()
            return title
        
        data.title = data.title.apply(normalize_title)
        dataset = CustomDataset(data, tokenizer= self.tokenizer, max_len= self.max_len)
        
        testing_loader = DataLoader(dataset, batch_size= 1, shuffle= False)  
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in testing_loader:
                ids = batch['ids']
                mask = batch['mask']
                token_type_ids = batch['token_type_ids']

                outputs = self.model(ids, mask, token_type_ids)
                predictions.extend(torch.sigmoid(outputs).cpu().numpy())
        return predictions
import os
def main():
    text = ['The Great Muppet Caper', 'Doctor Zhivago', 'Frankenstein Meets the Wolf Man']
    data = pd.DataFrame({'title': text})
    # test = pd.read_csv('origin_data\movies_test.dat', engine='python',
    #                      sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False)
    weight_path = 'content\weight\model-fine-tune2.pth'
    model = BertModel(weight_path, max_len= 7)
    # if os.path.exists(weight_path):
    #     print('afasdf')
    res = model.predict(data)
    print(np.array(res).shape)
    
if __name__ == '__main__':
    main()
        
            
    
