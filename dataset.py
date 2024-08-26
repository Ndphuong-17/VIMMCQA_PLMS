
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import re

class CustomVIMMCQADataset(Dataset):
    def __init__(self, csv_path, tokenizer, label_num=True):
        # Load dataset
        self.data = pd.read_csv(csv_path, encoding='utf-8')
        # Apply preprocessing
        self.data = self.data.apply(lambda col: col.apply(lambda x: '. '.join(self.preprocessing_para(str(x)))))
        self.tokenizer = tokenizer

        if label_num:
            self.data['result'] = self.data['result'].apply(
                lambda result_list: [1 if option in result_list else 0 for option in ['A', 'B', 'C', 'D']]
            )

        # Check lengths
        print(f'Full dataset length: {len(self.data)}')

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)


    def __len__(self, idx = None):
        if idx is None:
            return len(self.data)


        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data.iloc[idx]
        return len(row)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        options = []
        options.append(row['A']), options.append(row['B']), options.append(row['C']), options.append(row['D'])
        sample = {
            'context': row['context'],
            'question': row['question'],
            'options' : options,
            'labels': row['result']
        }

        return sample

    def preprocessing_para(self, paragraph):
        paragraph = paragraph.split('.')
        texts = sum([para.split(';') for para in paragraph], [])
        texts = [re.sub('\s+', ' ', t.strip()) for t in texts]
        texts = [t for t in texts if t.strip() != "" and t.strip() != 'nan']

        return texts

    def collate_fn(self, batch):
      list_context, list_question, list_option = [], [], []
      list_tokenized_context, list_tokenized_input, list_label = [], [], []

      for item in batch:
        context = item['context'].split('. ')
        question = item['question']

        for option, label in zip(item['options'], item['labels']):
          tokenized_context = self.tokenizer(context, return_tensors = 'pt', padding = True, truncation = True, max_length = 256)

          list_tokenized_context.append(tokenized_context), list_context.append(item['context'])
          list_question.append(question), list_option.append(option), list_label.append(label)
        
      tokenized_input = self.tokenizer(list_question, list_option, return_tensors = 'pt', padding = True, truncation = True, max_length = 256)

      res = {
            'raw_context' : list_context,
            'raw_question' : list_question,
            'raw_option' : list_option,
            'tokenized_context' : list_tokenized_context,
            'tokenized_input' : tokenized_input,
            'label' : torch.tensor(list_label),
      }

      return res
