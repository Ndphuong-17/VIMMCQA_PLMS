import torch.nn as nn
import torch
try:
    from EmbbeddingTransformer import  ParagraphTransformer #, DocumentTransformer
except:
    try:
        from src.EmbbeddingTransformer import  ParagraphTransformer
    except:
        from VIMMCQA_PLMS.src.EmbbeddingTransformer import  ParagraphTransformer
import re
from string import punctuation



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FeatureOutput(torch.nn.Module):
    def __init__(self):
        super(FeatureOutput, self).__init__()
        self.LayerNorm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True).to(device)
        self.dropout = nn.Dropout(p=0.1, inplace=False).to(device)
        
    def new_forward(self, context_features, # torch.Size([768])
                    question_features):     # torch.Size([768])
        """
        Combine context and question features and calculate their mean.

        Args:
        - context_features: Tensor of size [n, 768] representing context features
        - question_features: Tensor of size [n, 768] representing question features

        Returns:
        - combined_features: Tensor of size [768] representing combined and averaged features
        """
        # Assuming tensors context_features and question_features are located on different devices
        # Move them to the same device
        context_features = context_features.to(device)
        question_features = question_features.to(device)

        # Now, perform the LayerNorm operation
        features = torch.cat([context_features.unsqueeze(0), question_features.unsqueeze(0)], dim=0).to(device)

        # cat 2 feature
        #features = torch.cat([context_features.unsqueeze(0), question_features.unsqueeze(0)], dim =0)
        try:
            features = self.LayerNorm(features.to(device))
        except:
            features = self.LayerNorm(features.to('cpu'))
        #features = self.LayerNorm(features)
        features = self.dropout(features)
        features = torch.mean(features, dim =0)
        
        return features.to(device)

from sentence_transformers import SentenceTransformer, models
import torch.nn as nn
import torch

class Pooler(torch.nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(in_features=768, out_features=768, bias=True)
        self.activation = nn.Tanh()
        
    def new_forward(self, features):     # torch.Size([n, 768])
        """
        Apply pooling operation on the input features.

        Args:
        - features: Tensor of size [n, 768] representing input features

        Returns:
        - pooled_features: Tensor of size [n, 768] representing pooled features
        """
        try:
            features = self.dense(features.to(device))
        except:
            features = self.dense(features.to('cpu'))
            
        features = self.activation(features)
        
        return features.to(device)    


class mcqa_Clasification(torch.nn.Module):
    """
    Multi-Choice Question Answering (MCQA) Classification model.
    
    Args:
    - model_args (str): Arguments for the embedding model.
    - device (torch.device): Device to run the model on.
    
    Output:
    - predicted_label (Tensor): Predicted labels.
    - true_label (Tensor): True labels.
    """
    def __init__(self, model_args: str):
        super(mcqa_Clasification, self).__init__()
        self.embedding = ParagraphTransformer(model_args)
        self.FeatureOutput = FeatureOutput()
        self.pooler = Pooler()
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(in_features=768, out_features=4, bias=True)
        print("Initializing mcqa-classification model completely.")
    
    # @property
    # def device(self):
    #     return self.device
    
    def forward(self, **data_dict):
        """
        Forward pass method for the BiClassification model.

        Args:
        - data_dict (dict): A dictionary containing 'contexts', 'ques_opt', and 'labels'.

        Returns:
        - predicted_label (Tensor): Predicted labels.
        - true_label (Tensor): True labels.
        """
        contexts_vector, ques_opt_vector, labels = data_dict.values()
        predicted_label = self.new_forward(contexts_vector = contexts_vector, 
                                           question_options_vector = ques_opt_vector)
        #predicted_label = int(predicted_label[0])
        
        # Convert predicted_label to tensor and move to device
        #predicted_label = torch.tensor(predicted_label, device=device)

        # Convert true labels to tensor and move to device
        #true_label = torch.tensor(label, device=device)

        
        # Ensure the predicted labels are in the format of [1, 1, 0, 1]
        predicted_label = torch.argmax(predicted_label, dim=1).to(device)

        return {
            'predicted_label': predicted_label.float(),
            'label': torch.tensor(labels, device=device).float()
        }

        # return {
        #     'predicted_label': predicted_label.float(),
        #     'label': labels.float()
        # }
    

    def new_forward(self, contexts = None, question_options = None, contexts_vector = None,  question_options_vector = None):
        """
        Forward pass method for processing contexts and question options.
        
        Args:
        - contexts (list): List of context sentences.
        - question_options (list): List of question options.
        - contexts_vector (Tensor): Pre-calculated embedding vectors for contexts.
        - question_options_vector (Tensor): Pre-calculated embedding vectors for question options.
        
        Returns:
        - predicted_labels (Tensor): Predicted labels.
        """
        contexts_vector = contexts_vector if contexts_vector is not None\
        else self.embedding_corpus(contexts)
        
        question_options_vector = question_options_vector if question_options_vector is not None\
        else self.embedding_corpus(question_options)
        
        features = self.FeatureOutput.new_forward(contexts_vector, question_options_vector)
        features = self.pooler.new_forward(features)
        try: 
            predicted_labels = self.classifier(features.to(device))
        except:
            predicted_labels = self.classifier(features.to('cpu')).to(device)
            
        #predicted_labels = torch.argmax(predicted_labels, dim=1).to(device)
        return predicted_labels
    

    
    def embedding_corpus(self, paragraphs: list[str] = None):
        
        processed_datas = [self.preprocessing_para(para) for para in paragraphs]
            
        # embedding
        # output = torch.Size([n, 768])
        embedding_vector = torch.cat([self.embedding.new_forward(para).view(1, -1) for para in processed_datas], 
                                     dim = 0) 
        return embedding_vector
        
    
    def preprocessing_para(self, paragraph):
        paragraph = paragraph.split('.')
        paragraph = sum([para.split(';') for para in paragraph], [])
        texts = [self.preprocessing_data(text) for text in paragraph]
        texts = [re.sub('\s+', ' ', t.strip()) for t in texts]
        texts = [t for t in texts if t != ""]

        return texts
        
    def preprocessing_data(self, sample):
        # Removing all punctuation
        punct = set(punctuation) - {'_'}
        pattern = "[" + '\\'.join(punct) + "]"
        sample = re.sub(pattern, "", sample)

        # If the sample becomes empty after removing punctuation, return it as is
        if not sample.strip():
            return sample

        # Normalize whitespace
        sample = re.sub(r"\s+", " ", sample)

        return sample.strip().lower()        



class scqa_Clasification(torch.nn.Module):
    """
    Multi-Choice Question Answering (MCQA) Classification model.
    
    Args:
    - model_args (str): Arguments for the embedding model.
    - device (torch.device): Device to run the model on.
    
    Output:
    - predicted_label (Tensor): Predicted labels.
    - true_label (Tensor): True labels.
    """
    def __init__(self, model_args: str):
        super(scqa_Clasification, self).__init__()
        self.embedding = ParagraphTransformer(model_args)
        self.FeatureOutput = FeatureOutput()
        self.pooler = Pooler()
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(in_features=768, out_features=2, bias=True)
        print("Initializing mcqa-classification model completely.")
         
    def forward(self, **data_dict):
        """
        Forward pass method for the BiClassification model.

        Args:
        - data_dict (dict): A dictionary containing 'contexts', 'ques_opt', and 'labels'.

        Returns:
        - predicted_label (Tensor): Predicted labels.
        - true_label (Tensor): True labels.
        """
        contexts_vector, ques_opt_vector, labels = data_dict.values()
        predicted_label = self.new_forward(contexts_vector = contexts_vector, 
                                           question_options_vector = ques_opt_vector)
        #predicted_label = int(predicted_label[0])
        
        # Convert predicted_label to tensor and move to device
        #predicted_label = torch.tensor(predicted_label, device=device)

        # Convert true labels to tensor and move to device
        #true_label = torch.tensor(label, device=device)

        return {
            'predicted_label': predicted_label.float(),
            'label': labels.float()
        }
    def forward_1(self, **data_dict):
        """
        Forward pass method for the MCQA Classification model.
        
        Args:
        - data_dict (dict): A dictionary containing 'contexts', 'ques_opt', and 'labels'.
        
        Returns:
        - predicted_label (Tensor): Predicted labels.
        - true_label (Tensor): True labels.
        """
        contexts, ques_opts, labels = data_dict.values()
        predicted_label = self.new_forward(contexts = contexts, 
                                           question_options = ques_opts)
        predicted_label = [int(x) for x in predicted_label.tolist()]
        
        # Convert predicted_label to tensor and move to device
        predicted_label = torch.tensor(predicted_label, device=device).float()

        # Convert true labels to tensor and move to device
        true_label = torch.tensor(labels, device=device).float()

        return {
            'predicted_label': predicted_label,
            'label': true_label
        }
        
    def new_forward(self, contexts = None, question_options = None, contexts_vector = None,  question_options_vector = None):
        """
        Forward pass method for processing contexts and question options.
        
        Args:
        - contexts (list): List of context sentences.
        - question_options (list): List of question options.
        - contexts_vector (Tensor): Pre-calculated embedding vectors for contexts.
        - question_options_vector (Tensor): Pre-calculated embedding vectors for question options.
        
        Returns:
        - predicted_labels (Tensor): Predicted labels.
        """
        contexts_vector = contexts_vector if contexts_vector is not None\
        else self.embedding_corpus(contexts)
        
        question_options_vector = question_options_vector if question_options_vector is not None\
        else self.embedding_corpus(question_options)
        
        features = self.FeatureOutput.new_forward(contexts_vector, question_options_vector)
        features = self.pooler.new_forward(features)
        try: 
            predictions = self.classifier(features.to(device))
        except:
            predictions = self.classifier(features.to('cpu')).to(device)
            
        predicted_labels = torch.argmax(predictions, dim=1).to(device)
        return predicted_labels
    
    def embedding_corpus(self, paragraphs: list[str] = None):
        
        processed_datas = [self.preprocessing_para(para) for para in paragraphs]
            
        # embedding
        # output = torch.Size([n, 768])
        embedding_vector = torch.cat([self.embedding.new_forward(para).view(1, -1) for para in processed_datas], 
                                     dim = 0) 
        return embedding_vector
        
    
    def preprocessing_para(self, paragraph):
        paragraph = paragraph.split('.')
        paragraph = sum([para.split(';') for para in paragraph], [])
        texts = [self.preprocessing_data(text) for text in paragraph]
        texts = [re.sub('\s+', ' ', t.strip()) for t in texts]
        texts = [t for t in texts if t != ""]

        return texts
        
    def preprocessing_data(self, sample):
        # Removing all punctuation
        punct = set(punctuation) - {'_'}
        pattern = "[" + '\\'.join(punct) + "]"
        sample = re.sub(pattern, "", sample)

        # If the sample becomes empty after removing punctuation, return it as is
        if not sample.strip():
            return sample

        # Normalize whitespace
        sample = re.sub(r"\s+", " ", sample)

        return sample.strip().lower()        