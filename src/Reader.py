import torch.nn as nn
import torch
from src.EmbbeddingTransformer import  ParagraphTransformer
import re
from string import punctuation


class FeatureOutput(torch.nn.Module):
    def __init__(self):
        super(FeatureOutput, self).__init__()
        self.device11 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.LayerNorm = nn.LayerNorm(768, eps=1e-05, elementwise_affine=True).to(self.device11)
        self.dropout = nn.Dropout(p=0.1, inplace=False).to(self.device11)

    def new_forward(self, context_features, question_features):
        """
        Combine context and question features and calculate their mean.

        Args:
        - context_features: Tensor of size [n, 768] representing context features
        - question_features: Tensor of size [n, 768] representing question features

        Returns:
        - combined_features: Tensor of size [768] representing combined and averaged features
        """
        # Move tensors to the correct device
        context_features = context_features.to(self.device11)
        question_features = question_features.to(self.device11)

        # Concatenate and normalize features
        features = torch.cat([context_features.unsqueeze(0), question_features.unsqueeze(0)], dim=0)
        features = self.LayerNorm(features)
        features = self.dropout(features)
        combined_features = torch.mean(features, dim=0)

        return combined_features
    
class Pooler(torch.nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        self.device11 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dense = nn.Linear(in_features=768, out_features=768, bias=True).to(self.device11)
        self.activation = nn.Tanh()

    def new_forward(self, features):
        """
        Apply pooling operation on the input features.

        Args:
        - features: Tensor of size [n, 768] representing input features

        Returns:
        - pooled_features: Tensor of size [n, 768] representing pooled features
        """
        features = features.to(self.device11)
        features = self.dense(features)
        pooled_features = self.activation(features)

        return pooled_features


class mcqa_Clasification(nn.Module):
    def __init__(self, model_args: str):
        super(mcqa_Clasification, self).__init__()
        self.device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = ParagraphTransformer(model_args).to(self.device1)
        self.FeatureOutput = FeatureOutput().to(self.device1)
        self.pooler = Pooler().to(self.device1)
        self.dropout = nn.Dropout(p=0.1).to(self.device1)
        self.classifier = nn.Linear(768, 4).to(self.device1)
        print("Initializing mcqa-classification model completely.")
    
    def forward(self, **data_dict):
        contexts_vector, ques_opt_vector, labels = data_dict.values()
        predicted_label = self.new_forward(contexts_vector=contexts_vector, 
                                           question_options_vector=ques_opt_vector)
        
        labels = torch.tensor(labels, device=self.device1).float()

        predicted_label = torch.argmax(predicted_label, dim=1).to(self.device1)

        return {
            'predicted_label': predicted_label.float(),
            'label': labels
        }

    def new_forward(self, contexts_vector, question_options_vector):

        # contexts_vector = contexts_vector.to(self.device1)
        # question_options_vector = question_options_vector.to(self.device1)
        
        features = self.FeatureOutput.new_forward(contexts_vector, question_options_vector)
        features = self.pooler.new_forward(features)
        features = self.dropout(features)
        
        predicted_labels = self.classifier(features)
        
        return predicted_labels