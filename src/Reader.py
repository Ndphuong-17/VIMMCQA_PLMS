import torch.nn as nn
import torch
from string import punctuation


class FeatureOutput(torch.nn.Module):
    def __init__(self):
        super(FeatureOutput, self).__init__()
        self.device11 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.LayerNorm = nn.LayerNorm(768, eps=1e-05, elementwise_affine=True).to(self.device11)
        self.dropout = nn.Dropout(p=0.15, inplace=False).to(self.device11)

    def new_forward(self, vec1, vec2 = None):
        """
        Combine context and question features and calculate their mean.

        Args:
        - context_features: Tensor of size [n, 1, 768] representing context features
        - question_features: Tensor of size [n, 4, 768] representing question features

        Returns:
        - combined_features: Tensor of size [n, 768] representing combined and averaged features
        """
        # Move tensors to the correct device
        vec1 = vec1.to(self.device11)   # shape [n, 4, 768]
        if vec2 is not None:
            vec2 = vec2.to(self.device11)   # shape [n, 1, 768]

            # Concatenate and normalize features
            features = torch.cat([vec1, vec2], dim=1) # [n, 5, 768]
        else:
            features = vec1.to(self.device11)   
        features = self.LayerNorm(features)
        features = self.dropout(features)
        combined_features = torch.mean(features, dim=1)

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
        self.FeatureOutput = FeatureOutput().to(self.device1)
        self.pooler = Pooler().to(self.device1)
        self.dropout = nn.Dropout(p=0.1).to(self.device1)
        self.classifier = nn.Linear(768, 4).to(self.device1)
        print("Initializing mcqa-classification model completely.")
    
    def forward(self, **data_dict):
        ques_opt_vector, contexts_vector, labels = data_dict.values()
        predicted_label = self.new_forward(ques_opt_vector,contexts_vector)

        return {
            'logits': predicted_label.float(),
            'label': labels
        }

    def new_forward(self, vec1, vec2 = None):

        
        features = self.FeatureOutput.new_forward(vec1, vec2)
        features = self.pooler.new_forward(features)
        features = self.dropout(features)
        predicted_labels = self.classifier(features)
        
        return predicted_labels