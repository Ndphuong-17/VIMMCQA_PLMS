import torch
import torch.nn as nn
from src.Reader import mcqa_Clasification
from src.preprocessing import preprocessing_para
from src.EmbbeddingTransformer import  ParagraphTransformer
from src.Retriever import Retrieval 
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import torch
import os
import re
from string import punctuation


device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VIMMCQA(torch.nn.Module):
    """
    VIMMCQA (Visual and Interactive Multi-Modal Question Answering) model.

    Args:
        - model_args: Arguments for the embedding model.
    Output:
        - predicted_label: Predicted labels.
        - label: True labels.
        - evidence: Relevant evidence.
    """

    def __init__(self, model_args):
        super(VIMMCQA, self).__init__()
        self.mcqa = mcqa_Clasification(model_args = model_args)
        self.threshold = model_args.threshold if model_args.threshold else 0.5

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {save_directory}")


    @classmethod
    def from_pretrained(cls, load_directory, model_args):
        model = cls(model_args=model_args)  # Create a new instance of the model
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Model loaded from {load_directory}")
        return model
    
    def forward(self, **data_dict):
        """
        Forward pass method for the VIMMCQA model.
        Args:
            {
                'vector_context': vector_context,
                'vector_ques_opt': vector_ques_opt,
                'tensor_label' : tensor_label
            }

        Returns:
            - predicted_label: Predicted labels.
            - label: True labels.
        """
        #print(data_dict)
        # print("--- VIMMCQA ---")

        contexts_embedding, ques_opt_embedding, tensor_label = data_dict.values()

        logits = self.mcqa.new_forward(contexts_vector = contexts_embedding,
                                question_options_vector = ques_opt_embedding)
        # print(f"logits: {logits.shape}, {logits.requires_grad}") # should be return [n, num_options] and True
        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > self.threshold).float()

        # print(f"predicted_labels: {predicted_labels.shape}, {predicted_labels.requires_grad}") # should be return [n, num_options] and True
        # print(f"logits, required gradients: {logits.requires_grad}")

        # Compute the loss using BCEWithLogitsLoss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, tensor_label)

        # print(f"predicted_label: {predicted_labels.shape}, {predicted_labels.requires_grad} \nlabels: {tensor_label.shape}, {tensor_label.requires_grad}")

        print(f"predicted_labels: {predicted_labels} \n probabilities: {probabilities} \nlabels: {tensor_label}")


        return {
            'predicted_label': predicted_labels,
            'label': tensor_label,
            'loss': loss
        }
    


class DataCollator:
    def __init__(self,
                 model_args,   
                 corpus = None
                 ):
        super(DataCollator, self).__init__()
        self.task = model_args.task
        self.model_args = model_args  

        # Initialize only what is needed for the task
        if self.task == 'VIMMCQA':
            self.embedding = ParagraphTransformer(model_args)
        elif self.task == 'full_VIMMCQA':
            self.retrieval = Retrieval(model_args=model_args, corpus=corpus)
    
    def initialize_retrieval(self):
        """Initialize retrieval only when needed."""
        
        # Load segmented corpus
        with open(self.model_args.old_wseg_corpus_file, 'r', encoding='utf-8') as _file:
            datas = _file.read()
        wseg_datas = datas.split('\n')
        print(len(wseg_datas))
        print("Initializing corpus wseg_datas completely.")

        self.retrieval = Retrieval(model_args=self.model_args, corpus=wseg_datas)

    def __call__(self, raw_batch_dict):
        if self.task == 'full_VIMMCQA':
            raise Exception('This task can run now, will be update soon')
            return self.full_VIMMCQA(raw_batch_dict)
        
        elif self.task == 'VIMMCQA':
            try:
                # print("Containing context in dataset")
                return self.VIMMCQA(raw_batch_dict)
            except KeyError:
                print("Do not include context in the data; automatically switching to full_VIMMCQA to retrieve context for continuous.")
                self.task = 'full_VIMMCQA'
                raise Exception('This task can run now, will be update soon')
                self.initialize_retrieval()
                return self.full_VIMMCQA(raw_batch_dict)
    
    def VIMMCQA(self, raw_batch_dict):
        # print("--- raw_batch_dict --- ", len(raw_batch_dict))
        # print(raw_batch_dict[0].keys())

        ques_options = [preprocessing_para(data['ques_opt']) for data in raw_batch_dict]
        contexts = [preprocessing_para(data['context']) for data in raw_batch_dict]
        labels = [data['label'] for data in raw_batch_dict]
        
        # Convert labels to numeric values
        numeric_labels = [list(map(int, label.strip('[]').split(', '))) for label in labels]
        # Convert to tensor with requires_grad=True
        tensor_label = torch.tensor(numeric_labels, dtype=torch.float32, device=device, requires_grad=True)
        # print(tensor_label.requires_grad) # Should return True

        ques_opt_embedding = self.embedding.new_forward(ques_options)
        contexts_embedding = self.embedding.new_forward(contexts)

        # should return [n, 768] and True
        return {
            'ques_opt_embedding': ques_opt_embedding, 
            'contexts_embedding': contexts_embedding, 
            'tensor_label': tensor_label
        }

    

def compute_metric(logits, tensor_label):
    """
    Compute metrics for multi-label classification.

    Args:
        logits (Tensor): Raw model outputs of shape [batch_size, num_classes].
        tensor_label (Tensor): One-hot encoded ground truth labels of shape [batch_size, num_classes].

    Returns:
        dict: A dictionary containing accuracy, recall, f1, mse, and precision.
    """
    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits).cpu().detach().numpy()
    
    # Convert probabilities to binary predictions
    predictions = (probabilities > 0.5).astype(int)

    # Convert tensors to numpy arrays for scikit-learn metrics
    y_true = tensor_label.cpu().detach().numpy()
    y_pred = predictions

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')  # Use 'macro' for multi-label classification
    f1 = f1_score(y_true, y_pred, average='macro')  # Use 'macro' for multi-label classification
    precision = precision_score(y_true, y_pred, average='macro')  # Use 'macro' for multi-label classification
    mse = ((y_true - y_pred) ** 2).mean()

    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'mse': mse,
    }

    return metrics

def compute_metrics(p):
    logits = p.predictions
    labels = p.label_ids
    
    # Compute metrics
    return compute_metric(torch.tensor(logits, dtype=torch.float), torch.tensor(labels, dtype=torch.float))
