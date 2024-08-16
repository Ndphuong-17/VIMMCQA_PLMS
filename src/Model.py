import torch
import torch.nn as nn
from src.Reader import mcqa_Clasification
try:
    from EmbbeddingTransformer import  ParagraphTransformer #, DocumentTransformer
except:
    try:
        from src.EmbbeddingTransformer import  ParagraphTransformer
    except:
        from VIMMCQA_PLMS.src.EmbbeddingTransformer import  ParagraphTransformer
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
        - data_args: Additional data arguments.
        - device: Device to run the model on.
        - corpus: Corpus data.
        - corpus_vector: Pre-calculated corpus vectors.

    Output:
        - predicted_label: Predicted labels.
        - label: True labels.
        - evidence: Relevant evidence.
    """

    def __init__(self,
                 model_args
                #  corpus = None,
                #  corpus_vector = None,
                #  compute_metric = None,
                #  bm25 = None
                 ):
        super(VIMMCQA, self).__init__()
        self.mcqa = mcqa_Clasification(model_args = model_args)
        # self.compute_metric = compute_metric
    
    

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Optionally save any other configurations needed
        # config_path = os.path.join(save_directory, "config.json")
        # with open(config_path, "w") as f:
        #     json.dump(self.config, f)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory, model_args):
        model = cls(model_args=model_args)  # Create a new instance of the model
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Model loaded from {load_directory}")
        return model
    
    def mcqaClassification(self, query_vectors, relevant_vectors):
        """
        Perform Multi-Choice Question Answering classification.

        Args:
            - query_vectors: Vectors of query.
            - relevant_vectors: Vectors of relevant evidence.

        Returns:
            - predicted_labels: Predicted labels.
        """
        predicted_labels = self.mcqa.new_forward(contexts_vector = relevant_vectors,  
                                                 question_options_vector = query_vectors)
        
        # print(predicted_labels)
        # predicted_label = [int(x[0]) if isinstance(x, list) and len(x) > 0 else int(x) for x in predicted_labels.tolist()]


        predicted_label = [[int(x) for x in t] for t in predicted_labels.tolist()]
        
        # Convert predicted_label to tensor and move to device
        predicted_label = torch.tensor(predicted_label, device=device)
        
        return predicted_label
    
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
        print("--- VIMMCQA ---")
        relevant_vectors, query_vectors, tensor_label = data_dict.values()
        

        # Convert tensor_label to float
        tensor_label = torch.tensor(tensor_label, device=device).float()

        print('relevant_vectors: ', relevant_vectors.shape)
        print('vector_ques_opt: ', query_vectors.shape)
        print('tensor_label: ', tensor_label.shape)
        
        
        # predicted_labels = self.mcqaClassification(query_vectors = query_vectors.to(device), 
        #                                            relevant_vectors = relevant_vectors.to(device)).float()

        logits = self.mcqa.new_forward(contexts_vector = relevant_vectors,  
                                                 question_options_vector = query_vectors)
        
        # Ensure logits and labels are on the same device
        logits = logits.to(device)
        tensor_label = tensor_label.to(device)
        
        # Compute the loss using BCEWithLogitsLoss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, tensor_label)

        predicted_labels = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        predicted_labels = (predicted_labels > 0.5).float().to(device)

        print(f"predicted_label: {predicted_labels.shape} labels: {tensor_label.shape}")
    
        return loss, predicted_labels
    

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
            return self.full_VIMMCQA(raw_batch_dict)
        

        elif self.task == 'VIMMCQA':
            
            try:
                raw_batch_dict[0]['context']
                print("Containing context in dataset")
                return self.VIMMCQA(raw_batch_dict)
            
            except KeyError:
                print("Do not include context in the data; automatically switching to full_VIMMCQA to retrieve context for continuous.")
                self.task = 'full_VIMMCQA'
                self.initialize_retrieval()
                return self.full_VIMMCQA(raw_batch_dict)
    
    def full_VIMMCQA(self, raw_batch_dict):
        print("--- raw_batch_dict --- ", len(raw_batch_dict))
        #print(raw_batch_dict)
        print(raw_batch_dict[0].keys())
        ques_options = [data['ques_opt'] for data in raw_batch_dict]
        #print(ques_options)
        #ques_options = sum(ques_options, [])
        labels = [data['label'] for data in raw_batch_dict]
        #ques_options, labels = raw_batch_dict[0].values()
        numeric_labels = [list(map(int, label.strip('[]').split(', '))) for label in labels]

        tensor_label = torch.tensor(numeric_labels).to(device)

        # tensor_label = torch.tensor(labels).to(device).view(-1, 1)
        vector_ques_opt = torch.zeros(0, 768).to(device)
        vector_context = torch.zeros(0, 768).to(device)

        for ques_option in ques_options:
            _result_retrieval = self.retrieval.retrieval(query=ques_option)
#                 print("_result_retrieval[3]: ", _result_retrieval[3].shape)
#                 print("_result_retrieval[2]: ", _result_retrieval[2].shape)
            vector_ques_opt = torch.cat((vector_ques_opt, _result_retrieval[3].unsqueeze(0)), dim=0)
            vector_context = torch.cat((vector_context, _result_retrieval[2]), dim=0)
        
        print(f'vector_ques_opt {vector_ques_opt.shape}')
        print(f'vector_context {vector_context.shape}')
        print(f'tensor_label:   {tensor_label.shape}')
        print("--- data_collator completely ---")
        return {
            'vector_context': vector_context,
            'vector_ques_opt': vector_ques_opt,
            'tensor_label' : tensor_label
        }
    
    
    def VIMMCQA(self, raw_batch_dict):
        print("--- raw_batch_dict --- ", len(raw_batch_dict))
        print(raw_batch_dict[0].keys())
        ques_options = [data['ques_opt'] for data in raw_batch_dict]
        contexts = [data['context'] for data in raw_batch_dict]

        labels = [data['label'] for data in raw_batch_dict]
        numeric_labels = [list(map(int, label.strip('[]').split(', '))) for label in labels]
        tensor_label = torch.tensor(numeric_labels).to(device)

        # tensor_label = torch.tensor(labels).to(device).view(-1, 1)
        vector_ques_opt = torch.zeros(0, 768).to(device)
        vector_context = torch.zeros(0, 768).to(device)

        for ques_option, context in zip(ques_options, contexts):
            ques_option = self.preprocessing_para(ques_option)
            context = self.preprocessing_para(context)

            vector_ques_opt = torch.cat((
                vector_ques_opt, 
                self.embedding.new_forward(ques_option).unsqueeze(0).to(device)
            ), dim=0)
            
            vector_context = torch.cat((
                vector_context, 
                self.embedding.new_forward(context).unsqueeze(0).to(device)
            ), dim=0)
        
        print(f'vector_ques_opt {vector_ques_opt.shape}')
        print(f'vector_context {vector_context.shape}')
        print(f'tensor_label:   {tensor_label.shape}')
        print("--- data_collator completely ---")
        return {
            'vector_context': vector_context,
            'vector_ques_opt': vector_ques_opt,
            'tensor_label' : tensor_label
        }
    
    
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
    
    # Convert to tensors
    logits = torch.tensor(logits, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)
    
    # Compute metrics
    return compute_metrics(logits, labels)
