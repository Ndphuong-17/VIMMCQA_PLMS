from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch
import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class EmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super(EmbeddingModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def forward(self, texts, max_length=None):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        with torch.no_grad():  # Ensure no gradients are computed for this step
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

class ParagraphTransformer(nn.Module):
    def __init__(self, model_args):
        super(ParagraphTransformer, self).__init__()
        self.device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = EmbeddingModel(model_args.model_name_or_path).to(self.device1)
        
        self.embedding_dim = 768  # Assuming 768 as the embedding dimension
        self.linear = nn.Linear(self.embedding_dim, 768).to(self.device1)
        self.LayerNorm = nn.LayerNorm(768).to(self.device1)
        self.dropout = nn.Dropout(p=0.1).to(self.device1)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1).to(self.device1)

    def new_forward(self, features_list):
        # Flatten the list of lists into a single list of sentences
        flattened_features = [sentence for sublist in features_list for sentence in sublist]

        # Get embeddings for all sentences at once
        embeddings = self.embedding(flattened_features)
        embeddings = embeddings.to(self.device1)
        embeddings.requires_grad_()

        # Transpose and apply average pooling
        embeddings = embeddings.transpose(1, 2)  # Shape: [total_sentences, hidden_size, sequence_length]
        pooled_output = self.avg_pooling(embeddings)  # Shape: [total_sentences, hidden_size, 1]
        pooled_output = pooled_output.squeeze(dim=-1)  # Shape: [total_sentences, hidden_size]

        # Apply linear transformation, normalization, and dropout
        out = self.linear(pooled_output)  # Shape: [total_sentences, 768]
        out = self.LayerNorm(out)  # Shape: [total_sentences, 768]
        out = self.dropout(out)  # Shape: [total_sentences, 768]

        # Now reshape and reduce to [n, 768] by averaging each sublist
        sentence_counts = [len(sublist) for sublist in features_list]  # Number of sentences in each sublist
        reshaped_out = torch.split(out, sentence_counts)  # Split based on the sentence counts
        final_output = torch.stack([sublist.mean(dim=0) for sublist in reshaped_out])  # Shape: [n, 768]

        return final_output.to(self.device1)





# class DocumentTransformer(SentenceTransformer):
#     def __init__(self, ParaEmbedding = None):

#         super(DocumentTransformer, self).__init__()
        

#         self.ParaEmbedding = ParaEmbedding 
#     def new_forward(self, document: list[list[str]], passError = False):
#         tensors = []
#         for paragraph in tqdm.tqdm(document, desc= "Embedding Document: "):
#             embedding = self.ParaEmbedding.new_forward(paragraph, notice= False, passError= passError)
#             if embedding is not None:
#                 tensors.append(embedding)

#         # Concatenate them along the first dimension
#         merged_tensor = torch.cat(tensors, dim=0)

#         # Verify the new shape
#         print("Shape of document_embeddings: ", merged_tensor.shape)
        
#         return merged_tensor