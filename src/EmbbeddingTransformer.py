from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch
import tqdm

from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ParagraphTransformer(SentenceTransformer):
    def __init__(self, model_args):

        # Initialize the ParagraphTransformer by inheriting from SentenceTransformer
        # SentenceTransformer is a library for generating sentence embeddings

        # Initialize the SentenceTransformer model for generating sentence embeddings
        # This model will be used to encode individual sentences in a paragraph
        super(ParagraphTransformer, self).__init__()
        
        self.embedding = SentenceTransformer(model_args.model_name_or_path, device )
        
        if model_args.model_name_or_path == 'BAAI/bge-m3' or model_args.dimension == 1024:
            self.linear = nn.Linear(1024, 768)  # Add a linear layer to reduce the last dimension
        
        self.linear = nn.Linear(768, 768)  # Add a linear layer to reduce the last dimension

         
        self.LayerNorm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.avg_pooling =  nn.AdaptiveAvgPool1d(1)
    
    # @property
    # def device(self):
    #     return self.device

    def new_forward(self, features):
        """
        Forward pass method for the ParagraphTransformer.
        
        Args:
        - features: A list of sentences forming a paragraph
        
        Returns:
        - out: Encoded representation of the paragraph
        """
        # Perform the forward pass through the original model
        out = self.embedding.encode(features)
        #out = self.encode(features)
        #out = super(CustomSentenceTransformer, self).forward(features)
        try:
            out = self.LayerNorm(torch.Tensor(out).to(device))
        except:
            out = self.LayerNorm(torch.Tensor(out).to('cpu'))
        out = self.dropout(out)
        # Convert to torch and apply average pooling
        out = torch.Tensor(out).unsqueeze(0).transpose(1, 2)
        
        # Apply average pooling
        out = self.avg_pooling(out).squeeze()  # Apply 1D avg pooling and squeeze
        return out.to(device)



class DocumentTransformer(SentenceTransformer):
    def __init__(self, ParaEmbedding = None):

        super(DocumentTransformer, self).__init__()
        

        self.ParaEmbedding = ParaEmbedding 
    def new_forward(self, document: list[list[str]], passError = False):
        tensors = []
        for paragraph in tqdm.tqdm(document, desc= "Embedding Document: "):
            embedding = self.ParaEmbedding.new_forward(paragraph, notice= False, passError= passError)
            if embedding is not None:
                tensors.append(embedding)

        # Concatenate them along the first dimension
        merged_tensor = torch.cat(tensors, dim=0)

        # Verify the new shape
        print("Shape of document_embeddings: ", merged_tensor.shape)
        
        return merged_tensor
    


        