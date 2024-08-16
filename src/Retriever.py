import numpy as np
from rank_bm25 import BM25Okapi
import torch
import re
from string import punctuation
try:
    from EmbbeddingTransformer import  ParagraphTransformer #, DocumentTransformer
except:
    try:
        from src.EmbbeddingTransformer import  ParagraphTransformer
    except:
        from VIMMCQA_PLMS.src.EmbbeddingTransformer import  ParagraphTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Retrieval(torch.nn.Module):
    """
    Retrieval model for retrieving relevant contexts given a query.
    
    Args:
    - model_args: Arguments for the embedding model.
    - data_args: Additional data arguments.
    - device: Device to run the model on.
    - corpus: Corpus data.
    - corpus_vector: Pre-calculated corpus vectors.
    - bm25: BM25 instance.
    
    Output:
    - predicted_label: Predicted labels.
    - true_label: True labels.
    """

    def __init__(self, 
                 model_args,
                 corpus = None,
                 ):
        super(Retrieval, self).__init__()
        #pretrained_model = AutoModel.from_pretrained(model_name_or_path)
        self.embedding = ParagraphTransformer(model_args)
        
        # getting corpus wseg or not
        if corpus is not None:
            self.wseg_datas = corpus
            self.non_wseg_datas = [para.replace('_', ' ') for para in corpus]
        else:
            raise ValueError('Need a corpus tu initialize a model.')
        print("Initializing corpus data completely.")
        
        # self.corpus_vector = corpus_vector if corpus_vector is not None\
        # else self.embedding_corpus(self.wseg_datas)
        # print("Initializing corpus vector completely.")

        tokenized_corpus = [doc.split(" ") for doc in corpus]

        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # self.bm25 = bm25 if bm25 is not None \
        # else BM25Okapi(self.corpus_vector)
        #else self.initializing_bm25(self.wseg_datas)
        print("Initializing bm25 completely.")
        
        print("Initializing Retrieval model completely.")
    
    
    # @property
    # def device(self):
    #     return self.device

    def retrieval(self, query = '', weight = [0.9, 1.1], get_num_top = 5, k = 1):
        
        # preprocessing query
        tokenized_query = self.preprocessing_data(query).split(" ")
        query = self.preprocessing_para(query)

        # caculating embedding vectors of datas
        # self.corpus_vector
        
        # initialize bm25
        #self.bm25

        # caculate bm25_scores between query and datas
        bm25_scores_norm = self.caculate_normalize_bm25(tokenized_query, get_num_top = get_num_top)

        # caculate cosine_similarity between query and datas
        query_vector, cosine_similarity_score = self.caculate_cosine_similarity(query)

        # caculate final score
        final_scores = (weight[0]*torch.Tensor(bm25_scores_norm).to(device) + weight[1]*cosine_similarity_score)/sum(weight)
        final_scores = final_scores.to(device)
        
        if k == 1:
            # get evidence having max scores
            top_scores_ids = np.argsort(final_scores.detach().to('cpu').numpy())[::-1]

            # Get paragraph and vector of paragraph similarity with query
            relevant_wseg_context = self.wseg_datas[top_scores_ids[0]]
            relevant_context = self.non_wseg_datas[top_scores_ids[0]]
            relevant_vector  = self.embedding_corpus([self.wseg_datas[top_scores_ids[0]]])
        elif k > 1:
            # Get evidence having max scores
            top_scores_ids = np.argsort(final_scores.detach().to('cpu').numpy())[::-1][:k]

            # Get paragraphs and vectors of paragraphs similar to query
            relevant_wseg_context = [self.wseg_datas[idx] for idx in top_scores_ids]
            relevant_context = [self.non_wseg_datas[idx] for idx in top_scores_ids]
            relevant_vector = torch.cat([self.embedding_corpus([self.wseg_datas[idx]]) for idx in top_scores_ids], dim=0)
        else:
            raise ValueError("The top k must to greater or equal 1.")


        return relevant_wseg_context, relevant_context, relevant_vector.to(device), query_vector.to(device), final_scores.to(device)

    def caculate_cosine_similarity(self, query = None, vector_query = None):
        query_vector = self.embedding.new_forward(query) if vector_query is None\
        else vector_query
        try:
            scores = torch.cosine_similarity(query_vector.unsqueeze(0).to(device), self.corpus_vector.to(device), dim=1).to(device)
        except:
            scores = 0

        return query_vector,scores
    
    def caculate_normalize_bm25(self, tokenized_query, get_num_top = 5):
        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_index = np.argsort(bm25_scores)[::-1]
        m2 = [0]*len(bm25_scores)
        for x, i in enumerate(top_index[:get_num_top]):
            rate =  1 / 2**x
            if rate < 1e-8:
                rate = 0
            m2[i] = rate
        m2 = np.array(m2) * 2
        bm25_scores *= m2
        epsilon = 1e-10
        bm25_scores_norm = bm25_scores / (np.sum(bm25_scores) + epsilon)
        #bm25_scores_norm = bm25_scores / np.sum(bm25_scores)

        return bm25_scores_norm

    def initializing_bm25(self, wseg_datas: list[str] = None):
        if wseg_datas is None:
            processed_datas = [self.preprocessing_data(para) for para in self.wseg_datas]
        else:
            processed_datas = [self.preprocessing_data(para) for para in wseg_datas]
        tokenized_corpus = [para.split(" ") for para in processed_datas]
        
        bm25 = BM25Okapi(tokenized_corpus)

        return bm25
    
    
    def embedding_corpus(self, wseg_datas: list[str] = None):
        if wseg_datas is None:
            processed_datas = [self.preprocessing_para(para) for para in self.wseg_datas]
        else:
            processed_datas = [self.preprocessing_para(para) for para in wseg_datas]
            
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
    
    