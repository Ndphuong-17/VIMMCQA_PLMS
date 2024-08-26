
from transformers import AutoModel, AutoConfig, AutoModelForMultipleChoice
from torch import nn
import torch

class VimMCQAModel(nn.Module):
  def __init__(self, model_name_or_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
      super(VimMCQAModel, self).__init__()
      self.base = self.load_base_architecture(model_name_or_path)
      self.device = device

  def load_base_architecture(self, model_name_or_path):
    # Config
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.add_cross_attention = True
    config.is_decoder = True

    return AutoModelForMultipleChoice.from_config(config)

  def embedding_context(self, contexts):
    results = []
    for context in contexts:
      embedded_context = self.base.roberta.embeddings(context.input_ids.to(self.device))
      results.append(embedded_context.mean(dim=(0, 1)))

    return torch.stack(results).unsqueeze(dim = 1)

  def forward(self, sample):
    input_embeddings = self.base.roberta.embeddings(sample['tokenized_input'].input_ids.to(self.device)) # embedding inputs
    context_embeddings = self.embedding_context(sample['tokenized_context']) # embedding contexts

    attention_mask = sample['tokenized_input'].attention_mask.to(self.device)
    if attention_mask.dim() == 2:  # If shape is (batch_size, seq_len)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)

    output = self.base.roberta.encoder(hidden_states=input_embeddings,
                          attention_mask=attention_mask,
                          encoder_hidden_states=context_embeddings) # batch_size, sequence_length, 768

    prediction = self.base.classifier(self.base.dropout(output.last_hidden_state.mean(dim = 1))).squeeze(dim=1) # batch_size

    return prediction
