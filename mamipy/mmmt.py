import copy
    
import torch
import torch.nn as nn

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    modeling_utils,
    modeling_outputs
)

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import BaseModelOutputWithPooling

class BatchEmbeddingAlignmentLoss(nn.Module):
    def __init__(self):
        super(BatchEmbeddingAlignmentLoss, self).__init__()
        self.loss_fn = nn.L1Loss() #nn.MSELoss()
        self.epsilon = 1e-8

    def _pairs(self, input):
        """
        input has shape B x N x E, where B is batch size, N is sequence length and E is embedding length
        input is reshaped to N x B x E
        Then computes the outer product of each (B x E) matrix as (N x B x E) . (N x E x B)
        to construct the matrix (N x B x B) such that the element at (l, i, j) = input[i, l] dot input[j, l]
        """
        input_normalized = torch.nn.functional.normalize(input, dim=-1)
        # reshape labels to: sequence length x batch_size x embedding
        input_normalized_batched = torch.permute(input_normalized, [1, 0, 2])
        # transpose batches per sequence index
        input_normalized_batched_T = torch.permute(input_normalized_batched.T, [2, 0, 1])
        # construct sequence item pairs
        return torch.matmul(input_normalized_batched, input_normalized_batched_T)

    def forward(self, input, target):
        device = input.device

        input_paris = self._pairs(input)
        target_paris = self._pairs(target)

        loss = self.loss_fn(input_paris, target_paris)

        return loss

class ModalTransformerEncoder(nn.Module):
    """Generic Modal Transformer Encoder which takes in an encoder backbone and
    (BERT) position and token type embedding and LayerNorm"""

    def __init__(self, modal_hidden_size, transformer_hidden_size, encoder, embeddings, transformer_nhead=8, transformer_num_layers=6, projection_alignment=False):
        super().__init__()
        self.encoder = encoder
        self.proj_embeddings = nn.Linear(modal_hidden_size, transformer_hidden_size)
        self.projection_alignment = projection_alignment
        if projection_alignment:
            self.projection_loss_fn = BatchEmbeddingAlignmentLoss()

        # copy position embedding
        self.position_embeddings = nn.Embedding(embeddings.position_embeddings.num_embeddings, transformer_hidden_size)
        self.position_embeddings.weight = nn.Parameter(embeddings.position_embeddings.weight.detach())
        self.position_embeddings.weight.requires_grad = True

        # copy embedding of token type 0
        self.token_type_embeddings = nn.Embedding(1, transformer_hidden_size)
        self.token_type_embeddings.weight = nn.Parameter(embeddings.token_type_embeddings.weight.detach()[0].unsqueeze(0))
        self.token_type_embeddings.weight.requires_grad = True

        # copy LayerNorm parameters
        self.LayerNorm = nn.LayerNorm(transformer_hidden_size, eps=embeddings.LayerNorm.eps)
        self.LayerNorm.weight = nn.Parameter(embeddings.LayerNorm.weight.detach())
        self.LayerNorm.weight.requires_grad = True
        self.LayerNorm.bias = nn.Parameter(embeddings.LayerNorm.bias.detach())
        self.LayerNorm.bias.requires_grad = True

        self.dropout = nn.Dropout(p=embeddings.dropout.p)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_hidden_size, nhead=transformer_nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=transformer_num_layers)

    def embed(self, input_modal, modal_encoder_kwargs={}):
        device = input_modal.device
        batch_size = input_modal.size(0)
        encoding, mask = self.encoder(input_modal, **modal_encoder_kwargs)
        if mask is None: mask = torch.ones(encoding.size()[:-1], device=encoding.device)
        encoding = encoding.type(torch.float32)
        token_embeddings = self.proj_embeddings(encoding)
        projection_loss = self.projection_loss_fn(token_embeddings, encoding) if self.projection_alignment else torch.zeros(1, device=device)

        seq_length = token_embeddings.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.expand(batch_size, seq_length)

        token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, mask, projection_loss

    def forward(self, input_modal, modal_encoder_kwargs={}):
        embeddings, mask, projection_loss = self.embed(input_modal, modal_encoder_kwargs=modal_encoder_kwargs)
        # pytorch expects True for positions to be ignored and False ow
        # However, in input, 1 positions shouldn't be ignored while 0s should be
        # So, we invert the mask
        out = self.transformer_encoder(embeddings, src_key_padding_mask=mask < 1)
        return out, mask, projection_loss

class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding."""

    def __init__(self, modal_hidden_size, transformer_hidden_size, encoder, embeddings, token_type_id, end_token_id, hidden_dropout_prob, projection_alignment=False):
        super().__init__()
        self.encoder = encoder
        
        self.proj_embeddings = nn.Linear(modal_hidden_size, transformer_hidden_size)
        self.projection_alignment = projection_alignment
        if projection_alignment:
            self.projection_loss_fn = BatchEmbeddingAlignmentLoss()
            
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.token_type_id = token_type_id
        self.end_token_id = end_token_id

    def forward(self, input_modal, modal_encoder_kwargs={}):
        device = input_modal.device
        batch_size = input_modal.size(0)
        encoding, mask = self.encoder(input_modal, **modal_encoder_kwargs)
        token_embeddings = self.proj_embeddings(encoding)
        projection_loss = self.projection_loss_fn(token_embeddings, encoding) if self.projection_alignment else torch.zeros(1, device=device)

        end_token_embeds = self.word_embeddings(
            torch.tensor(self.end_token_id, dtype=torch.long, device=device)
        ).expand(batch_size, 1, -1)
        token_embeddings = torch.cat([token_embeddings, end_token_embeds], dim=1)

        seq_length = token_embeddings.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        token_type_ids = torch.tensor(self.token_type_id, dtype=torch.long, device=device)
        token_type_ids = token_type_ids.unsqueeze(0).expand(batch_size, seq_length)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        if mask is None:
            mask = torch.ones(embeddings.size()[:-1], device=device)
        elif mask.size(1) == embeddings.size(1) - 1:
            mask = torch.cat([mask, torch.ones((batch_size, 1), device=device)], dim=1) # for the sep
        return embeddings, mask, projection_loss

class MMSTModel(nn.Module, ModuleUtilsMixin):
    """Multi Modal Sigle/Shared Transformer"""
    def __init__(self, config, transformer, encoders, start_token_id, end_token_id, pool_output=False, projection_alignment=False):
        super().__init__()

        self.config = config
        self.transformer = transformer
        self.pool_output = pool_output

        # we have token type for each encoder + 1 for text encoder and cls token
        token_type_embeddings_weight = self.transformer.embeddings.token_type_embeddings.weight.detach()
        token_type_embeddings_weight = torch.cat([
            token_type_embeddings_weight,
            token_type_embeddings_weight[1].unsqueeze(0).repeat(len(encoders) - 1, 1)
        ], dim=0)
        token_type_embeddings = nn.Embedding(len(encoders) + 1, config.hidden_size)
        token_type_embeddings.weight = nn.Parameter(token_type_embeddings_weight)
        token_type_embeddings.weight.requires_grad = True
        self.transformer.embeddings.token_type_embeddings = token_type_embeddings

        modal_encoders = []
        for i, (encoder, modal_hidden_size) in enumerate(encoders):
            modal_encoders.append(ModalEmbeddings(modal_hidden_size, config.hidden_size, encoder,
                self.transformer.embeddings, token_type_id=i+1, end_token_id=end_token_id,
                hidden_dropout_prob=config.hidden_dropout_prob, projection_alignment=projection_alignment))
        self.modal_encoders = nn.ModuleList(modules=modal_encoders)

        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

    def forward(
        self,
        input_modal,
        input_ids,
        attention_mask=None,
        modal_encoders_kwargs=None
    ):
        input_txt_shape = input_ids.size()
        batch_size = input_txt_shape[0]

        device = input_ids.device

        if isinstance(input_modal, list) and len(input_modal) != len(self.modal_encoders):
            raise ValueError("input_modal should be a tensor or a list of tensors with length equals to number of encoders")

        # Embed start token
        start_token_embeds = self.transformer.embeddings(
            input_ids=torch.tensor([[self.start_token_id]], dtype=torch.long, device=device).repeat(batch_size, 1),
            position_ids=None,
            token_type_ids=torch.zeros((batch_size, 1), dtype=torch.long, device=device),
            inputs_embeds=None
        )

        # Embed modal
        modal_embeddings_list = []
        modal_mask_list = []
        projection_loss = torch.zeros(1, device=device)
        for i, encoder in enumerate(self.modal_encoders):
            modal = input_modal[i] if isinstance(input_modal, list) else input_modal
            modal_encoder_kwargs = modal_encoders_kwargs[i] if modal_encoders_kwargs is not None else {}
            modal_embedding, modal_mask, modal_projection_loss = encoder(modal, modal_encoder_kwargs=modal_encoder_kwargs)
            modal_embeddings_list.append(modal_embedding)
            modal_mask_list.append(modal_mask)
            projection_loss += modal_projection_loss

        modal_embeddings = torch.cat(modal_embeddings_list, 1)
        modal_attention_mask = torch.cat(modal_mask_list, 1)

        # Embed text
        token_type_ids = torch.zeros(input_txt_shape, dtype=torch.long, device=device)
        txt_embeddings = self.transformer.embeddings(
            input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
        )

        # Concatenate all embeddings
        embedding_output = torch.cat([start_token_embeds, modal_embeddings, txt_embeddings], dim=1)
        seq_length = embedding_output.size(1)

        input_shape = embedding_output.size()[:-1]

        txt_attention_mask = attention_mask if attention_mask is not None else torch.ones(input_txt_shape, device=device)

        attention_mask = torch.cat(
            [
                torch.ones((batch_size, 1), device=device, dtype=torch.long), # first one for start token
                modal_attention_mask,
                txt_attention_mask
            ], dim=1
        )
        encoder_attention_mask = torch.ones(input_shape, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        head_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.transformer.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=False,
        )

        if self.pool_output:
            return self.transformer.pooler(encoder_outputs[0]), attention_mask, projection_loss
        else:
            return encoder_outputs[0], attention_mask, projection_loss

class MMMTModel(nn.Module):
    """Multi Modal Multi Transformer"""
    def __init__(self, config, txt_transformer, encoders, start_token_id=None,
        pool_txt=False, transformer_nhead=8, transformer_num_layers=6, projection_alignment=False):
        super().__init__()

        self.config = config
        self.txt_transformer = txt_transformer
        self.pool_txt = pool_txt

        modal_encoders = []
        for i, (encoder, modal_hidden_size) in enumerate(encoders):
            modal_encoders.append(ModalTransformerEncoder(modal_hidden_size, config.hidden_size, encoder,
                self.txt_transformer.embeddings, transformer_nhead=transformer_nhead, transformer_num_layers=transformer_num_layers,
                projection_alignment = projection_alignment))
        self.modal_encoders = nn.ModuleList(modules=modal_encoders)

        self.start_token_id = start_token_id

    def forward(
        self,
        input_modal,
        input_ids,
        attention_mask=None,
        modal_encoders_kwargs=None
    ):
        batch_size = input_ids.size(0)

        device = input_ids.device

        if isinstance(input_modal, list) and len(input_modal) != len(self.modal_encoders):
            raise ValueError("input_modal should be a tensor or a list of tensors with length equals to number of encoders")

        all_embeddings_list = []
        all_masks_list = []

        # Embed modal
        projection_loss = torch.zeros(1, device=device)
        for i, encoder in enumerate(self.modal_encoders):
            modal = input_modal[i] if isinstance(input_modal, list) else input_modal
            modal_encoder_kwargs = modal_encoders_kwargs[i] if modal_encoders_kwargs is not None else {}
            embd, mask, modal_projection_loss = encoder(modal, modal_encoder_kwargs=modal_encoder_kwargs)
            all_embeddings_list.append(embd)
            all_masks_list.append(mask)
            projection_loss += modal_projection_loss

        # Embed text
        if self.start_token_id is not None:
            start_token_ids = torch.tensor([self.start_token_id], dtype=torch.long, device=device).expand(batch_size, 1)
            input_ids = torch.cat([start_token_ids, input_ids], dim=1)
            if attention_mask is not None:
                start_token_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
                attention_mask = torch.cat([start_token_mask, attention_mask], dim=1)

        txt_mask = attention_mask if attention_mask is not None else torch.ones(input_ids.size(), device=device)
        txt_transformer_output = self.txt_transformer(input_ids=input_ids, attention_mask=txt_mask, return_dict=True)
        txt_embeddings = txt_transformer_output.pooler_output.unsqueeze(1) if self.pool_txt else txt_transformer_output.last_hidden_state
        all_embeddings_list.append(txt_embeddings)
        all_masks_list.append(torch.ones((batch_size, 1), device=device) if self.pool_txt else txt_mask)

        # Concatenate all embeddings
        embedding_output = torch.cat(all_embeddings_list, dim=1)
        mask = torch.cat(all_masks_list, dim=1)

        return embedding_output, mask, projection_loss
