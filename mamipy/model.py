import torch
import torch.nn as nn
import torchvision

import clip

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    DetrFeatureExtractor,
    DetrModel,
    DetrForObjectDetection,
    DetrConfig
)

from . import mmmt
from .mmmt import (MMSTModel, MMMTModel)

from collections import OrderedDict



def set_parameter_requires_grad(model, flag):
    for param in model.parameters():
        param.requires_grad = flag

class CLIPImageEmbeddingLayer(nn.Module):
    def __init__(self, num_patches=(2, 2), finetune=False):
        super(CLIPImageEmbeddingLayer, self).__init__()       
        
        self.finetune = finetune
        self.num_patches = num_patches
        
        clip_model, preprocess = clip.load("RN50x4", jit=False)
        image_size = preprocess.transforms[0].size # 288
        self.patches = num_patches[0] * num_patches[1] + 1 # whole image + 4 slices
        self.embed_length = clip_model.ln_final.normalized_shape[0] # 640

        # same as preprocess but works on image tensors
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=image_size, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            torchvision.transforms.CenterCrop(size=image_size),
            torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        self.clip = clip_model.float()
        set_parameter_requires_grad(self.clip, finetune)
        
        self.name = 'clip'
    
    def slice_image_tensor(self, images):
        """
        Slice an image tensor into nxm patches
        """
        #torch.Tensor.unfold(dimension, size, step)
        #slices the images into patches[0]*patches[1] number of patches

        patch_size = [l // p for l, p in zip(images[0][0].size(), self.num_patches)]
        # display('patch_size: ' + str(patch_size))

        patches = images.unfold(1, 3, 1)
        for i, lp in enumerate(patch_size):
            patches = patches.unfold(i + 2, lp, lp)

        # B * 4 x 3 x w x h
        return patches.reshape(-1, *patches.size()[-3:])
    
    def forward(self, images, clip_features=None):
        # B x 3 x w x h -> B x 5 x 640
        batch_size = images.size(0)

        if self.finetune or clip_features is None:
            images_transformed = self.image_transform(images)
            slices_transformed = self.image_transform(self.slice_image_tensor(images))
            
            images_and_slices = torch.cat([
                    images_transformed.unsqueeze(1),
                    slices_transformed.reshape(batch_size, self.patches - 1, *slices_transformed.size()[-3:])
                ], dim=1).reshape(-1, *images_transformed.size()[-3:])
                    
            embd = self.clip.encode_image(images_and_slices)
            embd = embd.reshape(batch_size, self.patches, self.embed_length)

            if torch.isnan(embd.sum()).item():
              print('images', images.sum().item(), images, "\n",
                'images_transformed: ', images_transformed.sum().item(), images_transformed, "\n",
                'slices_transformed: ', slices_transformed.sum().item(), slices_transformed, "\n",
                'images_and_slices: ', images_and_slices.sum().item(), images_and_slices, "\n")
        else:
            embd = clip_features
        
        mask = torch.ones((batch_size, self.patches), device=images.device).type(torch.float32)
        
        

        #print('embd: ' + str(embd.size()))
        #print('mask: ' + str(mask.size()))
        return embd, mask # B x 5 x 640, B x 5

class DETRImageEmbeddingLayer(nn.Module):
    def __init__(self, fallback_topk=4, finetune=False):
        super(DETRImageEmbeddingLayer, self).__init__()       
        
        self.finetune = finetune
        self.fallback_topk = fallback_topk
        
        detr_model_name = "facebook/detr-resnet-50"
        self.detr_feature_extractor = DetrFeatureExtractor.from_pretrained(detr_model_name)
        detr_model = DetrForObjectDetection.from_pretrained(detr_model_name)

        self.patches = detr_model.config.num_queries # 100
        self.embed_length = detr_model.config.d_model # 256
        self.detr = detr_model
        set_parameter_requires_grad(self.detr, finetune)
        
        self.name = 'detr'
    
    def forward(self, images, detr_features=None, detr_logits=None):
        if self.finetune or detr_features is None:
            # B x 3 x w x h -> B x 100 x 256
            encoding = self.detr_feature_extractor(list(images), return_tensors="pt")
            outputs = self.detr(**encoding)
            
            embd = outputs.last_hidden_state
            detr_logits = outputs.logits
        else:
            embd = detr_features

        mask1 = detr_logits.softmax(-1).argmax(-1)
        mask1 = torch.where(mask1 == 91, 0, 1).type(torch.float32) # ignore the no-object feature

        zero_mask = (mask1.sum(1) < 1).reshape(mask1.size(0), 1).repeat((1, mask1.size(1))) # if all objects are no-object's
        mask2 = detr_logits[..., :-1].softmax(-1).max(-1).values # softmax excluding the no-object class
        mask2 = torch.zeros(mask2.size(), device=images.device).scatter(1, mask2.topk(self.fallback_topk).indices, 1) # take top 4 objects
        mask = torch.where(zero_mask, mask2, mask1)

        if mask.sum(1).min() < 1:
            print('still has all zeros mask')
            print('mask', mask)
            print('mask1', mask1)
            print('mask2', mask2)

        #print('embd: ' + str(embd.size()))
        #print('mask: ' + str(mask.size()))
        return embd, mask # B x 100 x 256, B x 100

class CLIPFeaturesImageEmbeddingLayer(nn.Module):
    def __init__(self):
        super(CLIPFeaturesImageEmbeddingLayer, self).__init__()       
        
        clip_model, preprocess = clip.load("RN50x4", jit=False)
        #image_size = preprocess.transforms[0].size # 288
        self.patches = 5 # whole image + 4 slices
        self.embed_length = clip_model.ln_final.normalized_shape[0] # 640

        self.name = 'clip'
      
    def forward(self, input_modal, clip_features=None):
        clip_mask = torch.ones(clip_features.size()[:-1], device=input_modal.device).type(torch.float32)
        return clip_features, clip_mask
    
class DETRFeaturesImageEmbeddingLayer(nn.Module):
    def __init__(self):
        super(DETRFeaturesImageEmbeddingLayer, self).__init__()       
        
        detr_model_name = "facebook/detr-resnet-50"
        config = DetrConfig.from_pretrained(detr_model_name)

        self.patches = config.num_queries # 100
        self.embed_length = config.d_model # 256

        self.fallback_topk = 4

        self.name = 'detr'

    def forward(self, input_modal, detr_features=None, detr_logits=None):
        # detr_mask = detr_logits.softmax(-1).argmax(-1)
        # # detr_mask = torch.cat([
        # #     torch.where(detr_mask == 91, 0, 1), # ignore the no-object feature
        # #     torch.ones(1) # for the sep
        # # ]).type(torch.float32)
        # detr_mask = torch.where(detr_mask == 91, 0, 1).type(torch.float32) # ignore the no-object feature
        # if detr_mask.sum().item() == 0: # if all objects are no-object's
        #     detr_mask = detr_logits[..., :-1].softmax(-1).max(1).values # softmax exluding the no-object class
        #     detr_mask = torch.zeros(detr_mask.size(), device=input_modal.device).scatter(0, detr_mask.topk(4).indices, 1) # take top 4 objects

        mask1 = detr_logits.softmax(-1).argmax(-1)
        mask1 = torch.where(mask1 == 91, 0, 1).type(torch.float32) # ignore the no-object feature

        zero_mask = (mask1.sum(1) < 1).reshape(mask1.size(0), 1).repeat((1, mask1.size(1))) # if all objects are no-object's
        mask2 = detr_logits[..., :-1].softmax(-1).max(-1).values # softmax excluding the no-object class
        mask2 = torch.zeros(mask2.size(), device=input_modal.device).scatter(1, mask2.topk(self.fallback_topk).indices, 1) # take top 4 objects
        detr_mask = torch.where(zero_mask, mask2, mask1)

        return detr_features, detr_mask

class BatchBinaryCosineLoss(nn.Module):
    """
    Batch: works on all pairs in the batch
    Binary: targets are binary labels (0 for negative and 1 for positive)
    
    Assumes all positive instances are close, and are apart from negative instances
    """
    def __init__(self, vector_per_class=False, negatives_are_close=False):
        super(BatchBinaryCosineLoss, self).__init__()
        self.vector_per_class = vector_per_class
        self.negatives_are_close = negatives_are_close
        self.epsilon = 1e-8
        
    def _labels_pairs(self, lbs):
        """
        lbs has shape B x N, where B is batch size and N is number of labels per instance
        lbs is reshaped to N x B x 1
        Then computes the outer product of each (B x 1) vector as (N x B x 1) . (N x 1 x B)
        to construct the matrix (N x B x B) such that the element at (l, i, j) = lbs[i, l] * lbs[j, l]
        """
        return self._pairs(torch.unsqueeze(lbs, -1))
    
    def _pairs(self, lbs):
        """
        lbs has shape B x N x E, where B is batch size, N is number of labels per instance and E is embedding length
        lbs is reshaped to N x B x E
        Then computes the outer product of each (B x E) matrix as (N x B x E) . (N x E x B)
        to construct the matrix (N x B x B) such that the element at (l, i, j) = lbs[i, l] dot lbs[j, l]
        """
        # reshape labels to: number of labels per instance x batch_size x embedding
        lbs_batched = torch.permute(lbs, [1, 0, 2])
        # transpose batches per label
        lbs_batched_T = torch.permute(lbs_batched.T, [2, 0, 1])
        # construct labels pairs
        return torch.matmul(lbs_batched, lbs_batched_T)
        
    def forward(self, input, target, target_mask=None):
        device = input.device
        batch_size = input.size()[0]

        # zeros diagonal, ones everywhere
        eye_comp = (1 - torch.eye(batch_size, device=device))

        if self.negatives_are_close:
            pairs_selector = eye_comp.expand([target.size(1), *eye_comp.size()])
        else:
            # select only pairs with different instances with at least one positive instance
            # here we don't care whether two negative instances have similar embeddings or not
            # False when both labels are 0, False diagonal (a pair of the same instance), and True otherwise
            # zero diagonals per label by broadcasting eye_comp per label and do hadamard product
            pairs_selector = eye_comp * (self._labels_pairs(target + 1) - 1)
            
        if target_mask is not None:
            # if target_mask is a vector, convert it to a matrix
            if len(target_mask.size()) == 1: target_mask = target_mask.unsqueeze(0)
            pairs_selector *= self._labels_pairs(target_mask)
            
        mask = pairs_selector > 0

        # convert labels from 0 or 1 to -1 or 1
        # then construct labels pairs; 1 indicates similar labels, -1 indicates dissimilar labels
        pairs_labels_similarity = self._labels_pairs(2 * target - 1)
        
        input_normalized = torch.nn.functional.normalize(input, dim=-1)
        if self.vector_per_class:
            pairs_cosine_similiarity = self._pairs(input_normalized)
        else:
            pairs_cosine_similiarity = torch.matmul(input_normalized, input_normalized.T)
        
        pairs_labels_similarity = torch.masked_select(pairs_labels_similarity, mask)
        pairs_cosine_similiarity = torch.masked_select(pairs_cosine_similiarity, mask)
        selected_pairs_count = pairs_labels_similarity.size()[0]

        # loss = 0.5 * (1 - pairs_labels_similarity * pairs_cosine_similiarity)
        # - when labels are similar (i.e. corresponding pairs_labels_similarity = 1),
        #   loss = 0.5 * (1 - pairs_cosine_similiarity)
        #   which is minimum (=0) when embeddings are actually similar (pairs_cosine_similiarity=1)
        #   and is maximum (=1) when embeddings are 2xpi dissimilar (pairs_cosine_similiarity=-1)
        # - when labels are dissimilar (i.e. corresponding pairs_labels_similarity = -1),
        #   loss = 0.5 * (1 + pairs_cosine_similiarity)
        #   which is maximum (=1) when embeddings are similar (pairs_cosine_similiarity=1)
        #   and is minimum (=0) when embeddings are 2xpi dissimilar (pairs_cosine_similiarity=-1)
        
        a = 0.5 * (1 - pairs_labels_similarity * pairs_cosine_similiarity)
        ## consider values before 0.1 as 0
        ## consider values after 0.9 as 1
        #a = torch.where(a > 0.1, a, torch.tensor(0.0))
        #a = torch.where(a < 0.9, a, torch.tensor(1.0))
        
        # we scale down the loss of similar labels
        multiplier = torch.where(pairs_labels_similarity > 0.0, 0.5, 1.0)
        a = multiplier * a
        
        # get the mean loss
        # divide sum by epsilon for numerical stability when there are no pairs selected (i.e. when selected_pairs_count = 0)
        # it's symmetric, divide by 2
        loss = 0.5 * a.sum() / max(self.epsilon, selected_pairs_count)
        
        return loss

class ContrastiveMLPUnit(nn.Module):
    def __init__(self, input_size, output_size, dropout_p=0.1, has_activation=True, enable_embed_loss=False, negatives_are_close=False, vector_per_class=False):
        super(ContrastiveMLPUnit, self).__init__()

        if has_activation:
            self.f = nn.Sequential(OrderedDict([
                ('dense', nn.Linear(input_size, output_size)),
                ('activation', nn.GELU())
            ]))
        else:
            self.f = nn.Linear(input_size, output_size)

        self.dropout = nn.Dropout(p=dropout_p)
        
        self.enable_embed_loss = enable_embed_loss
        if enable_embed_loss:
            self.cosine_loss = BatchBinaryCosineLoss(vector_per_class=vector_per_class, negatives_are_close=negatives_are_close)
    
    def forward(self, input):
        return self.dropout(self.f(input))
    
    def forward_with_loss(self, input, target=None, target_mask=None):
        output = self.dropout(self.f(input))
        if self.enable_embed_loss:
            embed_loss = self.cosine_loss(output, target, target_mask)
            if target is None:
                raise ValueError('target must be provided when enable_embed_loss is True')
        else:
            embed_loss = torch.zeros(1, device=input.device)
        return output, embed_loss
        
class ContrastiveMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1, layers_count=4, enable_embed_loss=False, negatives_are_close=False, vector_per_class=False):
        super(ContrastiveMLP, self).__init__()
        self.enable_embed_loss = enable_embed_loss
        
        h = [hidden_size] * (layers_count - 1)
        input_size_per_layer = [input_size] + h
        output_size_per_layer = h + [output_size]
        dropout_p_per_layer = [dropout_p] * (layers_count - 1) + [0.0] # no dropout in last layer
        enable_embed_loss_per_layer = [enable_embed_loss] * (layers_count - 1) + [False] # no embed loss in last layer
        has_activation_per_layer = [True] * (layers_count - 1) + [False] # no activation in last layer
        self.layers = nn.ModuleList([
            ContrastiveMLPUnit(n, o, p, a, e, negatives_are_close, vector_per_class) for n, o, p, a, e in zip(
                input_size_per_layer, output_size_per_layer, dropout_p_per_layer, enable_embed_loss_per_layer, enable_embed_loss_per_layer
            )
        ])

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def forward_with_loss(self, input, target=None, target_mask=None):
        if self.enable_embed_loss and target is None:
            raise ValueError('target must be provided when enable_embed_loss is True')
            
        total_embed_loss = torch.zeros(1, device=input.device)
        output = input
        for layer in self.layers:
            output, embed_loss = layer.forward_with_loss(output, target, target_mask)
            total_embed_loss += embed_loss
        return output, total_embed_loss
        
class VariationalClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_output_classes, dropout_p=0.1,
     layers_count=4, enable_kl_loss=True, enable_embed_loss=False,
      negatives_are_close=False, vector_per_class=False, pos_weight=None):
        super(VariationalClassifier, self).__init__()
        self.hidden_size = hidden_size
        
        self.enable_kl_loss = enable_kl_loss
        self.enable_embed_loss = enable_embed_loss
        self.vector_per_class = vector_per_class
        
        if enable_kl_loss:
            self.f = nn.Sequential(OrderedDict([
                ('dense', nn.Linear(input_size, hidden_size)),
                ('activation', nn.Tanh())
            ]))
            # g computes mu, can be negative
            self.g = nn.Sequential(OrderedDict([
                ('dense', nn.Linear(hidden_size, hidden_size)),
                ('activation', nn.Tanh())
            ]))
            # h computes sigma, should be >= 0
            self.h = nn.Sequential(OrderedDict([
                ('dense', nn.Linear(hidden_size, hidden_size)),
                ('activation', nn.Sigmoid())
            ]))
            classifier_input_size = hidden_size
        else:
            classifier_input_size = input_size
        
        self.classifier = ContrastiveMLP(classifier_input_size, hidden_size, num_output_classes, dropout_p, layers_count, enable_embed_loss, negatives_are_close, vector_per_class)
        
        if enable_embed_loss:
            self.cosine_loss = BatchBinaryCosineLoss(vector_per_class=vector_per_class, negatives_are_close=negatives_are_close)
        
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)# if num_output_classes == 1 or vector_per_class else nn.CrossEntropyLoss()

    def encode(self, embed):
        if self.enable_kl_loss:
            f = self.f(embed)
            mu = self.g(f)
            sigma = self.h(f)
            z = torch.randn(f.size(), device=embed.device) if self.training else torch.zeros(f.size(), device=embed.device)
            embed_hat = z * sigma + mu

            return embed_hat, mu, sigma
        else:
            return embed, None, None
        
    def forward(self, embed):
        embed_hat, mu, sigma = self.encode(embed)
        classifications = self.classifier(embed_hat)
        if self.vector_per_class:
            classifications = classifications.squeeze(-1)
        #print(f"classifications: {classifications}")
        return classifications, embed_hat, mu, sigma
    
    def forward_with_loss(self, embed, labels, labels_mask=None):
        embed_hat, mu, sigma = self.encode(embed)
        classifications, embed_loss = self.classifier.forward_with_loss(embed_hat, labels, labels_mask)
        if self.vector_per_class:
            classifications = classifications.squeeze(-1)

        classifier_loss = self.loss_fn(classifications, labels)
        if torch.isnan(classifier_loss).item():
          print('classifications: ', classifications, "\n", 'embed_hat: ', embed_hat)
        
        total_loss = classifier_loss + embed_loss
        
        if self.enable_kl_loss:
            # KL divergence from N(0, 1)
            if labels_mask is None:
                kl_loss = (0.5 * (sigma * sigma + mu * mu - 1 - 2 * torch.log(sigma))).mean()
            else:
                # disable kl_loss when all labels are disabled
                mask = torch.where(torch.sum(labels_mask, -1) > 0, 1, 0)
                kl_loss = (mask * 0.5 * (sigma * sigma + mu * mu - 1 - 2 * torch.log(sigma))).mean()
            total_loss += kl_loss
        else:
            kl_loss = torch.zeros(1, device=embed.device)
        
        if self.enable_embed_loss:
            embed_loss += self.cosine_loss(embed_hat, labels, labels_mask)
            total_loss += embed_loss
        
        return total_loss, classifications, embed_hat, mu, sigma, kl_loss, embed_loss, classifier_loss

class ClassQueryEmbeddingLoss(nn.Module):
    def __init__(self):
        super(ClassQueryEmbeddingLoss, self).__init__()
    
    def forward(self, query, embedding, targets):
        # query has size: C x E; number of classes x embedding length
        # embedding has size: B x C x E; batch size x number of classes x embedding length
        # targates has size: B x C; batch size x number of classes
        
        # normalize each query vector then reshape query to C x E x 1 (each E x 1 is a normalized vector)
        query_normalized = torch.nn.functional.normalize(query, dim=-1).unsqueeze(-1)
        # normalize each embedding vector then reshape embedding to C x B x E
        embedding_normalized = torch.nn.functional.normalize(embedding, dim=-1).permute(1, 0, 2)
        # queries x embeddings cosine similarity;
        # for each instance in the batch, for each class
        # compute cosine similarity between query and embedding
        # (C x B x 1) = (C x B x E) . (C x E x 1)
        cosine_similiarity = torch.matmul(embedding_normalized, query_normalized)
        # remove last redundant dimension and trasnpose to get batch-first shape B x C
        cosine_similiarity = cosine_similiarity.squeeze(-1).T
        # re-scale from [-1, 1] to [0, 1]
        cosine_similiarity = 0.5 * (1 + cosine_similiarity)
        loss = targets * (1 - cosine_similiarity) + (1 - targets) * cosine_similiarity
        return loss.mean()
    
class TaskMultiplexedClassifier(nn.Module):
    def __init__(
        self,
        data_config,
        enable_decoder,
        decoder_nhead,
        decoder_num_layers,
        d_model,
        dropout_prob,
        classifier_hidden_size,
        classifier_layers_count,
        # classifier_per_class,
        enable_kl_loss,
        enable_embed_loss,
        negatives_are_close,
        enable_class_query_embedding_loss
    ):
        super(TaskMultiplexedClassifier, self).__init__()
        
        self.tasks = []
        for d in data_config: self.tasks.extend(d['tasks'])

        self.enable_decoder = enable_decoder
        classifier_per_class = False
        self.classifier_per_class = classifier_per_class
        
        self.enable_class_query_embedding_loss = enable_class_query_embedding_loss
        
        # tasks_classes_queries = {}
        tasks_classifiers = {}
        
        for task in self.tasks:
            task_num_output_classes = len(task['labels'])
            # if enable_decoder: tasks_classes_queries[task['name']] = nn.Embedding(task_num_output_classes, d_model)
            task_pos_weight = torch.tensor(task['pos_weight']) if task.get('pos_weight') is not None else None
            if classifier_per_class:
                tasks_classifiers[task['name']] = nn.ModuleDict(dict([
                    (
                        label,
                        # 1 head classifier for each class, each has own parameters
                        VariationalClassifier(d_model, classifier_hidden_size, 1, dropout_prob, classifier_layers_count, enable_kl_loss, enable_embed_loss,
                                              negatives_are_close, vector_per_class=enable_decoder, pos_weight=task_pos_weight)
                    ) for label in task['labels']
                ]))
            else:
                # when enable_decoder=True : 1 head classifier for each class, all share the same parameters
                # when enable_decoder=False: 1 multi head classifier for all classes
                tasks_classifiers[task['name']] = VariationalClassifier(d_model, classifier_hidden_size, 1 if enable_decoder else task_num_output_classes,
                                                                        dropout_prob, classifier_layers_count, enable_kl_loss, enable_embed_loss,
                                                                        negatives_are_close, vector_per_class=enable_decoder, pos_weight=task_pos_weight)

        if enable_decoder:
            # decoder part
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=decoder_nhead, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_num_layers)

            labels = []
            for t in self.tasks: labels.extend(t['labels'])
            labels = list(set(labels))
            self.labels_to_idx = dict([(l, i) for i, l in enumerate(labels)])
            self.classes_queries = nn.Embedding(len(labels), d_model)
            # self.tasks_classes_queries = nn.ModuleDict(tasks_classes_queries)
            if enable_class_query_embedding_loss: self.class_query_embedding_loss = ClassQueryEmbeddingLoss()
        else:
            self.dropout = nn.Dropout(dropout_prob)
        
        self.tasks_classifiers = nn.ModuleDict(tasks_classifiers)
        
    def embeddings_for_task(self, encoder_output, encoder_output_mask, task):
        if self.enable_decoder:
            device = encoder_output.device
            batch_size = encoder_output.size(0)
            tasks_embds = {}
            sequence_output = encoder_output

            classes_queries_ids = torch.tensor([self.labels_to_idx[l] for l in task['labels']], dtype=torch.long, device=device)
            classes_queries_emb = self.classes_queries(classes_queries_ids.repeat(batch_size, 1))
            # pytorch expects True for positions to be ignored and False ow
            # However, in input, 1 positions shouldn't be ignored while 0s should be
            # So, we invert the mask
            classes_queries_emb_hat = self.decoder(classes_queries_emb, sequence_output, memory_key_padding_mask=encoder_output_mask < 1)

            return classes_queries_emb_hat
        else:
            pooled_output = encoder_output
            return self.dropout(pooled_output)
        
    def embeddings(self, encoder_output, encoder_output_mask):
        """
        Returns embeddings per task
        """
        return dict([(task['name'], self.embeddings_for_task(encoder_output, encoder_output_mask, task)) for task in self.tasks])
        
    def embeddings_with_loss(self, encoder_output, encoder_output_mask, labels):
        """
        Returns embeddings per task
        """
        device = encoder_output.device
        tasks_embds = {}
        for task in self.tasks:
            task_embd = self.embeddings_for_task(encoder_output, encoder_output_mask, task)
            loss = None
            if self.enable_decoder and self.enable_class_query_embedding_loss:
                classes_queries_ids = torch.tensor([self.labels_to_idx[l] for l in task['labels']], dtype=torch.long, device=device)
                classes_queries_emb = self.classes_queries(classes_queries_ids)
                loss = self.class_query_embedding_loss(
                    self.classes_queries(classes_queries_ids),
                    task_embd,
                    labels[task['name']]
                )
            else:
                loss = torch.zeros(1, device=device)
            tasks_embds[task['name']] = task_embd, loss
        return tasks_embds

        
    def forward(self, encoder_output, encoder_output_mask):
        tasks_embds = self.embeddings(encoder_output, encoder_output_mask)
        
        output = {}
        for task in self.tasks:
            emb = tasks_embds[task['name']]
            classifier = self.tasks_classifiers[task['name']]
            classifications, embed_hat, mu, sigma = classifier(emb)
            # print('classifications: ' + str(classifications.size()))
            output[task['name']] = classifications, emb, embed_hat, mu, sigma
        
        return output
    
    def forward_with_loss(self, encoder_output, encoder_output_mask, labels):
        tasks_embds = self.embeddings_with_loss(encoder_output, encoder_output_mask, labels)
        
        output = {}
        total_loss = None
        n = 0
        for task_name, task_labels in labels.items():
            emb, query_embd_loss = tasks_embds[task_name]
            classifier = self.tasks_classifiers[task_name]
            task_total_loss, classifications, embed_hat, mu, sigma, kl_loss, embd_loss, classifier_loss = classifier.forward_with_loss(emb, task_labels)
            # print('classifications: ' + str(classifications.size()))
            task_total_loss += query_embd_loss
            output[task_name] = classifications, emb, embed_hat, mu, sigma, classifier_loss, kl_loss, embd_loss, query_embd_loss, task_total_loss
            n += 1
            total_loss = task_total_loss if total_loss is None else (total_loss + task_total_loss)
        
        if n > 0: total_loss /= n

        return output, total_loss

class MAMIClassifierV16(nn.Module):
    """
    data_config = [
        {
            "name": "MAMI",
            "df": MAMI_df,
            "labels": ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence'],
            "labels_mask": [],
            "tasks": [
                {"name": "Task A", "labels": ['misogynous'], "labels_index": []},
                {"name": "Task B", "labels": ['shaming', 'stereotype', 'objectification', 'violence'], "labels_index": []}
            ]
        },
        {
            "name": "fb_hateful_meme",
            "df": fb_hateful_meme_df,
            "labels": ['hateful'],
            "labels_mask": [],
            "tasks": [
                {"name": "Hateful Meme", "labels": ['hateful'], "labels_index": []}
            ]
        }
    ]
    """
    
    def __init__(
        self,
        bert_model_name,
        data_config=[],
        tokenizer=None,
        finetune_txt=False,
        enable_kl_loss=True,
        enable_embed_loss=False,
        negatives_are_close=False,
        classifier_layers_count=4,
        classifier_hidden_size=512,
        share_transformer_encoder=True,
        pool_txt=False,
        modal_transformer_encoder_nhead=8,
        modal_transformer_encoder_num_layers=6,
        pool_output=False,
        decoder_nhead=8,
        decoder_num_layers=6,
        image_encoders=['clip', 'detr'],
        image_encoders_finetune=[None, None],
        clip_num_patches=(2, 2),
        detr_fallback_topk=4,
        projection_alignment=False,
        enable_class_query_embedding_loss=False
    ):
        super(MAMIClassifierV16, self).__init__()
        
        transformer_config = AutoConfig.from_pretrained(bert_model_name) 
        transformer = AutoModel.from_pretrained(bert_model_name, config=transformer_config)
        img_encoders = []
        for i, e in enumerate(image_encoders):
            finetune_img = image_encoders_finetune[i]
            if finetune_img is not None:
                if   e == 'clip': img_encoders.append(CLIPImageEmbeddingLayer(clip_num_patches, finetune_img))
                elif e == 'detr': img_encoders.append(DETRImageEmbeddingLayer(detr_fallback_topk, finetune_img))
            else:
                if   e == 'clip': img_encoders.append(CLIPFeaturesImageEmbeddingLayer())
                elif e == 'detr': img_encoders.append(DETRFeaturesImageEmbeddingLayer())
        if tokenizer == None:
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
        
        self.cls_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
                
        config = MMBTConfig(transformer_config, num_labels=1)# num_labels don't care; not used
        # self.hidden_size = config.hidden_size
        
        self.share_transformer_encoder = share_transformer_encoder

        if share_transformer_encoder:
            self.mmxt = MMSTModel(
                config,
                transformer,
                [(e, e.embed_length) for e in img_encoders],
                start_token_id=self.cls_token_id,
                end_token_id=self.sep_token_id,
                pool_output=pool_output,
                projection_alignment=projection_alignment
            )
        else:
            self.mmxt = MMMTModel(
                config,
                transformer,
                [(e, e.embed_length) for e in img_encoders],
                start_token_id=self.cls_token_id,
                pool_txt=pool_txt,
                transformer_nhead=modal_transformer_encoder_nhead,
                transformer_num_layers=modal_transformer_encoder_num_layers,
                projection_alignment=projection_alignment
            )

        self.enable_decoder = not pool_output or not share_transformer_encoder

        self.classifier = TaskMultiplexedClassifier(
            data_config,
            self.enable_decoder,
            decoder_nhead,
            decoder_num_layers,
            config.hidden_size,
            config.hidden_dropout_prob,
            classifier_hidden_size,
            classifier_layers_count,
            enable_kl_loss,
            enable_embed_loss,
            negatives_are_close,
            enable_class_query_embedding_loss
        )

        
        set_parameter_requires_grad(transformer, finetune_txt)
        #set_parameter_requires_grad(transformer.embeddings.word_embeddings, False)
        
    def embeddings(self, img, img_kwargs, txt_ids, txt_mask, txt_token_type_ids):                        
        outputs = self.mmxt(input_modal=img, input_ids=txt_ids, attention_mask=txt_mask, modal_encoders_kwargs=img_kwargs)
        return outputs
    
    def forward(self, img, img_kwargs, txt_ids, txt_mask, txt_token_type_ids):
        emb, mask, _ = self.embeddings(img, img_kwargs, txt_ids, txt_mask, txt_token_type_ids)
        return self.classifier(emb, mask)
    
    def forward_with_loss(self, img, img_kwargs, txt_ids, txt_mask, txt_token_type_ids, labels):
        emb, mask, projection_loss = self.embeddings(img, img_kwargs, txt_ids, txt_mask, txt_token_type_ids)
        output, total_loss = self.classifier.forward_with_loss(emb, mask, labels)
        total_loss += projection_loss
        return output, projection_loss, total_loss
