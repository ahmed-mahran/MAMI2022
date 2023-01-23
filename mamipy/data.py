# Copyright 2022 Ahmed Mahran. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Sampler, RandomSampler, SequentialSampler

from PIL import Image
from torchvision import transforms


from sklearn.model_selection import train_test_split

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


ROOT_PATH = '/content/data'
# ROOT_PATH = '/home/shared/users/ahmed-mahran/MAMI/data'

def data_root_path(name='MAMI', is_train=True):
    tag = 'train' if is_train else 'test'
    return f"{ROOT_PATH}/{name}/{tag}/"

def load_data(name='MAMI', is_train=True):
    df = pd.read_csv(data_root_path(name, is_train) + 'list.csv', sep='\t')
    df.rename(columns={'Text Transcription':'text_transcription'}, inplace=True)
    if is_train:
      if name == 'MAMI':
          # remove bad data (only 4 rows)
          df = df.drop(df[(df['misogynous'] == 0) & (df[['shaming', 'stereotype', 'objectification', 'violence']].sum(axis=1) > 0)].index)
      elif name == 'fb_hateful_meme':
          df.rename(columns={'label':'hateful'}, inplace=True)
    return df

def image_path(file_name, name='MAMI', is_train=True):
    return data_root_path(name, is_train) + "data/" + file_name

def slice_image(image, patches=(2, 2)):
    """
    Slice an image tensor into nxm patches
    """
    #torch.Tensor.unfold(dimension, size, step)
    #slices the images into patches[0]*patches[1] number of patches
    
    patch_size = [l // p for l, p in zip(image[0].size(), patches)]
    # display('patch_size: ' + str(patch_size))
    
    patches = image.unfold(0, 3, 1)
    for i, lp in enumerate(patch_size):
        patches = patches.unfold(i + 1, lp, lp)
        
    return patches.squeeze(0)

def image_with_slices(image, patches=(2, 2)):
    """
    Slice an image tensor into an nxm patches returing the original image as well
    as the first element into the sequence of patches.
    """
    return [image, *slice_image(image, patches).flatten(0, 1).unbind()]

def read_detr_features(file_name, name='MAMI', is_train=True):
    root_path = data_root_path(name, is_train) + 'detr_features'
    img_name, img_type = tuple(file_name.split('.'))
    features = torch.load(f'{root_path}/{img_name}_tensor.pt').type(torch.float32)
    logits = torch.load(f'{root_path}/{img_name}_logits.pt').type(torch.float32)
    return features, logits

def read_clip_features(file_name, name='MAMI', is_train=True):
    root_path = data_root_path(name, is_train)
    img_name, img_type = tuple(file_name.split('.'))
    img_features = torch.load(f'{root_path}clip_features/{img_name}_tensor.pt')
    img_features.requires_grad = False
    slices_features = torch.load(f'{root_path}slices_2x2_clip_features/{img_name}_tensor.pt')
    features = torch.cat([img_features, slices_features]).type(torch.float32)
    return features


class MAMIDataset(Dataset):
    """
    tasks = [
        {'name': '', 'labels': []}
    ]
    """
    
    def __init__(
        self, data, image_transform, tokenizer, max_tokens=80, patches=(2, 2),
        name='MAMI', image_col='file_name', text_col='text_transcription', tasks=[],
        is_train=True, image_encoders=['clip', 'detr']
        ):
        self.tokenizer = tokenizer
        self.data = data
        self.name = name
        self.tasks = tasks
        self.texts = [x.lower() for x in data[text_col].tolist()]
        self.images = data[image_col].tolist()
        if True or is_train:
            self.targets = {}
            for task in tasks:
                self.targets[task['name']] = data.loc[:, task['labels']].values.tolist()
        self.max_tokens = max_tokens
        self.image_transform = image_transform
        self.texts_length = len(self.texts)
        self.patches = patches
        self.to_tensor = transforms.ToTensor() #transforms.PILToTensor()
        self.is_train = is_train
        self.image_encoders = set(image_encoders)

        self.image_transform = transforms.Compose([
            transforms.Resize(size=512, interpolation=transforms.functional.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.CenterCrop(size=512)
            #transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
    def __len__(self):
        return self.texts_length

    def __getitem__(self, index):
        img_file_name = self.images[index]
        image = self.to_tensor(Image.open(image_path(img_file_name, self.name, self.is_train)).convert('RGB'))
        if self.image_transform != None:
            image = self.image_transform(image)
        ##images = image_with_slices(image, patches=self.patches)
        ##images = torch.stack([self.image_transform(image) for image in images])

        text = str(self.texts[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_tokens + 1, # + 1 for [CLS] token
            return_token_type_ids=True,
            padding='max_length',
            truncation=True
        )
        res = {
            'index': torch.tensor(index),
            
            'img': image, #images,
            
            # MMBT concatenates in this order embedding_output = torch.cat([modal_embeddings, txt_embeddings], 1)
            # so final embedding looks like:
            # [CLS] img_patch_1 ... img_patch_n [SEP] txt_token_1 ... txt_token_m [SEP] [PAD] ... [PAD]
            # just remove the start token [CLS] as it will be added by the model
            'txt_ids': torch.tensor(inputs['input_ids'][1:], dtype=torch.long),
            'txt_mask': torch.tensor(inputs['attention_mask'][1:], dtype=torch.long),
            'txt_token_type_ids': torch.tensor(inputs["token_type_ids"][1:], dtype=torch.long),
        }

        if 'clip' in self.image_encoders:
            res['clip_features'] = read_clip_features(img_file_name, self.name, self.is_train)

        if 'detr' in self.image_encoders:
            detr_features, detr_logits = read_detr_features(img_file_name, self.name, self.is_train)
            res['detr_features'] = detr_features
            res['detr_logits'] = detr_logits

        if True or self.is_train:
            for task in self.tasks:
                task_name = task['name']
                labels = self.targets[task_name][index]
                res[f'{task_name}_targets'] = torch.tensor(labels, dtype=torch.float)

        return res

class MultiplexedBatchSampler(Sampler[List[int]]):
    r"""Wraps other samplers to yield a mini-batch of indices such that a single mini-batch
    contains elements from one sampler.
    """
    
    def __init__(self, samplers: List[Sampler[int]], batch_size: int, drop_last: bool, generator=None) -> None:
        self.samplers = samplers
        self.batch_size = batch_size
        self.drop_last = drop_last
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        active_samplers_indicies = [i for i in range(len(self.samplers))]
        samplers_iters = [iter(sampler) for sampler in self.samplers]
        
        samplers_base_index, s = [0], 0
        for i in range(len(self.samplers) - 1):
            s += len(self.samplers[i])
            samplers_base_index.append(s)
        
        while len(active_samplers_indicies) > 0:
            active_sampler_idx = torch.randint(low=0, high=len(active_samplers_indicies), size=(1,), generator=self.generator).item()
            sampler_idx = active_samplers_indicies[active_sampler_idx]
            sampler = samplers_iters[sampler_idx]
            sampler_base_index = samplers_base_index[sampler_idx]
            
            for i in range(self.batch_size):
                idx = next(sampler, None)
                if idx is not None:
                    batch.append(sampler_base_index + idx)
                else:
                    # end of this sampler
                    del active_samplers_indicies[active_sampler_idx]
                    break
            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                yield batch
            batch = []

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return sum([len(sampler) // self.batch_size for sampler in self.samplers])  # type: ignore[arg-type]
        else:
            return sum([(len(sampler) + self.batch_size - 1) // self.batch_size for sampler in self.samplers])  # type: ignore[arg-type]

def prepare(data_config, test_data_config, tokenizer, max_tokens=80, patches=(2, 2), image_size=288,
  limit=None, train_frac=0.8, train_batch_size=16, eval_batch_size=16, image_encoders=['clip', 'detr']):

    """
    data_config = [
        {
            "name": "MAMI",
            "df": MAMI_df,
            "labels": ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence'],
            "tasks": [
                {"name": "Task A", "labels": ['misogynous']},
                {"name": "Task B", "labels": ['shaming', 'stereotype', 'objectification', 'violence']}
            ]
        },
        {
            "name": "fb_hateful_meme",
            "df": fb_hateful_meme_df,
            "labels": ['hateful'],
            "tasks": [
                {"name": "Hateful Meme", "labels": ['hateful']}
            ]
        }
    ]
    """
    
    # CLIP transforms
    image_transform = transforms.Compose([
        transforms.Resize(size=image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=image_size)
        #transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    
    train_datasets = []
    eval_datasets = []
    
    train_samplers = []
    eval_samplers = []
        
    for data in data_config:
        df = data['df']
        df_size = df.shape[0]
        train_size = train_frac if limit is None else int(min(df_size, limit) * train_frac)
        test_size = (1 if limit is None else min(df_size, limit)) - train_size
    
        train_df, eval_df = train_test_split(df, test_size=test_size, train_size=train_size, random_state=3704, stratify=df[data['labels']])
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)
        
        for task in data['tasks']:
            # neg / pos
            task['pos_weight'] = [train_df[train_df[c] == 0].shape[0] / train_df[train_df[c] == 1].shape[0] for c in task['labels']]

        if data.get('disabled', False):
          print(f'Data {data["name"]} is disabled')
          train_df = train_df.sample(n=train_batch_size, random_state=3704)
          eval_df = eval_df.sample(n=eval_batch_size, random_state=3704)
        
        train_dataset = MAMIDataset(train_df, image_transform, tokenizer, max_tokens=max_tokens, patches=patches,
                                    name=data['name'], tasks=data['tasks'], is_train=True, image_encoders=image_encoders)
        eval_dataset = MAMIDataset(eval_df, image_transform, tokenizer, max_tokens=max_tokens, patches=patches,
                                   name=data['name'], tasks=data['tasks'], is_train=True, image_encoders=image_encoders)
        
        data['train_dataset'] = train_dataset
        data['eval_dataset'] = eval_dataset
        
        train_datasets.append(train_dataset)
        eval_datasets.append(eval_dataset)
        
        train_samplers.append(RandomSampler(train_dataset))
        eval_samplers.append(SequentialSampler(eval_dataset))
        

    test_datasets = []
    test_samplers = []
        
    for data in test_data_config:
        df = data['df']
        test_dataset = MAMIDataset(df, image_transform, tokenizer, max_tokens=max_tokens, patches=patches,
                                   name=data['name'], tasks=data['tasks'], is_train=False, image_encoders=image_encoders)
        data['test_dataset'] = test_dataset
        test_datasets.append(test_dataset)
        test_samplers.append(SequentialSampler(test_dataset))
        
    train_dataset = ConcatDataset(train_datasets)
    eval_dataset = ConcatDataset(eval_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    train_batch_sampler = MultiplexedBatchSampler(train_samplers, train_batch_size, drop_last=False, generator=None)
    eval_batch_sampler = MultiplexedBatchSampler(eval_samplers, eval_batch_size, drop_last=False, generator=None)
    test_batch_sampler = MultiplexedBatchSampler(test_samplers, eval_batch_size, drop_last=False, generator=None)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers = 2
    )

    eval_loader = DataLoader(
        eval_dataset, 
        batch_sampler=eval_batch_sampler,
        num_workers = 2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_sampler=test_batch_sampler,
        num_workers = 2
    )

    return train_loader, eval_loader, data_config, test_loader, test_data_config

def prepare_eval(df, tokenizer, max_tokens=80, patches=(2, 2), image_size=288,
    limit=None, frac=None, batch_size=16, image_encoders=['clip', 'detr']):
    # CLIP transforms
    image_transform = transforms.Compose([
        transforms.Resize(size=image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=image_size)
        #transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
        
    limited_df = df.sample(n=limit, frac=frac) if limit != None or frac != None else df
    
    tasks = [
                {"name": "Task A", "labels": ['misogynous']},
                {"name": "Task B", "labels": ['shaming', 'stereotype', 'objectification', 'violence']}
            ]

    dataset = MAMIDataset(limited_df, image_transform, tokenizer, max_tokens=max_tokens, patches=patches, name='MAMI', tasks=tasks, is_train=False, image_encoders=image_encoders)

    sampler = SequentialSampler(dataset)


    loader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=batch_size,
        num_workers = 2
    )

    
    return loader