from pathlib import Path
from tqdm.notebook import tqdm

from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import time
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

from . import scores
from .scores import (compute_scoreA, compute_scoreB)

def save_checkpoint(save_path, model, state_dict):

    if save_path == None:
        return
    
    full_state_dict = {
        'model_state_dict': model.state_dict(),
        'state_dict': state_dict
    }
    
    torch.save(full_state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
    
def load_checkpoint(load_path, model, device=None):
    
    if load_path==None:
        return
    
    full_state_dict = torch.load(load_path, map_location=device) if device is not None else torch.load(load_path)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(full_state_dict['model_state_dict'])
    return full_state_dict['state_dict']

def save_predictions(path, dataset, indicies, labels, preds, probas):
    images = [dataset.images[i] for i in indicies]
    df = pd.DataFrame({'image': images, 'label': labels, 'pred': preds, 'proba': probas}, columns=['image', 'label', 'pred', 'proba'])
    df.to_csv(path, encoding='utf-8', index=False)
    return df

def time_elapsed_as_str(since):
    time_elapsed = time.time() - since
    minutes_part_elapsed = time_elapsed // 60
    seconds_part_elapsed = time_elapsed % 60
    return f"{minutes_part_elapsed}m {seconds_part_elapsed}s"

from google.protobuf import text_format
from tensorboard.plugins.projector.projector_config_pb2 import (EmbeddingInfo, ProjectorConfig)
class TbEmbeddingTracker():    
    def __init__(self, path):
        self.path = path
        self.projector_config = ProjectorConfig()
    
    def tensor_name(self, tag, global_step):
        return "{}:{}".format(tag, str(global_step).zfill(5))
    
    def encode(self, rawstr):
        retval = rawstr
        retval = retval.replace("%", "%%%02x" % (ord("%")))
        retval = retval.replace("/", "%%%02x" % (ord("/")))
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))
        return retval
    
    def resume(self):
        config_file = f"{self.path}projector_config.pbtxt"
        if Path(config_file).is_file():
            with open(config_file, 'r') as f:
                self.projector_config = text_format.Parse(f.read(), ProjectorConfig())
        
    def append_embedding(self, mat, metadata, global_step, tag):
        sub_dir = f"{str(global_step).zfill(5)}/{self.encode(tag)}"
        sub_dir_abs = f"{self.path}{sub_dir}"
        Path(sub_dir_abs).mkdir(parents=True, exist_ok=True)
        
        # append to tensor file
        with open(f"{sub_dir_abs}/tensors.tsv", 'a') as f:
            for x in mat:
                x = [str(i.item()) for i in x]
                f.write('\t'.join(x) + '\n')
        
        # write metadata
        with open(f"{sub_dir_abs}/metadata.tsv", 'a') as f:
            f.write('\n'.join([str(x) for x in metadata]) + '\n')
        
        # write projector_config.pbtxt
        tensor_config_exists = False
        tensor_config_name = self.tensor_name(tag, global_step)
        for config in self.projector_config.embeddings:
            if config.tensor_name == tensor_config_name:
                tensor_config_exists = True
                break
        if not tensor_config_exists:
            info = EmbeddingInfo()
            info.tensor_name = tensor_config_name
            info.tensor_path = f"{sub_dir}/tensors.tsv"
            if metadata is not None:
                info.metadata_path = f"{sub_dir}/metadata.tsv"
            self.projector_config.embeddings.extend([info])
            with open(f"{self.path}projector_config.pbtxt", 'w') as f:
                f.write(text_format.MessageToString(self.projector_config))
    
class BatchForward():
    def __init__(self, model, device, data_config, threshold, emb_tracker, writer, log, tag):
        self.model = model
        self.device = device
        self.data_config = data_config
        self.threshold = threshold
        self.emb_tracker = emb_tracker
        self.writer = writer
        self.log = log
        self.tag = tag
        
        self.eval_loss = 0.0
        self.eval_classifier_loss = 0.0
        self.eval_kl_loss = 0.0
        self.eval_embedding_loss = 0.0
        self.eval_projection_loss = 0.0
        self.eval_query_embedding_loss = 0.0
        self.nb_eval_steps = 0
        
        self.task_eval_loss = {}
        self.task_eval_classifier_loss = {}
        self.task_eval_kl_loss = {}
        self.task_eval_embedding_loss = {}
        self.task_eval_projection_loss = {}
        self.task_eval_query_embedding_loss = {}
        self.task_nb_eval_steps = {}
        
        self.preds = {}
        self.proba = {}
        self.out_label_ids = {}
        self.indicies = {}
                
        for data in data_config:
            for task in data['tasks']:
                task_name = task['name']
                self.task_eval_loss[task_name] = 0.0
                self.task_eval_classifier_loss[task_name] = 0.0
                self.task_eval_kl_loss[task_name] = 0.0
                self.task_eval_embedding_loss[task_name] = 0.0
                self.task_eval_projection_loss[task_name] = 0.0
                self.task_eval_query_embedding_loss[task_name] = 0.0
                self.task_nb_eval_steps[task_name] = 0
                self.preds[task_name] = []
                self.proba[task_name] = []
                self.out_label_ids[task_name] = []
                self.indicies[task_name] = []
        
    def forward(self, batch, step, global_step, dump_embeddings=False, log_and_reset_loss=False):
        device = self.device
        # batch = tuple(t.to(device) for t in batch)
        index = batch['index']
        img = batch['img'].to(device)
        img_kwargs = []

        if batch.get(f'clip_features') is not None:
            img_kwargs.append({'clip_features': batch['clip_features'].to(device)})

        if batch.get(f'detr_features') is not None:
            img_kwargs.append({
                'detr_features': batch['detr_features'].to(device),
                'detr_logits': batch['detr_logits'].to(device)
            })

        txt_ids = batch['txt_ids'].to(device)
        txt_mask = batch['txt_mask'].to(device)
        txt_token_type_ids = batch['txt_token_type_ids'].to(device)
        
        labels = {}
        for data in self.data_config:
            for task in data['tasks']:
                task_name = task['name']
                if batch.get(f'{task_name}_targets') is not None:
                    labels[task_name] = batch[f'{task_name}_targets'].to(device)
        
        output, projection_loss, total_loss = self.model.forward_with_loss(img, img_kwargs, txt_ids, txt_mask, txt_token_type_ids, labels)
        
        n_tasks = len(output)
        for task, task_output in output.items():
        
            logits, emb, embed_hat, mu, sigma, classifier_loss, kl_loss, embd_loss, query_embd_loss, task_total_loss = task_output
            classifications = torch.sigmoid(logits)
            
            self.eval_classifier_loss += classifier_loss.item() / n_tasks
            self.eval_kl_loss += kl_loss.item() / n_tasks
            self.eval_embedding_loss += embd_loss.item() / n_tasks
            self.eval_query_embedding_loss += query_embd_loss.item() / n_tasks
            
            self.task_eval_loss[task] += task_total_loss.item() + projection_loss.item()
            self.task_eval_classifier_loss[task] += classifier_loss.item()
            self.task_eval_kl_loss[task] += kl_loss.item()
            self.task_eval_embedding_loss[task] += embd_loss.item()
            self.task_eval_projection_loss[task] += projection_loss.item()
            self.task_eval_query_embedding_loss[task] += query_embd_loss.item()
            self.task_nb_eval_steps[task] += 1
            
            self.indicies[task].extend(index.detach().cpu().numpy().tolist())
            self.preds[task].extend((classifications.detach().cpu().numpy() > self.threshold).tolist())
            self.proba[task].extend(classifications.detach().cpu().numpy().tolist())
            labels_list = labels[task].detach().cpu().numpy().tolist()
            self.out_label_ids[task].extend(labels_list)
        
        self.eval_projection_loss += projection_loss.item()
        self.eval_loss += total_loss.item()
        self.nb_eval_steps += 1

        #if dump_embeddings:
        #    tag = 'train' if self.model.training else 'test'
        #    if len(self.classes) > 1:
        #        tag = None
        #    else:
        #        self.emb_tracker.append_embedding(emb, labels_list, step, f'{tag}_emb')
        #        self.emb_tracker.append_embedding(emb_hat, labels_list, step, f'{tag}_emb_hat')
            
        if log_and_reset_loss:
            self.log_and_reset_loss(global_step)
        
        return output, total_loss

    def forward_eval(self, batch):
        device = self.device
        # batch = tuple(t.to(device) for t in batch)
        index = batch['index']
        img = batch['img'].to(device)
        img_kwargs = []

        if batch.get(f'clip_features') is not None:
            img_kwargs.append({'clip_features': batch['clip_features'].to(device)})

        if batch.get(f'detr_features') is not None:
            img_kwargs.append({
                'detr_features': batch['detr_features'].to(device),
                'detr_logits': batch['detr_logits'].to(device)
            })

        txt_ids = batch['txt_ids'].to(device)
        txt_mask = batch['txt_mask'].to(device)
        txt_token_type_ids = batch['txt_token_type_ids'].to(device)

        output = self.model.forward(img, img_kwargs, txt_ids, txt_mask, txt_token_type_ids)
        
        for task, task_output in output.items():
            if self.indicies.get(task) is not None:
                logits, embed_hat, mu, sigma = task_output
                classifications = torch.sigmoid(logits)
                self.indicies[task].extend(index.detach().cpu().numpy().tolist())
                self.preds[task].extend((classifications.detach().cpu().numpy() > self.threshold).tolist())
                self.proba[task].extend(classifications.detach().cpu().numpy().tolist())
            
        return output
        
    def log_and_reset_loss(self, global_step):        
        max_task_name = 1
        for data in self.data_config:
            for task in data['tasks']:
                max_task_name = max(max_task_name, len(task['name']))

        def for_task(task=None):
            nb_eval_steps = self.nb_eval_steps if task is None else self.task_nb_eval_steps[task]
            if nb_eval_steps != 0:
                if task is None:
                    tag1 = ''
                    tag2 = ''
                    eval_loss = self.eval_loss
                    eval_classifier_loss = self.eval_classifier_loss
                    eval_kl_loss = self.eval_kl_loss
                    eval_embedding_loss = self.eval_embedding_loss
                    eval_projection_loss = self.eval_projection_loss
                    eval_query_embedding_loss = self.eval_query_embedding_loss
                    self.eval_loss = 0.0
                    self.eval_classifier_loss = 0.0
                    self.eval_kl_loss = 0.0
                    self.eval_embedding_loss = 0.0
                    self.eval_projection_loss = 0.0
                    self.eval_query_embedding_loss = 0.0
                    self.nb_eval_steps = 0
                else:
                    tag1 = task + '/'
                    tag2 = task
                    eval_loss = self.task_eval_loss[task]
                    eval_classifier_loss = self.task_eval_classifier_loss[task]
                    eval_kl_loss = self.task_eval_kl_loss[task]
                    eval_embedding_loss = self.task_eval_embedding_loss[task]
                    eval_projection_loss = self.task_eval_projection_loss[task]
                    eval_query_embedding_loss = self.task_eval_query_embedding_loss[task]
                    self.task_eval_loss[task] = 0.0
                    self.task_eval_classifier_loss[task] = 0.0
                    self.task_eval_kl_loss[task] = 0.0
                    self.task_eval_embedding_loss[task] = 0.0
                    self.task_eval_projection_loss[task] = 0.0
                    self.task_eval_query_embedding_loss[task] = 0.0
                    self.task_nb_eval_steps[task] = 0
                
                eval_loss /= nb_eval_steps
                eval_classifier_loss /= nb_eval_steps
                eval_kl_loss /= nb_eval_steps
                eval_embedding_loss /= nb_eval_steps
                eval_projection_loss /= nb_eval_steps
                eval_query_embedding_loss /= nb_eval_steps


                self.writer.add_scalar(f'Loss/{tag1}{self.tag}', eval_loss, global_step=global_step)
                self.writer.add_scalar(f'Classifier Loss/{tag1}{self.tag}', eval_classifier_loss, global_step=global_step)
                self.writer.add_scalar(f'KL Loss/{tag1}{self.tag}', eval_kl_loss, global_step=global_step)
                self.writer.add_scalar(f'Embdedding Loss/{tag1}{self.tag}', eval_embedding_loss, global_step=global_step)
                self.writer.add_scalar(f'Projection Loss/{tag1}{self.tag}', eval_projection_loss, global_step=global_step)
                self.writer.add_scalar(f'Class Query Embdedding Loss/{tag1}{self.tag}', eval_query_embedding_loss, global_step=global_step)

                self.log(f"[{self.tag}] [{global_step}] [{tag2:>{max_task_name}}] average losses for last {nb_eval_steps:>2} steps: classifier={eval_classifier_loss:.4f}, kl={eval_kl_loss:.4f}, embedding={eval_embedding_loss:.4f}, projection={eval_projection_loss:.4f}, class query embedding={eval_query_embedding_loss:.4f}, total={eval_loss:.4f}")
            
        for_task(None)
        for data in self.data_config:
            for task in data['tasks']:
                for_task(task['name'])
    
    def save_predictions(self, path_prefix, suffix):
        for data in self.data_config:
            dataset = data[f'{self.tag}_dataset']
            for task in data['tasks']:
                task_name = task['name']
                save_predictions(f'{path_prefix}preds_{task_name}_{self.tag}_{suffix}.csv', dataset,
                                 self.indicies[task_name], self.out_label_ids[task_name], self.preds[task_name], self.proba[task_name])
    
    def accuracy(self, global_step, path_prefix, preds_suffix='last'):
        self.log_and_reset_loss(global_step)
        
        result = {
            "global_step": global_step,
            "loss": self.eval_loss,
            "classifier_loss": self.eval_classifier_loss,
            "kl_loss": self.eval_kl_loss,
            "embedding_loss": self.eval_embedding_loss,
            "projection_loss": self.eval_projection_loss,
            "query_embedding_loss": self.eval_query_embedding_loss,
            "tasks": {}
        }
        
        max_task_name = 1
        for data in self.data_config:
            for task in data['tasks']:
                max_task_name = max(max_task_name, len(task['name']))

        for data in self.data_config:
            if data.get('disabled', False): continue
            for task in data['tasks']:
                task_name = task['name']
                classes = task['labels']

                out_label_ids = self.out_label_ids[task_name]
                preds = self.preds[task_name]
                proba = self.proba[task_name]
                indicies = self.indicies[task_name]
                
                acc          = accuracy_score(out_label_ids, preds)
                auc_macro    = roc_auc_score(out_label_ids, proba, average="macro")
                auc_weighted = roc_auc_score(out_label_ids, proba, average="weighted")
                f1_macro     = f1_score(out_label_ids, preds, average="macro")
                f1_weighted  = f1_score(out_label_ids, preds, average="weighted")
                f1_s         = f1_score(out_label_ids, preds, average=None)
                if len(classes) == 1:
                    task_score_name = 'scoreA'
                    task_score = compute_scoreA(out_label_ids, preds)
                else:
                    task_score_name = 'scoreB'
                    task_score = compute_scoreB(out_label_ids, preds)

                
                accuracy_per_class = {}
                for i, clazz in enumerate(classes):
                    out_label_ids_i = [r[i] for r in out_label_ids]
                    preds_i = [r[i] for r in preds]
                    proba_i = [r[i] for r in proba]
                    obj = {}
                    obj['accuracy']  = accuracy_score(out_label_ids_i, preds_i)
                    obj['auc']       = roc_auc_score(out_label_ids_i, proba_i, average=None)
                    obj['auc_macro'] = roc_auc_score(out_label_ids_i, proba_i, average="macro")
                    obj['f1']        = f1_s[i]
                    obj['f1_macro']  = f1_score(out_label_ids_i, preds_i, average="macro")
                    obj['scoreA']    = compute_scoreA(out_label_ids_i, preds_i)
                    accuracy_per_class[clazz] = obj
        
                result['tasks'][task_name] = {
                    "loss": self.task_eval_loss[task_name],
                    "classifier_loss": self.task_eval_classifier_loss[task_name],
                    "kl_loss": self.task_eval_kl_loss[task_name],
                    "embedding_loss": self.task_eval_embedding_loss[task_name],
                    "query_embedding_loss": self.task_eval_query_embedding_loss[task_name],
                    "classes": classes,
                    "accuracy": acc,
                    "auc_macro": auc_macro,
                    "auc_weighted": auc_weighted,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "task_score_name": task_score_name,
                    "task_score": task_score,
                    "accuracy_per_class": accuracy_per_class,
                    "prediction": preds,
                    "labels": out_label_ids,
                    "proba": proba,
                    "indicies": indicies
                }
        
        
                if self.writer != None:
                    pr_curve_name = f'{task_name}/train_pr_curve' if self.model.training else f'{task_name}/pr_curve'
                    self.writer.add_pr_curve(pr_curve_name, torch.tensor(out_label_ids), torch.tensor(proba), global_step=global_step)
                    self.writer.add_scalar(f'Accuracy/{task_name}/{self.tag}', acc, global_step=global_step)
                    self.writer.add_scalar(f'AUC/{task_name}/macro/{self.tag}', auc_macro, global_step=global_step)
                    self.writer.add_scalar(f'AUC/{task_name}/weighted/{self.tag}', auc_weighted, global_step=global_step)
                    self.writer.add_scalar(f'F1/{task_name}/macro/{self.tag}', f1_macro, global_step=global_step)
                    self.writer.add_scalar(f'F1/{task_name}/weighted/{self.tag}', f1_weighted, global_step=global_step)
                    self.writer.add_scalar(f'{task_score_name}/{task_name}/{self.tag}', task_score, global_step=global_step)
                    for i, clazz in enumerate(classes):
                        obj = accuracy_per_class[clazz]
                        self.writer.add_scalar(f'Accuracy/{clazz}/{self.tag}', obj['accuracy'], global_step=global_step)
                        self.writer.add_scalar(f'AUC/{clazz}/{self.tag}', obj['auc'], global_step=global_step)
                        self.writer.add_scalar(f'AUC/{clazz}/macro/{self.tag}', obj['auc_macro'], global_step=global_step)
                        self.writer.add_scalar(f'F1/{clazz}/{self.tag}', obj['f1'], global_step=global_step)
                        self.writer.add_scalar(f'F1/{clazz}/macro/{self.tag}', obj['f1_macro'], global_step=global_step)
                        self.writer.add_scalar(f'scoreA/{clazz}/{self.tag}', obj['scoreA'], global_step=global_step)
                
                self.log(f"[{self.tag}] [{global_step}] [{task_name:>{max_task_name}}]: {task_score_name} = {task_score:.4f}")
                self.log(f"[{self.tag}] [{global_step}] [{task_name:>{max_task_name}}]: Accuracy = {acc:.4f}")
                self.log(f"[{self.tag}] [{global_step}] [{task_name:>{max_task_name}}]: AUC/macro = {auc_macro:.4f}, AUC/weighted = {auc_weighted:.4f}")
                self.log(f"[{self.tag}] [{global_step}] [{task_name:>{max_task_name}}]: F1/macro = {f1_macro:.4f}, F1/weighted = {f1_weighted:.4f}")
                row_format = "{:>15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
                self.log(row_format.format("Class", "Accuracy", "AUC", "AUC/macro", "F1", "F1/macro", "scoreA"))
                for clazz in classes:
                    obj = accuracy_per_class[clazz]
                    self.log(row_format.format(clazz, f'{obj["accuracy"]:.4f}', f'{obj["auc"]:.4f}', f'{obj["auc_macro"]:.4f}',
                                              f'{obj["f1"]:.4f}', f'{obj["f1_macro"]:.4f}', f'{obj["scoreA"]:.4f}'))
            
        self.save_predictions(path_prefix, preds_suffix)
            
        return result
                
class ModelTrainer():
    def __init__(self, model, optimizer, scheduler, data_config, train_loader, eval_loader, test_data_config, test_loader, epochs, path_prefix, threshold=0.5, eval_every=100, gradient_accumulation_steps=20, max_grad_norm=0.5, device='cpu'):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_config = data_config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_data_config = test_data_config
        self.test_loader = test_loader
        self.epochs = epochs
        self.path_prefix = path_prefix
        Path(path_prefix).mkdir(parents=True, exist_ok=True)
        
        self.threshold = threshold
        self.eval_every = eval_every
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.global_step = 0

        self.best_val_accuracy = {}
        for tag, data_conf in [('eval', data_config), ('test', test_data_config)]:
            self.best_val_accuracy[tag] = {}
            for data in data_conf:
                for task in data['tasks']:
                    self.best_val_accuracy[tag][task['name']] = {'value': 0.0, 'task': task['name']}
                    for c in task['labels']: self.best_val_accuracy[tag][c] = {'value': 0.0, 'task': task['name']}

        
        # create logger with 'ModelTrainer'
        self.logger = logging.getLogger('ModelTrainer')
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(path_prefix + 'trainer.log')
        fh.setLevel(logging.DEBUG)
        self.logger.handlers = []
        self.logger.addHandler(fh)
        
        self.emb_tracker = TbEmbeddingTracker(path_prefix)
        
    def log(self, msg):
        self.logger.info(msg)
        print(msg)
        
    def log_line(self, shape='-'):
        self.log(shape * 125)
        
    def evaluate(self): 
        eval_start_time = time.time()
        self.model.eval()
        self.log_line('.')
        self.log('Evaluating')

        total_evaluations = len(self.train_loader) // self.eval_every
        current_evaluation = ((self.global_step % len(self.train_loader)) + 1) // self.eval_every # taking mod as we might be resuming
        
        eval_loop_config = [
            (self.data_config, self.eval_loader, 'eval'),
            (self.test_data_config, self.test_loader, 'test'),
        ]
        
        results = {}
        
        for data_config, data_loader, tag in eval_loop_config:
            batch_forward = BatchForward(self.model, self.device, data_config, self.threshold, self.emb_tracker, self.writer, self.log, tag)

            with torch.no_grad():
                for i, batch in enumerate(tqdm(data_loader), 0):
                    batch_forward.forward(
                        batch,
                        step=self.global_step,
                        global_step=self.global_step,
                        # dump test embeddings every 10th of evaluations and the last evaluation
                        dump_embeddings=(current_evaluation % max(1, total_evaluations // 10) == 0 or current_evaluation == total_evaluations - 1),
                        log_and_reset_loss=False
                    )

            val_result = batch_forward.accuracy(self.global_step, path_prefix=self.path_prefix, preds_suffix='last')

            val_accuracy = []
            max_c_name = 1
            for data in data_config:
                if data.get('disabled', False): continue
                for task in data['tasks']:
                    task_name = task['name']
                    classes = task['labels']
                    task_val_result = val_result['tasks'][task_name]
                    # best_measures[task_name] = task_val_result['task_score_name'] # 'f1_macro' if len(classes) == 1 else 'f1_weighted'
                    # val_accuracy[task_name] = task_val_result['task_score'] # task_val_result[best_measures[task_name]]
                    val_accuracy.append({'name': task_name, 'task': task_name, 'measure': task_val_result['task_score_name'], 'value': task_val_result['task_score']})
                    max_c_name = max(max_c_name, len(task_name))
                    for c in classes:
                        # best_measures[c] = 'scoreA' # 'f1_macro'
                        # val_accuracy[c] = task_val_result['accuracy_per_class'][c][best_measures[c]]
                        val_accuracy.append({'name': c, 'task': task_name, 'measure': 'scoreA', 'value': task_val_result['accuracy_per_class'][c]['scoreA']})
                        max_c_name = max(max_c_name, len(c))

            to_save = []

            for i in val_accuracy:
                c = i['name']
                acc = i['value']
                measure = i['measure']
                task = i['task']

                best_acc = self.best_val_accuracy[tag][c]['value']
                best_task = self.best_val_accuracy[tag][c]['task']
                # checkpoint
                if acc > best_acc:
                    val_acc_increase = acc - best_acc
                    val_acc_increase_percentage = 100.0 * val_acc_increase / best_acc if best_acc != 0 else 100.0
                    self.best_val_accuracy[tag][c] = {'value': acc, 'task': task}
                    to_save.append(c)
                    self.log(f"[{c:>{max_c_name}}] Increased {measure:>6} ++ New = {acc:.4f} Last {measure:>6} = {best_acc:.4f} Increased by: {val_acc_increase:.4f} ({val_acc_increase_percentage:.4f}%)")
                else:
                    val_acc_decrease = best_acc - acc
                    val_acc_decrease_percentage = 100.0 * val_acc_decrease / best_acc if best_acc != 0 else 100.0
                    self.log(f"[{c:>{max_c_name}}] Decreased {measure:>6} -- New = {acc:.4f} Best {measure:>6} = {best_acc:.4f} Decreased by: {val_acc_decrease:.4f} ({val_acc_decrease_percentage:.4f}%)")

            val_result['best_val_accuracy'] = self.best_val_accuracy[tag]

            for c in to_save:
                # save_checkpoint(f'{self.path_prefix}{tag}_best_{c}.pt', self.model, val_result)
                batch_forward.save_predictions(path_prefix=self.path_prefix, suffix=f'best_{c}')
            if tag == 'eval': save_checkpoint(f'{self.path_prefix}last.pt', self.model, val_result)
            
            results[tag] = val_result

        time_elapsed = time_elapsed_as_str(eval_start_time)
        self.log(f'Finished evaluating - Time: {time_elapsed}')
        self.log_line('.')
            
        return results
    
    def resume(self, checkpoint_path=None):
        load_path = checkpoint_path if checkpoint_path != None else f'{self.path_prefix}last.pt'
        state_dict = load_checkpoint(load_path, self.model)
        self.global_step = state_dict['global_step']
        self.best_val_accuracy = state_dict['best_val_accuracy']

        self.log(f"Resuming after global step {self.global_step} from checkpoint '{load_path}'")
        self.log_line('=')
        self.emb_tracker.resume()
        self.run()
    
    def run(self):
        start_time = time.time()
        self.writer = SummaryWriter(log_dir=self.path_prefix)
        optimizer_step = 0
        
        #self.writer.add_graph(self.model)
        
        self.model.zero_grad()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.log(f'Epoch {epoch+1}/{self.epochs}:')
            self.log_line('-')
            
            batch_forward = BatchForward(self.model, self.device, self.data_config, self.threshold, self.emb_tracker, self.writer, self.log, 'train')
            
            for step, batch in enumerate(tqdm(self.train_loader)):
                self.model.train()
                output, total_loss = batch_forward.forward(
                    batch,
                    step=epoch,
                    global_step=self.global_step,
                    # dump train embeddings every third of epochs and the last epoch
                    dump_embeddings=(epoch % max(1, self.epochs // 3) == 0 or epoch == self.epochs - 1),
                    log_and_reset_loss=((step + 1) % self.gradient_accumulation_steps == 0)
                )
                               
                if self.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.gradient_accumulation_steps

                total_loss.backward()
                
                self.global_step += 1

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule         
                    self.optimizer.zero_grad()
                    optimizer_step += 1

                if (step + 1) % self.eval_every == 0:
                    self.model.eval()
                    val_result = self.evaluate()
                    self.model.train()
            
            batch_forward.accuracy(self.global_step, path_prefix=self.path_prefix, preds_suffix='last')
            
            time_elapsed = time_elapsed_as_str(start_time)
            epoch_time_elapsed = time_elapsed_as_str(epoch_start_time)
            self.log(f'Epoch Time: {epoch_time_elapsed}')
            self.log(f'Total Time: {time_elapsed}')
            self.log_line('=')
        
        time_elapsed = time_elapsed_as_str(start_time)
        self.log(f'Finished! Total Time: {time_elapsed}')
        self.writer.close()
