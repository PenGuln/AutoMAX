
import logging
import torch
import libauc 
import numpy as np
from libauc.trainer import Trainer, CallbackHandler, TrainerCallback, TrainerState
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_cfg: dict):
    name = model_cfg.get("name", "").lower()
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer
    
class TransformerTrainer(Trainer):
    def __init__(self, 
                 model_cfg,
                 train_args, 
                 train_dataset, 
                 eval_dataset = None, 
                 metric = None, 
                 callbacks = None):
        #super().__init__()
        self.args = train_args
        self.model, self.tokenizer = build_model(model_cfg)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.state = TrainerState()
        self.state.total_epoch = self.args.epochs

        self.sampler, self.trainloader = self._get_train_dataloader(self.args)
        self.evalloaders = []
        if self.eval_dataset:
            for dataset in self.eval_dataset:
                self.evalloaders.append(self._get_eval_dataloader(dataset, self.args))
        

        self.data_len = self.sampler.pos_len + self.sampler.neg_len
        self.pos_len = self.sampler.pos_len
        self.neg_len = self.sampler.neg_len

        self.loss_fn, self.optimizer = self._construct_optimizer_and_loss(self.model, train_args)

        self.metric = metric
        if callbacks is None:
            self.callback_handler = CallbackHandler([], self.model, self.optimizer, self.loss_fn)
        else:
            self.callback_handler = CallbackHandler(callbacks, self.model, self.optimizer, self.loss_fn)
        self.callback_handler.on_init_end(self.args, self.state)

    def train(self):
       
        self.callback_handler.on_train_begin(self.args, self.state)
        train_log = []

        self.model = self.model.cuda()
        self.loss_fn = self.loss_fn.cuda()

        if self.args.resume_from_checkpoint:
            latest_checkpoint = self.get_latest_checkpoint(self.args.output_path)
            if latest_checkpoint:
                checkpoint = self.load_checkpoint(latest_checkpoint)
                logger.info(f"Resuming training from epoch {self.state.epoch}")
            else:
                logger.info("No checkpoint found in output folder, starting from scratch")
        
        for epoch in range(self.state.epoch, self.args.epochs):
            self.model.train()
            train_loss = []
            for data in tqdm(self.trainloader):
                self.callback_handler.on_step_begin(self.args, self.state)
                texts, targets, indices = data
                inputs = self.tokenizer.batch_encode_plus(
                    texts,
                    max_length=512,
                    add_special_tokens=True,
                    padding="max_length",
                    return_token_type_ids=True,
                    truncation=True,
                    return_tensors='pt'
                )
                ids = inputs['input_ids'].cuda()
                mask = inputs['attention_mask'].cuda()
                
                indices = indices.cuda()
                targets = targets.cuda()

                   
                outputs = self.model(ids, attention_mask=mask)
                y_pred = torch.sigmoid(outputs.logits)
                y_pred = torch.flatten(y_pred)

                if isinstance(self.loss_fn, libauc.losses.losses.CrossEntropyLoss):
                    loss = self.loss_fn(y_pred, targets)
                else:
                    loss = self.loss_fn(y_pred, targets, index=indices)

                self.optimizer.zero_grad()  
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                self.callback_handler.on_step_end(self.args, self.state)

            self.model.eval()
            train_loss = np.mean(train_loss)
            metrics, test_true, test_pred = self.evaluate_loop(self.model)
            train_log.append({
                "metrics" : metrics,
                "epoch" : epoch,
                "lr": self.optimizer.lr,
                "loss" : train_loss
            })
            self.callback_handler.on_epoch_end(
                self.args, self.state, 
                metrics=metrics, 
                train_loss=train_loss,
                lr=self.optimizer.lr,
                test_true=test_true, 
                test_pred=test_pred
            )

            # Save checkpoint periodically
            if (epoch + 1) % self.args.save_checkpoint_every == 0:
                checkpoint_path = os.path.join(self.args.output_path, self.args.experiment_name, f"epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path)
        
        self.callback_handler.on_train_end(self.args, self.state)
        
        # Save final model
        final_model_path = os.path.join(self.args.output_path, self.args.experiment_name, f"epoch_{self.args.epochs}.pt")
        self.save_checkpoint(final_model_path)
        
        return train_log
    

    def evaluate(self, loader, model):

        predictions = list()
        true_labels = list()
        model.eval()
        with torch.no_grad():
            for data in loader:
                texts, targets, _ = data
                inputs = self.tokenizer.batch_encode_plus(
                    texts,
                    max_length=512,
                    add_special_tokens=True,
                    padding="max_length",
                    return_token_type_ids=True,
                    truncation=True,
                    return_tensors='pt'
                )
                ids = inputs['input_ids'].cuda()
                mask = inputs['attention_mask'].cuda()
                outputs = model(ids, attention_mask=mask)
                #logits = torch.softmax(logits, dim=1)
                logits = torch.sigmoid(outputs.logits)
                predictions.append(logits.cpu().detach().numpy())
                true_labels.append(targets.cpu().numpy())
            predictions = np.concatenate(predictions)
            true_labels = np.concatenate(true_labels)
     
        test_true = true_labels
        test_pred = predictions
        result = self.metric(test_true, test_pred)
        return result, test_true, test_pred
