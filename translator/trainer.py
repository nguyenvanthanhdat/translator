import torch
from torch import nn
from collections import defaultdict
from typing import Dict, Union, Any, Optional, Tuple, List
from torch.utils.data import DataLoader

from transformers import Seq2SeqTrainer
from datasets import Dataset


class SBSTrainer(Seq2SeqTrainer):
    def __init__(self, alpha: int = 0.1, output_expl:bool = True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_expl = output_expl
        self.custom_state_dict = defaultdict(list)

    def get_train_dataloader(self) -> DataLoader:
        """
        Return the dataloader for training.

        :return: The training DataLoader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Return the dataloader for evaluate.

        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluate in training requires a eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        logs = dict()
        
        # predict
        pred_outputs = model(**inputs['pred'])
        pred_loss = (1. - self.alpha) * pred_outputs.loss
        logs['pred_loss'] = pred_loss.item()
        
        # expl
        expl_outputs = model(**inputs['expl'])
        expl_loss = self.alpha * expl_outputs.loss
        logs['expl_loss'] = expl_loss.item()
        
        # total loss
        loss = pred_loss + expl_loss

        for k, v in logs.items():
            # Set maxlen of list to avoid memory leak, useful when
            # customized_logging_list has not been cleaned correctly
            if len(self.custom_state_dict[k]) < 5000:
                self.custom_state_dict[k].append(v)

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            pred_outputs = self.compute_loss(model, inputs)

        loss = pred_outputs.detach()

        return (loss, None, None)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        
        if self.state.epoch is not None:
            logs['epoch'] = round(self.state.epoch, 2)
            
        # Inject Customised logging behavior
        for k, v in self.custom_state_dict.items():
            if len(v) > 0:
                if isinstance(v[0], torch.Tensor):
                    v = [value.items() for value in v]
                logs[k] = round(sum(v) / len(v), 4)
              
        self.custom_state_dict.clear()

        output = {**logs, **{"step": self.state.global_step}}
        
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)