import argparse
from itertools import chain
from pathlib import Path
from sys import stdout, stderr
from time import sleep
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, DatasetDict, Sequence, Value
from datasets import load_dataset, load_metric
from datasets.metric import Metric
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from attrdict.dictionary import AttrDict
from base.io import RedirStd, set_cuda_path, set_torch_ext_path, make_dir, load_attrs, save_attrs, new_path, remove_any
from base.str import horizontal_line, to_morphemes
from base.tensor import to_tensor_batch
from base.time import now, MyTimer
from base.tokenizer_korbert import KorbertTokenizer
from base.util import append_intersection
from korquad import korquad
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import AutoConfig, PretrainedConfig
from transformers import AutoModel, PreTrainedModel
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
from transformers.activations import ACT2FN


class SpanModel(nn.Module):
    """
    Span Model on top for extractive question-answering tasks like SQuAD
    - Refer to `transformers.models.bert.modeling_bert.BertForQuestionAnswering`
    """

    def __init__(self, state, tokenizer, logging=True):
        super().__init__()
        self.state = state
        self.tokenizer = tokenizer

        if logging:
            print(horizontal_line(c="-"))
        with RedirStd():
            if 'num_classes' not in self.state:
                self.state.num_classes = 2
            self.pretrained_config: PretrainedConfig = AutoConfig.from_pretrained(self.state.pretrained, num_labels=self.state.num_classes)
            self.pretrained_model: PreTrainedModel = AutoModel.from_pretrained(self.state.pretrained, config=self.pretrained_config)
            self.activation: nn.Module = ACT2FN[self.pretrained_config.hidden_act]
        if logging:
            self.check_pretrained()

        self.out_proj = nn.Linear(self.pretrained_config.hidden_size, self.pretrained_config.num_labels)

    def forward(self, **x):
        outputs = self.pretrained_model(**x)
        # last_hidden_state = self.pretrained_model(**x).last_hidden_state
        sequence_output = outputs[0]
        logits = self.out_proj(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits

    def check_pretrained(self, sample=True, file=stdout):
        print(horizontal_line(c="-"), file=stdout)
        model_desc = str(self.pretrained_model).splitlines()
        idx1 = next((i for i, x in enumerate(model_desc) if "(encoder)" in x), 8)
        idx2 = next((i for i, x in enumerate(model_desc) if "(pooler)" in x), -1)
        print(f"- {'pretrained_model':25s} = {chr(10).join(model_desc[:idx1] + ['  ...'] + model_desc[idx2:])}", file=file)
        if sample:
            plain_text = ["한국어 사전학습 모델을 공유합니다.", "지금까지 금해졌다."]
            morps_text = ["한국어/NNP 사전/NNG 학습/NNG 모델/NNG 을/JKO 공유/NNG 하/XSV ㅂ니다/EF ./SF", "지금/NNG 까지/JX 금하/VV 어/EC 지/VX 었/EP 다/EF ./SF"]
            batch_text = morps_text if "-morp" in self.state.pretrained else plain_text
            # inputs = self.tokenizer.batch_encode_plus(batch_text, padding='max_length', max_length=self.state.max_sequence_length, truncation=True)
            # output = self.pretrained_model(
            #     torch.tensor(inputs['input_ids']),
            #     torch.tensor(inputs['attention_mask']),
            #     torch.tensor(inputs['token_type_ids'])
            # )
            # hidden = output.last_hidden_state
            # print(f"- {'encoded.input_ids':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['input_ids']).size()))}) / {' '.join(str(x) for x in inputs['input_ids'][0][:25])} ... {' '.join(str(x) for x in inputs['input_ids'][0][-25:])}", file=file)
            # print(f"- {'encoded.attention_mask':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['attention_mask']).size()))}) / {' '.join(str(x) for x in inputs['attention_mask'][0][:25])} ... {' '.join(str(x) for x in inputs['attention_mask'][0][-25:])}", file=file)
            # print(f"- {'encoded.token_type_ids':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['token_type_ids']).size()))}) / {' '.join(str(x) for x in inputs['token_type_ids'][0][:25])} ... {' '.join(str(x) for x in inputs['token_type_ids'][0][-25:])}", file=file)
            # print(f"- {'forwarded.hidden_output':25s} = ({'x'.join(str(x) for x in list(hidden.size()))}) / {hidden[0]}", file=file)


class HeadModel(nn.Module):
    """
    Head Model for sentence-level classification or regression tasks.
    - Refer to `transformers.models.big_bird.modeling_big_bird.BigBirdClassificationHead`
    """

    def __init__(self, state, tokenizer, logging=True):
        super().__init__()
        self.state = state
        self.tokenizer = tokenizer

        if logging:
            print(horizontal_line(c="-"))
        with RedirStd():
            if 'num_classes' not in self.state:
                self.state.num_classes = 1
            self.pretrained_config: PretrainedConfig = AutoConfig.from_pretrained(self.state.pretrained, num_labels=self.state.num_classes)
            self.pretrained_model: PreTrainedModel = AutoModel.from_pretrained(self.state.pretrained, config=self.pretrained_config)
            self.activation: nn.Module = ACT2FN[self.pretrained_config.hidden_act]
        if logging:
            self.check_pretrained()

        self.dense = nn.Linear(self.pretrained_config.hidden_size, self.pretrained_config.hidden_size)
        self.dropout = nn.Dropout(self.pretrained_config.classifier_dropout if self.pretrained_config.classifier_dropout else self.pretrained_config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.pretrained_config.hidden_size, self.pretrained_config.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, **x):
        output = self.pretrained_model(**x).last_hidden_state
        x = output[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        if self.state.loss_metric.startswith('MSELoss'):
            return logits
        else:
            probs = self.sigmoid(logits)
            return probs

    def check_pretrained(self, sample=True, file=stdout):
        print(horizontal_line(c="-"), file=stdout)
        model_desc = str(self.pretrained_model).splitlines()
        idx1 = next((i for i, x in enumerate(model_desc) if "(encoder)" in x), 8)
        idx2 = next((i for i, x in enumerate(model_desc) if "(pooler)" in x), -1)
        print(f"- {'pretrained_model':25s} = {chr(10).join(model_desc[:idx1] + ['  ...'] + model_desc[idx2:])}", file=file)
        if sample:
            plain_text = ["한국어 사전학습 모델을 공유합니다.", "지금까지 금해졌다."]
            morps_text = ["한국어/NNP 사전/NNG 학습/NNG 모델/NNG 을/JKO 공유/NNG 하/XSV ㅂ니다/EF ./SF", "지금/NNG 까지/JX 금하/VV 어/EC 지/VX 었/EP 다/EF ./SF"]
            batch_text = morps_text if "-morp" in self.state.pretrained else plain_text
            # inputs = self.tokenizer.batch_encode_plus(batch_text, padding='max_length', max_length=self.state.max_sequence_length, truncation=True)
            # output = self.pretrained_model(
            #     torch.tensor(inputs['input_ids']),
            #     torch.tensor(inputs['attention_mask']),
            #     torch.tensor(inputs['token_type_ids'])
            # )
            # hidden = output.last_hidden_state
            # print(f"- {'encoded.input_ids':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['input_ids']).size()))}) / {' '.join(str(x) for x in inputs['input_ids'][0][:25])} ... {' '.join(str(x) for x in inputs['input_ids'][0][-25:])}", file=file)
            # print(f"- {'encoded.attention_mask':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['attention_mask']).size()))}) / {' '.join(str(x) for x in inputs['attention_mask'][0][:25])} ... {' '.join(str(x) for x in inputs['attention_mask'][0][-25:])}", file=file)
            # print(f"- {'encoded.token_type_ids':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['token_type_ids']).size()))}) / {' '.join(str(x) for x in inputs['token_type_ids'][0][:25])} ... {' '.join(str(x) for x in inputs['token_type_ids'][0][-25:])}", file=file)
            # print(f"- {'forwarded.hidden_output':25s} = ({'x'.join(str(x) for x in list(hidden.size()))}) / {hidden[0]}", file=file)


class MyTrainer(LightningLite):
    def __init__(self, config, prefix=None):
        self.prefix: Optional[str] = prefix
        self.state: AttrDict = load_attrs(config, pre={"name": Path(config).stem})
        self.state.finetuning_model = self.state.finetuning_model if "finetuning_model" in self.state else "HeadModel"
        self.state.label_text = self.state.label_text if "label_text" in self.state else None
        self.state.doc_stride = self.state.doc_stride if "doc_stride" in self.state else 0
        self.state.tokenize_online = "tokenize_online" in self.state and self.state.tokenize_online
        self.state.analyzer_netloc = self.state.analyzer_netloc if "analyzer_netloc" in self.state else None
        self.state.trim_whitespace = "trim_whitespace" in self.state and self.state.trim_whitespace
        self.state.offsets_mapping = self.state.offsets_mapping if "offsets_mapping" in self.state else False
        self.state.overflowing_tokens = self.state.overflowing_tokens if "overflowing_tokens" in self.state else False
        self.state.text_truncation = self.state.text_truncation if "text_truncation" in self.state else True
        set_cuda_path()
        set_torch_ext_path(n_run=self.state.gpus[0])
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.no_sequence_ids = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[object] = None
        self.dataloader: Dict[str, DataLoader] = {}
        self.loss_metric: Optional[_Loss] = None
        self.score_metric: Optional[Metric] = None
        self.raw_datasets: Optional[DatasetDict] = None
        self.cnt_tokenized: Optional[int] = None
        super().__init__(gpus=self.state.gpus, precision=self.state.precision,
                         strategy=DeepSpeedStrategy(stage=2) if self.state.strategy == "deepspeed" else self.state.strategy)

    def run(self):
        # INIT
        with RedirStd():
            seed_everything(self.state.seed)
            self.state.name += now("-%m%d.%H%M")
        if self.is_global_zero:
            print(horizontal_line(c="="))
        else:
            sleep(15.0)

        # MAIN
        with MyTimer(prefix=self.prefix, name=f"{Path(__file__).stem} ({self.state.name})", b=0, logging=self.is_global_zero):
            # BEGIN
            if self.is_global_zero:
                print(horizontal_line(c="="))
                for k in self.state.log_targets:
                    if k in self.state.keys():
                        self.log_state_value(key=k, file=stdout)
            else:
                sleep(15.0)

            # READY(data)
            self.load_tokenizer(logging=self.is_global_zero)
            self.encode_dataset(logging=self.is_global_zero, trimming=self.state.trim_whitespace, keep_input=False)
            self.state.steps_per_epoch = len(self.dataloader['train'])
            self.state.total_steps = self.state.num_train_epochs * self.state.steps_per_epoch
            epoch_per_step = 1.0 / self.state.steps_per_epoch

            # READY(output)
            self.state.output_dir = make_dir(Path(self.state.output_dir) / self.state.name)
            outfiles = {
                "state": self.state.output_dir / "trainer_state.json",
                "model": self.state.output_dir / "pytorch_model"
            }
            logs = dict({
                "step": 0,
                "epoch": 0.0,
                "record": []
            })
            self.state.records = logs["record"]

            # READY(model)
            with RedirStd():
                self.model = SpanModel(state=self.state, tokenizer=self.tokenizer, logging=self.is_global_zero) if self.state.finetuning_model == "SpanModel" \
                    else HeadModel(state=self.state, tokenizer=self.tokenizer, logging=self.is_global_zero)
                self.optimizer, self.scheduler = self.configure_optimizers()
                self.loss_metric = nn.MSELoss() if self.state.loss_metric == "MSELoss" else nn.CrossEntropyLoss() if self.state.loss_metric == "CrossEntropyLoss" else None
                assert self.loss_metric is not None, f"Undefined loss_metric: {self.state.loss_metric}"
                if self.state.score_metric.major == "korquad":
                    self.score_metric = korquad.Korquad()
                else:
                    self.score_metric = load_metric(self.state.score_metric.major, self.state.score_metric.minor if 'minor' in self.state.score_metric else None)
            if self.is_global_zero:
                self.state['model'] = f"{type(self.model).__qualname__}(pretrained={Path(self.state.pretrained).name})"
                self.state['optimizer'] = f"{type(self.optimizer).__qualname__}(lr={self.state.learning_rate})"
                self.state['scheduler'] = f"{type(self.scheduler).__qualname__}(step_size={self.state.scheduling_epochs}, gamma={self.state.scheduling_gamma})"
                self.state['loss_metric'] = f"{type(self.loss_metric).__qualname__}()"
                self.state['score_metric'] = f"load_metric({self.state.score_metric.major}{f', {self.state.score_metric.minor}' if 'minor' in self.state.score_metric else ''})"
                print(horizontal_line(c="-"))
                for k in ('model', 'optimizer', 'scheduler', 'loss_metric', 'score_metric'):
                    print(f"- {k:25s} = {self.state[k]}")
                print(horizontal_line(c="-"))
            self.model, self.optimizer = self.setup(self.model, self.optimizer)
            if self.is_global_zero:
                print(horizontal_line(c="="))
            else:
                sleep(1.0)

            # EPOCH
            for epoch in range(1, self.state.num_train_epochs + 1):
                # INIT
                sleep(1.0)
                metrics = {}
                current = f"(Epoch {epoch:02d})"
                if self.is_global_zero:
                    with RedirStd():
                        print(f"{now()} {current} composed #{self.global_rank + 1:01d}: learning_rate={self.get_learning_rate():.10f}", file=stderr)
                else:
                    sleep(2.0)

                # TRAIN
                sleep(3.0)
                self.model.train()
                outputs = []
                timer = MyTimer()
                with timer:
                    for batch_idx, batch in enumerate(tqdm(self.dataloader['train'], desc=f"{now()} {current} training #{self.global_rank + 1:01d}", bar_format="{l_bar}{bar:30}{r_bar}")):
                        batch = to_tensor_batch(batch, input_keys=self.tokenizer.model_input_names)
                        self.optimizer.zero_grad()
                        output = self.each_step(batch, batch_idx, input_keys=self.tokenizer.model_input_names)
                        outputs.append(output)
                        logs["step"] += 1
                        logs["epoch"] += epoch_per_step
                        logs["learning_rate"] = self.get_learning_rate()
                        self.backward(output['loss'])
                        self.optimizer.step()
                    self.scheduler.step()
                metrics["step"] = self.outputs_to_metrics(outputs, timer=timer)
                logs["epoch"] = round(logs["epoch"], 1)

                # METER
                sleep(3.0)
                self.model.eval()
                with torch.no_grad():
                    for k in self.raw_datasets.keys():
                        if k in self.dataloader and k in self.state.score_targets and self.dataloader[k] and self.state.score_targets[k]:
                            inputs = []
                            outputs = []
                            timer = MyTimer()
                            with timer:
                                for batch_idx, batch in enumerate(tqdm(self.dataloader[k], desc=f"{now()} {current} metering #{self.global_rank + 1:01d}", bar_format="{l_bar}{bar:30}{r_bar}")):
                                    batch = to_tensor_batch(batch, input_keys=self.tokenizer.model_input_names)
                                    output = self.each_step(batch, batch_idx, input_keys=self.tokenizer.model_input_names)
                                    outputs.append(output)
                                    inputs.append(batch)
                            metrics[k] = self.outputs_to_metrics(outputs, timer=timer)

                # SAVE
                logs["state_path"] = new_path(outfiles["state"], post=f'dev{self.global_rank}')
                logs["model_path"] = new_path(outfiles["model"], post=f'{logs["epoch"]:02.0f}e')
                self.save(self.model.state_dict(), filepath=remove_any(logs["model_path"]))
                logs["record"].append({
                    "step": logs["step"],
                    "epoch": logs["epoch"],
                    "device": self.global_rank,
                    "metrics": metrics,
                    "model_path": logs["model_path"] if logs['model_path'].exists() else None,
                    "learning_rate": logs["learning_rate"],
                })
                self.state.records = logs["record"]
                save_attrs(self.state, file=remove_any(logs["state_path"]), keys=self.state.log_targets)

                # PRINT
                sleep(3.0)
                with RedirStd():
                    for name, score in metrics.items():
                        print(f"{now()} {current} measured #{self.global_rank + 1:01d}: {name:5s} | {', '.join(f'{k}={score[k]:.4f}' for k in append_intersection(score.keys(), ['runtime']))}", file=stderr)
                sleep(1.0)
                with RedirStd():
                    if logs["model_path"].exists():
                        print(f"{now()} {current} exported #{self.global_rank + 1:01d}: model | {logs['model_path']}", file=stderr)
                sleep(1.0)
                if self.is_global_zero:
                    print(horizontal_line(c="-" if epoch < self.state.num_train_epochs else "="))

            # END
            sleep(1.0)

        # EXIT
        if self.is_global_zero:
            print(horizontal_line())
        else:
            sleep(3.0)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.state.learning_rate)
        scheduler = StepLR(optimizer, step_size=self.state.scheduling_epochs, gamma=self.state.scheduling_gamma, verbose=False)
        return optimizer, scheduler

    def forward(self, **x):
        return self.model(**x)

    def each_step(self, batch: dict, batch_idx, input_keys):
        x = {k: batch.pop(k) for k in input_keys}
        if self.state.finetuning_model == "SpanModel":
            p1, p2 = self.forward(**x)
            y1 = batch.pop(self.state.label_column[0])
            y2 = batch.pop(self.state.label_column[1])
            if len(y1.size()) > 1:
                y1 = y1.squeeze(-1)
            if len(y2.size()) > 1:
                y2 = y2.squeeze(-1)
            ignored_index = p1.size(1)
            y1 = y1.clamp(0, ignored_index)
            y2 = y2.clamp(0, ignored_index)
            loss1 = self.loss_metric(input=p1, target=y1)
            loss2 = self.loss_metric(input=p2, target=y2)
            loss = (loss1 + loss2) / 2
            return {'y1': y1, 'y2': y2, 'p1': p1, 'p2': p2, 'loss': loss}
        else:
            p = self.forward(**x)
            if len(self.state.label_column.split('.')) == 2:
                major, minor = self.state.label_column.split('.')
                y = batch.pop(major).pop(minor)
            else:
                y = batch.pop(self.state.label_column)
            if self.state.loss_metric.startswith('MSELoss'):
                p = p.view(-1)
                y = y.float().view(-1)
            else:
                y = y.long().view(-1)
            loss = self.loss_metric(input=p, target=y)
            return {'y': y, 'p': p, 'loss': loss}

    def outputs_to_metrics(self, outputs, timer: Optional[MyTimer] = None):
        score = {}
        if timer:
            score['runtime'] = timer.delta.total_seconds()
        if self.state.finetuning_model == "SpanModel":
            y1s = torch.cat([x['y1'] for x in outputs]).detach().cpu().numpy()
            y2s = torch.cat([x['y1'] for x in outputs]).detach().cpu().numpy()
            p1s = torch.cat([x['p1'] for x in outputs]).detach().cpu().numpy()
            p2s = torch.cat([x['p2'] for x in outputs]).detach().cpu().numpy()
            score['loss'] = torch.stack([x['loss'] for x in outputs]).detach().cpu().numpy().mean().item()
        else:
            ys = torch.cat([x['y'] for x in outputs]).detach().cpu().numpy()
            ps = torch.cat([x['p'] for x in outputs]).detach().cpu().numpy()
            score['loss'] = torch.stack([x['loss'] for x in outputs]).detach().cpu().numpy().mean().item()
            if len(ps.shape) > 1:
                ps = np.argmax(ps, axis=1)
            score.update(self.score_metric.compute(references=ys, predictions=ps))
        return score

    @staticmethod
    def outputs_to_predict(outputs, inputs, with_label=False):
        cols = {}
        for k in inputs[0].keys():
            cols[k] = list(chain.from_iterable(batch[k] for batch in inputs))
        cols['predict'] = torch.cat([x['p'] for x in outputs]).detach().cpu().numpy().tolist()
        if with_label:
            cols['label'] = torch.cat([x['y'] for x in outputs]).detach().cpu().numpy().tolist()
        rows = []
        for i in range(len(cols['predict'])):
            rows.append({k: cols[k][i] for k in cols.keys()})
        return rows

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def log_state_value(self, key, file=stdout):
        print(f"- {'state.' + key:25s} = ({type(self.state[key]).__qualname__}) {self.state[key]}", file=file)

    def load_tokenizer(self, logging=True):
        if logging:
            print(horizontal_line(c="-"))
        if "-morp" in self.state.pretrained:
            self.tokenizer = KorbertTokenizer.from_pretrained(self.state.pretrained, max_len=self.state.max_sequence_length,
                                                              tokenize_online=self.state.tokenize_online, analyzer_netloc=self.state.analyzer_netloc,
                                                              tokenize_chinese_chars=False, use_fast=False, do_lower_case=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.state.pretrained, max_len=self.state.max_sequence_length)
        self.no_sequence_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        if logging:
            self.check_tokeziner()

    def check_tokeziner(self, sample=True, file=stdout):
        print(horizontal_line(c="-"), file=file)
        print(f"- {'pretrained_tokenizer':25s} = {type(self.tokenizer).__qualname__} / "
              f"is_fast={self.tokenizer.is_fast} / "
              f"vocab_size={self.tokenizer.vocab_size} / "
              f"model_input_names={', '.join(self.tokenizer.model_input_names)} / "
              f"model_max_length={self.tokenizer.model_max_length} / "
              f"padding_side={self.tokenizer.padding_side} / "
              f"special_tokens={', '.join(self.tokenizer.all_special_tokens)}", file=file)
        if sample:
            plain = "[CLS] 한국어 사전학습 모델을 공유합니다. [SEP]"
            morps = "[CLS] 한국어/NNP 사전/NNG 학습/NNG 모델/NNG 을/JKO 공유/NNG 하/XSV ㅂ니다/EF ./SF [SEP]"
            ofs = None
            if self.state.tokenize_online:
                tks, ofs = self.tokenizer.tokenize_online(plain)
            else:
                tks = self.tokenizer.tokenize(morps if "-morp" in self.state.pretrained else plain)
            ids = self.tokenizer.convert_tokens_to_ids(tks)
            print(f"- {'tokenized.sample.plain':25s} =", plain, file=file)
            print(f"- {'tokenized.sample.morps':25s} =", morps, file=file)
            print(f"- {'tokenized.sample.tks':25s} =", ' '.join(tks), file=file)
            print(f"- {'tokenized.sample.ids':25s} =", ' '.join(map(str, ids)), file=file)
            if ofs:
                print(f"- {'tokenized.sample.ofs':25s} =", ' '.join([f'({s},{e})' for s, e in ofs]), file=file)

    def encode_dataset(self, logging=True, trimming=False, keep_input=False, default_label_cols=('label', 'labels')):
        # load raw datasets
        with RedirStd(stdout=None, stderr=None):
            self.raw_datasets = load_dataset("json", data_files={k: v for k, v in self.state.data_files.items() if v}, field="data")
        if "num_train_samples" in self.state.keys() and self.state.num_train_samples and \
                "num_test_samples" in self.state.keys() and self.state.num_test_samples and \
                self.state.num_train_samples > 0 and self.state.num_test_samples > 0:
            for k in self.raw_datasets.keys():
                self.raw_datasets[k] = Dataset.from_dict(self.raw_datasets[k][:(self.state.num_train_samples if k == "train" else self.state.num_test_samples)])

        # encode texts in datasets
        self.cnt_tokenized = 0
        # with RedirStd(stdout=None, stderr=None):
        if 'label_column' not in self.state:
            self.state.label_column = 'label'
        if isinstance(self.state.label_column, str):
            all_label_cols = set(default_label_cols).union({self.state.label_column.split(".")[0]})
        else:
            assert isinstance(self.state.label_column, (list, tuple)) and len(self.state.label_column) > 1
            all_label_cols = set(default_label_cols).union(set(self.state.label_column))
        first_split = list(self.raw_datasets.keys())[0]
        remove_columns = [x for x in self.raw_datasets[first_split].column_names if x not in all_label_cols] if not keep_input else None
        self.raw_datasets = self.raw_datasets.map(self.encode_text, with_indices=True, batched=True, batch_size=1,
                                                  fn_kwargs={'logging': logging, 'trimming': trimming}, load_from_cache_file=False, remove_columns=remove_columns)

        # setup dataloaders
        if self.state.train_batch_size <= 0:
            self.state.train_batch_size = self.state.max_batch_size[f"max_sequence_length={self.state.max_sequence_length}"][f"precision={self.state.precision}"]
            self.log_state_value(key='train_batch_size', file=stderr)
        if self.state.test_batch_size <= 0:
            self.state.test_batch_size = self.state.max_batch_size[f"max_sequence_length={self.state.max_sequence_length}"][f"precision={self.state.precision}"]
            self.log_state_value(key='test_batch_size', file=stderr)
        for k in self.raw_datasets.keys():
            self.dataloader[k] = self.setup_dataloaders(DataLoader(self.raw_datasets[k],
                                                                   batch_size=self.state.train_batch_size if k == "train" else self.state.test_batch_size,
                                                                   shuffle=False))

        # print dataset info
        if logging:
            for k, dataset in self.raw_datasets.items():
                dataset: Dataset = dataset
                for f, v in dataset.features.items():
                    assert isinstance(v, (Sequence, Value, dict)), f"feature({f}[{type(v)}]) is not {Sequence.__name__}, {Value.__name__} or {dict.__name__}"
                    if isinstance(v, dict):
                        for f2, v2 in v.items():
                            assert isinstance(v2, (Value, Sequence)), f"feature({f2}[{type(v2)}]) is not {Value.__name__} or {Sequence.__name__}"
                feature_specs = []
                for f, v in dataset.features.items():
                    if isinstance(v, dict):
                        feature_specs.append(', '.join(f'{f}.{f2}[{v2.dtype}]' for f2, v2 in v.items()))
                    else:
                        feature_specs.append(f'{f}[{v.dtype}]')
                print(f"- {'raw_datasets[' + k + ']':25s} = {dataset.num_rows:7,d} samples / {', '.join(feature_specs)}")
                self.state["data_files"][k] = {"file": self.state.data_files[k], "rows": dataset.num_rows}
            print(f"- {'input_text_columns':25s} = {self.state.input_text1}, {self.state.input_text2}")
            print(f"- {'label_column':25s} = {self.state.label_column}")

    def encode_text(self, examples, indices, logging=True, trimming=False, **kwargs):
        example = {}
        for k in examples.keys():
            example[k] = examples[k][0]
        example_index = indices[0]
        if self.state.input_text1 and self.state.input_text1 in example:
            example[self.state.input_text1 + "_origin"] = example[self.state.input_text1]
        if self.state.input_text2 and self.state.input_text2 in example:
            example[self.state.input_text2 + "_origin"] = example[self.state.input_text2]
        if trimming:
            if self.state.input_text1 and self.state.input_text1 in example:
                example[self.state.input_text1] = self.trim_whitespace(example[self.state.input_text1])
            if self.state.input_text2 and self.state.input_text2 in example:
                example[self.state.input_text1] = self.trim_whitespace(example[self.state.input_text1])
        if not self.state.tokenize_online and "-morp" in self.state.pretrained:
            if self.state.input_text1 and self.state.input_text1 in example:
                example[self.state.input_text1] = to_morphemes(example[self.state.input_text1])
            if self.state.input_text2 and self.state.input_text2 in example:
                example[self.state.input_text2] = to_morphemes(example[self.state.input_text2])
        text_pair = (
            (example[self.state.input_text1],) if self.state.input_text2 is None
            else (example[self.state.input_text1], example[self.state.input_text2])
        )
        encoded = self.tokenizer(*text_pair,
                                 padding='max_length', max_length=self.state.max_sequence_length, stride=self.state.doc_stride,
                                 return_offsets_mapping=self.state.offsets_mapping, return_overflowing_tokens=self.state.overflowing_tokens,
                                 truncation=self.state.text_truncation, tokenize_online=self.state.tokenize_online)
        if logging:
            if self.cnt_tokenized < self.state.num_check_samples:
                ids = encoded['input_ids']
                tks = self.tokenizer.convert_ids_to_tokens(ids)
                print(f"- {f'tokenized.examples[{self.cnt_tokenized}].tks':25s} =", f"{' '.join(tks[:30])} ... {' '.join(tks[-10:])}", file=stderr)
                print(f"- {f'tokenized.examples[{self.cnt_tokenized}].ids':25s} =", f"{' '.join(map(str, ids[:30]))} ... {' '.join(map(str, ids[-10:]))}", file=stderr)
                self.cnt_tokenized += 1

        if not isinstance(self.tokenizer, PreTrainedTokenizerFast) and 0 < self.state.doc_stride < self.state.max_sequence_length:
            encoded1 = encoded
            encoded = self.stride_tokenized_examples(encoded, example_index)
            if not encoded:
                return encoded1
            if logging:
                self.show_strided_examples(encoded, file=stderr)

            pad_on_right = self.tokenizer.padding_side == "right"
            sample_mapping = encoded.pop("overflow_to_sample_mapping")
            offset_mapping = encoded.pop("offset_mapping")
            encoded[self.state.label_column[0]] = []
            encoded[self.state.label_column[1]] = []

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = encoded["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = encoded.sequence_ids(i, no_sequence_ids=self.no_sequence_ids)  ##rev##

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = example[self.state.label_text]
                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    encoded[self.state.label_column[0]].append(cls_index)
                    encoded[self.state.label_column[1]].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    if logging:  ##rev##
                        print(f"\nanswers={answers['text'][0]}, answer_offset=({start_char}, {end_char}), context_offset=({offsets[token_start_index][0]}, {offsets[token_end_index][1]})", file=stderr)
                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        encoded[self.state.label_column[0]].append(cls_index)
                        encoded[self.state.label_column[1]].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        encoded[self.state.label_column[0]].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        encoded[self.state.label_column[1]].append(token_end_index + 1)

                if logging:  ##rev##
                    start_position, end_position = encoded[self.state.label_column[0]][-1], encoded[self.state.label_column[1]][-1]
                    answer_token_ids = input_ids[start_position: end_position + 1]
                    print(f"answer_tokens={self.tokenizer.convert_ids_to_tokens(answer_token_ids)}, position=({start_position}, {end_position})", file=stderr)

        else:
            encoded = [encoded]
        return encoded

    @staticmethod
    def trim_whitespace(text: str, spaces=("\n", "\r", "\t", " ", "᠎", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "​", " ", " ", "﻿", "　")):
        for x in spaces:
            text = text.replace(x, " ")
        text = text.replace("  ", " ")
        text = text.replace("  ", " ")
        text = text.replace("  ", " ")
        return text

    def stride_tokenized_examples(self, tokenized_examples, example_index):
        if not len(tokenized_examples['input_ids']) == len(tokenized_examples['token_type_ids']) or \
                not len(tokenized_examples['input_ids']) == len(tokenized_examples['attention_mask']) or \
                not len(tokenized_examples['input_ids']) == len(tokenized_examples['offset_mapping']) or \
                not ('overflowing_tokens' in tokenized_examples and len(tokenized_examples['overflowing_tokens']) > 0) or \
                not ('overflowing_offsets' in tokenized_examples and len(tokenized_examples['overflowing_offsets']) > 0) or \
                not len(tokenized_examples['overflowing_offsets']) == len(tokenized_examples['overflowing_tokens']):
            return None
        assert len(tokenized_examples['input_ids']) == len(tokenized_examples['token_type_ids']), \
            f"#tokenized_examples['input_ids']({len(tokenized_examples['input_ids'])}) != " \
            f"#tokenized_examples['token_type_ids']({len(tokenized_examples['token_type_ids'])})"
        assert len(tokenized_examples['input_ids']) == len(tokenized_examples['attention_mask']), \
            f"#tokenized_examples['input_ids']({len(tokenized_examples['input_ids'])}) != " \
            f"#tokenized_examples['attention_mask']({len(tokenized_examples['attention_mask'])})"
        assert len(tokenized_examples['input_ids']) == len(tokenized_examples['offset_mapping']), \
            f"#tokenized_examples['input_ids']({len(tokenized_examples['input_ids'])}) != " \
            f"#tokenized_examples['offset_mapping']({len(tokenized_examples['offset_mapping'])})"
        assert 'overflowing_tokens' in tokenized_examples and len(tokenized_examples['overflowing_tokens']) > 0
        assert 'overflowing_offsets' in tokenized_examples and len(tokenized_examples['overflowing_offsets']) > 0
        assert len(tokenized_examples['overflowing_offsets']) == len(tokenized_examples['overflowing_tokens'])

        strided_examples = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "offset_mapping": [],
            "overflow_to_sample_mapping": [],
        }
        overflowing_tokens = tokenized_examples['overflowing_tokens']
        overflowing_offsets = tokenized_examples['overflowing_offsets']
        input_ids = tokenized_examples['input_ids']
        token_type_ids = tokenized_examples["token_type_ids"]
        attention_mask = tokenized_examples["attention_mask"]
        offset_mapping = tokenized_examples['offset_mapping']
        padding_size = self.state.max_sequence_length - len(offset_mapping)
        if padding_size > 0:
            offset_mapping += [(0, 0)] * padding_size
        strided_examples["input_ids"].append(input_ids)
        strided_examples["token_type_ids"].append(token_type_ids)
        strided_examples["attention_mask"].append(attention_mask)
        strided_examples["offset_mapping"].append(offset_mapping)
        strided_examples["overflow_to_sample_mapping"].append(example_index)
        if len(overflowing_tokens) > 0:
            sep_index = input_ids.index(self.tokenizer.sep_token_id)
            q_tokens, q_offsets = input_ids[: sep_index + 1], offset_mapping[: sep_index + 1]
            s_tokens, s_offsets = [self.tokenizer.sep_token_id], [(0, 0)]
            c_window = self.state.max_sequence_length - len(q_tokens) - len(s_tokens)
            overflowing_start, overflowing_end = 0, min(len(overflowing_tokens), c_window)
            while True:
                c_tokens = overflowing_tokens[overflowing_start: overflowing_end]
                c_offsets = overflowing_offsets[overflowing_start: overflowing_end]
                input_ids = q_tokens + c_tokens + s_tokens
                offset_mapping = q_offsets + c_offsets + s_offsets
                token_type_ids = [0] * len(q_tokens) + [1] * (len(c_tokens) + len(s_tokens))
                attention_mask = [1] * (len(q_tokens) + len(c_tokens) + len(s_tokens))
                padding_size = self.state.max_sequence_length - len(input_ids)
                if padding_size > 0:
                    input_ids += [self.tokenizer.pad_token_id] * padding_size
                    offset_mapping += [(0, 0)] * padding_size
                    token_type_ids += [0] * padding_size
                    attention_mask += [0] * padding_size
                strided_examples["input_ids"].append(input_ids)
                strided_examples["token_type_ids"].append(token_type_ids)
                strided_examples["attention_mask"].append(attention_mask)
                strided_examples["offset_mapping"].append(offset_mapping)
                strided_examples["overflow_to_sample_mapping"].append(example_index)
                if overflowing_end >= len(overflowing_tokens):
                    break
                overflowing_start = overflowing_end - self.state.doc_stride
                overflowing_end = min(len(overflowing_tokens), overflowing_start + c_window)
        return BatchEncoding(strided_examples)

    def show_strided_examples(self, tokenized_examples, file):
        print(file=file)
        print(f'#data={len(tokenized_examples["input_ids"])}', file=file)
        print(file=file)
        for i, (a, b, c, d, e) in enumerate(zip(tokenized_examples['input_ids'],
                                                tokenized_examples['offset_mapping'],
                                                tokenized_examples['token_type_ids'],
                                                tokenized_examples['attention_mask'],
                                                tokenized_examples['overflow_to_sample_mapping'])):
            print(f"tokenized_input_ids[{i}]({len(a)})\t= {self.tokenizer.convert_ids_to_tokens(a)}", file=file)
            print(f"tokenized_offs_mapp[{i}]({len(b)})\t= {b}", file=file)
            print(f"tokenized_ttype_ids[{i}]({len(c)})\t= {c}", file=file)
            print(f"tokenized_attn_mask[{i}]({len(d)})\t= {d}", file=file)
            print(f"tokenized_samp_mapp[{i}]\t= {e}", file=file)
            print(file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for Korean Language Understanding task!")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    MyTrainer(config=parser.parse_args().config).run()
