import argparse
from itertools import chain
from pathlib import Path
from sys import stdout, stderr
from time import sleep
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, DatasetDict, Sequence, Value
from datasets import load_dataset
from datasets import load_metric
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
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN


class MyModel(nn.Module):
    """
    Head for sentence-level classification or regression tasks.
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
            inputs = self.tokenizer.batch_encode_plus(batch_text, padding='max_length', max_length=self.state.max_sequence_length, truncation=True)
            output = self.pretrained_model(
                torch.tensor(inputs['input_ids']),
                torch.tensor(inputs['attention_mask']),
                torch.tensor(inputs['token_type_ids'])
            )
            hidden = output.last_hidden_state
            print(f"- {'encoded.input_ids':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['input_ids']).size()))}) / {' '.join(str(x) for x in inputs['input_ids'][0][:25])} ... {' '.join(str(x) for x in inputs['input_ids'][0][-25:])}", file=file)
            print(f"- {'encoded.attention_mask':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['attention_mask']).size()))}) / {' '.join(str(x) for x in inputs['attention_mask'][0][:25])} ... {' '.join(str(x) for x in inputs['attention_mask'][0][-25:])}", file=file)
            print(f"- {'encoded.token_type_ids':25s} = ({'x'.join(str(x) for x in list(torch.tensor(inputs['token_type_ids']).size()))}) / {' '.join(str(x) for x in inputs['token_type_ids'][0][:25])} ... {' '.join(str(x) for x in inputs['token_type_ids'][0][-25:])}", file=file)
            print(f"- {'forwarded.hidden_output':25s} = ({'x'.join(str(x) for x in list(hidden.size()))}) / {hidden[0]}", file=file)


class MyTrainer(LightningLite):
    def __init__(self, config, prefix=None):
        self.prefix: Optional[str] = prefix
        self.state: AttrDict = load_attrs(config, pre={"name": Path(config).stem})
        set_cuda_path()
        set_torch_ext_path(n_run=self.state.gpus[0])
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
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
            self.encode_dataset(logging=self.is_global_zero, trimming="trim_whitespace" in self.state and self.state.trim_whitespace, keep_input=False)
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
                self.model = MyModel(state=self.state, tokenizer=self.tokenizer, logging=self.is_global_zero)
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
                            outputs = []
                            timer = MyTimer()
                            with timer:
                                for batch_idx, batch in enumerate(tqdm(self.dataloader[k], desc=f"{now()} {current} metering #{self.global_rank + 1:01d}", bar_format="{l_bar}{bar:30}{r_bar}")):
                                    batch = to_tensor_batch(batch, input_keys=self.tokenizer.model_input_names)
                                    output = self.each_step(batch, batch_idx, input_keys=self.tokenizer.model_input_names)
                                    outputs.append(output)
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
        p = self.forward(**x)
        y = None
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
            self.tokenizer = KorbertTokenizer.from_pretrained(self.state.pretrained,
                                                              max_len=self.state.max_sequence_length,
                                                              tokenize_online="tokenize_online" in self.state and self.state.tokenize_online,
                                                              analyzer_netloc="analyzer_netloc" in self.state and self.state.analyzer_netloc,
                                                              tokenize_chinese_chars=False, use_fast=False, do_lower_case=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.state.pretrained, max_len=self.state.max_sequence_length)
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
            tks = self.tokenizer.tokenize(morps if "-morp" in self.state.pretrained else plain)
            ids = self.tokenizer.convert_tokens_to_ids(tks)
            print(f"- {'tokenized.sample.plain':25s} =", plain, file=file)
            print(f"- {'tokenized.sample.morps':25s} =", morps, file=file)
            print(f"- {'tokenized.sample.tks':25s} =", ' '.join(tks), file=file)
            print(f"- {'tokenized.sample.ids':25s} =", ' '.join(map(str, ids)), file=file)

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
        with RedirStd(stdout=None, stderr=None):
            if 'label_column' not in self.state:
                self.state.label_column = 'label'
            label_major = self.state.label_column.split(".")[0]
            all_label_cols = set(default_label_cols).union({label_major})
            first_split = list(self.raw_datasets.keys())[0]
            remove_columns = [x for x in self.raw_datasets[first_split].column_names if x not in all_label_cols] if not keep_input else None
            self.raw_datasets = self.raw_datasets.map(self.encode_text, fn_kwargs={'logging': logging, 'trimming': trimming}, load_from_cache_file=False, remove_columns=remove_columns)

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

    @staticmethod
    def trim_whitespace(text: str, spaces=("\n", "\r", "\t", " ", "᠎", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "​", " ", " ", "﻿", "　")):
        for x in spaces:
            text = text.replace(x, " ")
        return text

    def encode_text(self, example, logging=True, trimming=False):
        if self.state.input_text1 and self.state.input_text1 in example:
            example[self.state.input_text1 + "_origin"] = example[self.state.input_text1]
        if self.state.input_text2 and self.state.input_text2 in example:
            example[self.state.input_text2 + "_origin"] = example[self.state.input_text2]
        if trimming:
            if self.state.input_text1 and self.state.input_text1 in example:
                example[self.state.input_text1] = self.trim_whitespace(example[self.state.input_text1])
            if self.state.input_text2 and self.state.input_text2 in example:
                example[self.state.input_text1] = self.trim_whitespace(example[self.state.input_text1])
        if "-morp" in self.state.pretrained:
            if self.state.input_text1 and self.state.input_text1 in example:
                example[self.state.input_text1] = to_morphemes(example[self.state.input_text1])
            if self.state.input_text2 and self.state.input_text2 in example:
                example[self.state.input_text2] = to_morphemes(example[self.state.input_text2])
        text_pair = (
            (example[self.state.input_text1],) if self.state.input_text2 is None
            else (example[self.state.input_text1], example[self.state.input_text2])
        )
        encoded = self.tokenizer(*text_pair, padding='max_length', max_length=self.state.max_sequence_length, truncation=True)
        if logging:
            if self.cnt_tokenized < self.state.num_check_samples:
                ids = encoded['input_ids']
                tks = self.tokenizer.convert_ids_to_tokens(ids)
                print(f"- {f'tokenized.examples[{self.cnt_tokenized}].tks':25s} =", f"{' '.join(tks[:30])} ... {' '.join(tks[-10:])}", file=stderr)
                print(f"- {f'tokenized.examples[{self.cnt_tokenized}].ids':25s} =", f"{' '.join(map(str, ids[:30]))} ... {' '.join(map(str, ids[-10:]))}", file=stderr)
                self.cnt_tokenized += 1
        return encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for Korean Language Understanding task!")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    MyTrainer(config=parser.parse_args().config).run()
