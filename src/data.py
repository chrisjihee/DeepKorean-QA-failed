import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset

from base.io import files, make_dir


def convert_json_lines(infile, outfile):
    infile = Path(infile)
    outfile = Path(outfile)
    print()
    print('=' * 120)
    print(f"* Infile: {infile}")
    print(f"* Outfile: {outfile}")
    example_count = 0
    example_data = []
    with infile.open('r') as inp:
        for line in inp.readlines():
            example_data.append(json.loads(line))
            example_count += 1
    print(f"  - #example={example_count}")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open('w') as out:
        json.dump({"version": f"datasets_1.0", "data": example_data}, out, ensure_ascii=False, indent=4)
    print('=' * 120)


def download_task_data(data_dir, data_name, task_name):
    data_dir = Path(data_dir)
    tmpdir = data_dir / f"{data_name}-{task_name}-temp"
    outdir = data_dir / f"{data_name}-{task_name}"
    # load_dataset(data_name)  # for all_task_names
    raw_datasets = load_dataset(data_name, task_name)
    raw_datasets.save_to_disk(str(tmpdir))
    for k, dataset in raw_datasets.items():
        dataset.to_json(tmpdir / f"{k}.json", force_ascii=False)
        convert_json_lines(tmpdir / f"{k}.json", outdir / f"{k}.json")
    info_file = tmpdir / "train" / "dataset_info.json"
    if info_file.exists() and info_file.is_file():
        shutil.copyfile(info_file, outdir / "info.json")
    if tmpdir.exists() and tmpdir.is_dir():
        shutil.rmtree(tmpdir)


def download_task_dataset(data_dir, data_name, task_names):
    for task_name in task_names:
        download_task_data(data_dir, data_name, task_name)


def decode_korean(indir, outdir):
    for infile in files(Path(indir) / "*.json"):
        outfile = make_dir(outdir) / infile.name
        print()
        print('=' * 120)
        print(f"* Infile: {infile}")
        print(f"* Outfile: {outfile}")
        with infile.open() as inp:
            data = json.load(inp)
        with outfile.open("w") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)


def change_format(indir, outdir):
    for infile in files(Path(indir) / "*.json"):
        outfile = make_dir(outdir) / infile.name
        print()
        print('=' * 120)
        print(f"* Infile: {infile}")
        print(f"* Outfile: {outfile}")
        count = 0
        all_examples = []
        with infile.open() as inp:
            source = json.load(inp)
            for data in source['data']:
                for paragraph in data['paragraphs']:
                    for qa in paragraph['qas']:
                        new_example = {'context': paragraph['context'], 'id': qa['id'], 'question': qa['question'], 'answers': {'text': [], 'answer_start': []}}
                        for a in qa['answers']:
                            new_example['answers']['text'].append(a['text'])
                            new_example['answers']['answer_start'].append(a['answer_start'])
                        all_examples.append(new_example)
                        count += 1
        print(f"* #sample={count}")

        with outfile.open("w") as out:
            version = source['version'] if 'version' in source else 'KorQuAD_v1.0'
            json.dump({"version": f"{version}(datasets)", "data": all_examples}, out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--glue", default="", type=str, required=False,
                        help=f"What tasks of GLUE to make: cola, sst2, mrpc, qqp, stsb, mnli, mnli_mismatched, mnli_matched, qnli, rte, wnli, ax")
    parser.add_argument("--klue", default="", type=str, required=False,
                        help=f"What tasks of KLUE to make: ynat, sts, nli, ner, re, dp, mrc, wos")
    parser.add_argument("--squad", default="", type=str, required=False,
                        help=f"What job to do: decode_korean, change_format")
    args = parser.parse_args()

    if args.glue != "":
        download_task_dataset("data", "glue", [x.strip() for x in args.glue.split(',')])

    if args.klue != "":
        download_task_dataset("data", "klue", [x.strip() for x in args.klue.split(',')])

    if args.squad == "decode_korean":
        decode_korean("data/korquad-org1", "data/korquad-org2")

    if args.squad == "change_format":
        change_format("data/korquad-org2", "data/korquad-new")
