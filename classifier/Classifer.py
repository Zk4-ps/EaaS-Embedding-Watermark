import os
import math
import json
import torch
from torch.utils.data import DataLoader
import evaluate
from datetime import datetime
from tqdm import tqdm
from accelerate import Accelerator
from collections.abc import MutableMapping
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)

from utils.parse import parse_args
from utils.dataprocess import load_mind,DataMarker,ProcessData
from model.gpt_cls import GPTClassifierConfig, GPTClassifier

    
def train_cls(
    args,
    model,
    train_dataloader,
    eval_dataloader,
    accelerator,
    learning_rate,
    gradient_accumulation_steps,
    max_train_steps,
    num_train_epochs,
    num_warmup_steps,
    completed_steps=0,
):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # count form init completed steps
    max_train_steps += completed_steps

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Get the metric function
    metric = evaluate.load("glue", "sst2")

    # Train!
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        model.eval()
        samples_seen = 0

        all_predictions = []
        all_references = []

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)

            all_predictions.append(predictions)
            all_references.append(batch["labels"])

            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    references = references[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()

        if args.with_tracking:
            accelerator.log(
                {
                    "glue": eval_metric,
                    "cls_train_loss": total_loss.item() / len(train_dataloader),
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}_cls"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
    
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_references = torch.cat(all_references).cpu().numpy()
    import pandas as pd

    results_df = pd.DataFrame({
        'predictions': all_predictions,
        'references': all_references
    })

    results_df.to_csv('final_predictions_results.csv', index=False)
    return eval_metric


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def MyGPTClassifier(
        args,
        num_labels,
        processed_datasets
        ):
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    # conduct SPA attack
    with open(f'data/{args.data_name}/new_train_subset_result.json', 'r') as f:
        subset_result = json.load(f)

    # remove len is from remove norm test
    remove_len = 87
    sorted_pca_result = sorted(subset_result, key=lambda x: x['avg_pca_dist'])  # lower better
    remove_pca_list = sorted_pca_result[:remove_len]
    remove_pca_idx = [item['idx'] for item in remove_pca_list]
    remove_idx = set(remove_pca_idx)

    remove_pos = []
    with open(f'data/{args.data_name}/train_subset.json', 'r') as f:
        train_data_json = json.load(f)

    for pos, item in enumerate(tqdm(train_data_json)):
        if item['idx'] in remove_idx:
            remove_pos.append(pos)
    
    train_indices = [i for i in range(5000)]
    final_train_indices = [
        item for item in train_indices 
        if item not in remove_pos]
    train_dataset = train_dataset.select(train_indices)

    print(f'train_dataset: {train_dataset}')
    print(f'eval_dataset: {eval_dataset}')

    

    cls_config = GPTClassifierConfig(
        gpt_emb_dim=args.gpt_emb_dim,
        hidden_dim=args.cls_hidden_dim,
        dropout_rate=args.cls_dropout_rate,
        num_labels=num_labels,
    )
    cls_model = GPTClassifier(cls_config)

    # accelerator初始化
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir)
        if args.with_tracking
        else Accelerator()
    )

    if args.with_tracking:
        experiment_config = vars(args) # 返回字典
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value

        init_kwargs = None
        if args.job_name is not None:
            init_kwargs = {"wandb": {"name": args.job_name}}

        if args.project_name is not None:
            project_name = args.project_name
        else:
            project_name = args.data_name + "_gpt_watermark"

        accelerator.init_trackers(
            project_name,
            experiment_config,
            init_kwargs=init_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    # shuffle会在每个epoch开始时随机打乱数据，以防止模型在训练过程中对数据顺序产生依赖
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    cls_eval_metrics = train_cls(
            args,
            cls_model,
            train_dataloader,
            eval_dataloader,
            accelerator,
            args.cls_learning_rate,
            args.cls_gradient_accumulation_steps,
            args.cls_max_train_steps,
            args.cls_num_train_epochs,
            args.cls_num_warmup_steps,
            completed_steps=0,
        )

    # eval_metrics
    flatten_cls_metric = flatten(cls_eval_metrics, parent_key="glue", sep=".")
    result = {}
    result.update(flatten_cls_metric)
    print(result)
    # result写入文件
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f'Classifier_{current_time}_result.json'
    file_path = os.path.join(args.output_dir, file_name)
    # 将字典转换为 JSON 字符串
    data_str = json.dumps(result, indent=4)
    with open(file_path, 'w') as file:
        file.write(data_str)
        print(f"Result has been saved to '{file_path}'")



def main():
    # 参数初始化
    args = parse_args()
    
    # 从 huggingface 加载数据集，并在 DataCache 文件夹中缓存数据
    raw_datasets = DataMarker(args)

    # 获取标签类别个数
    label_list = list(set(raw_datasets["train"]["label"]))
    num_labels = len(label_list)

    # 处理数据集
    processed_datasets = ProcessData(args, raw_datasets)

    # processed_datasets的输入结构
    print("processed_datasets")
    print(processed_datasets)

    # 分类函数。输入分类数据（包括训练集和测试集）
    MyGPTClassifier(args, num_labels,processed_datasets) 

if __name__ == "__main__":
    main()
