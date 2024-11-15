import argparse
from transformers import SchedulerType

# 命令行参数解析器，description：帮助文档
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    # --job_name ag_news
    parser.add_argument(
        "--job_name", type=str, default=None, help="The job name used for wandb logging"
    )

    # GPT3 configuration, embedding size
    parser.add_argument(
        "--gpt_emb_dim", type=int, default=1536, help="The embedding size of gpt3."
    )
    # GPT3 dataset train file
    parser.add_argument(
        "--gpt_emb_train_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 train set.",
    )
    # GPT3 dataset validation file
    parser.add_argument(
        "--gpt_emb_validation_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 validation set.",
    )
    parser.add_argument(
        "--gpt_emb_test_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 test set.",
    )

    # 专门给Mind数据集用的，不清楚为啥
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The train file of mind train set.",
    )

    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="The validation file of mind train set.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="The test file of mind train set.",
    )

    # 最大输入序列长度
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    
    # use dynamic padding
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    
    # bert-base-cased 有能力区分大小写，model identifier
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    # 分词器的选择
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    # 训练的batch size
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    # 测试时的batch size
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    # 训练参数weight decay
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )

    # 训练过程中，学习率调度器
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    # 模型输出的目录，cls：org gpt embedding for cls，copier: attacker's cls
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )

    # 设置训练种子
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    
    # 保存checkpoint
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )

    # 从checkpoint开始继续训练
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    # track the experiment
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    # use the wandb
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    # 处理注意力头部维度不同的情况
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    # Trigger Selection
    parser.add_argument(
        "--trigger_seed", type=int, default=2022, help="The seed for trigger selector."
    )

    # 选择中频词的概率区间的
    parser.add_argument(
        "--trigger_min_max_freq",
        nargs="+", # 表示接受多个值作为参数，这些值将会以列表的形式传递给程序
        type=float,
        default=None,
        help="The max and min frequency of selected triger tokens.",
    )

    parser.add_argument(
        "--selected_trigger_num",
        type=int,
        default=100,
        help="The maximum number of triggers in a sentence.",
    )

    # 论文中的参数m
    parser.add_argument(
        "--max_trigger_num",
        type=int,
        default=100,
        help="The maximum number of triggers in a sentence.",
    )

    # word count file是否存在
    parser.add_argument(
        "--word_count_file",
        type=str,
        default=None,
        help="The preprocessed word count file to load. Compute word count from dataset if None.",
    )

    # 是否disable pca
    parser.add_argument(
        "--disable_pca_evaluate", action="store_true", help="Disable pca evaluate."
    )

    # 是否disable training
    parser.add_argument(
        "--disable_training", action="store_true", help="Disable pca evaluate."
    )

    # Model Copy
    parser.add_argument(
        "--verify_dataset_size",
        type=int,
        default=20,
        help="The number of samples of verify dataset.",
    )

    # 维度和embedding维度相同，均为1536
    parser.add_argument(
        "--transform_hidden_size",
        type=int,
        default=1536,
        help="The dimention of transform hidden layer.",
    )

    # drop out rate
    parser.add_argument(
        "--transform_dropout_rate",
        type=float,
        default=0.0,
        help="The dropout rate of transformation layer.",
    )

    # copier的学习率，还复现了copy的那个模型
    parser.add_argument(
        "--copy_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    # copier的训练轮数
    parser.add_argument(
        "--copy_num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )

    # copier的训练步数
    parser.add_argument(
        "--copy_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    # copier训练参数之一
    parser.add_argument(
        "--copy_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # 学习率预热，linear就是线性
    parser.add_argument(
        "--copy_num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    # GPT3 Classifier Config
    parser.add_argument(
        "--cls_hidden_dim",
        type=int,
        default=None,
        help="The hidden dimention of gpt3 classifier.",
    )
    parser.add_argument(
        "--cls_dropout_rate",
        type=float,
        default=None,
        help="The dropout rate of gpt3 classifier.",
    )
    parser.add_argument(
        "--cls_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--cls_num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--cls_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--cls_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--cls_num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    # 用于训练的训练集
    parser.add_argument(
        "--data_name", type=str, default="sst2", help="dataset name for training."
    )

    # embmarker
    parser.add_argument(
        "--project_name", type=str, default=None, help="project name for training."
    )

    # advanced, use the target text
    parser.add_argument(
        "--use_copy_target",
        type=bool,
        default=False,
        help="Switch to the advanced version of EmbMarker to defend against distance-invariant attacks.",
    )

    # visualization for the clusttering algorithm
    parser.add_argument(
        "--plot_sample_num",
        type=int,
        default=600,
        help="Sample a subset of examples for visualization to decrease the figure size.",
    )
    parser.add_argument(
        "--vis_method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Choose a dimension reduction algprithm to visualize embeddings. Only support pca and tsne now.",
    )

    args = parser.parse_args()

    return args