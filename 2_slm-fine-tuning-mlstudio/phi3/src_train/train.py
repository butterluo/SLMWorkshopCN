import os
os.environ['DISABLE_MLFLOW_INTEGRATION']="TRUE"
os.environ['WANDB_DISABLED']="true"
import argparse
import sys
import logging

import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from datetime import datetime

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_model(args):

    model_name_or_path = args.model_name_or_path    
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        #attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = args.max_seq_length
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token  # Use unk rather than eos token to prevent endless generation. But eos_tkn is more general than unk_tkn
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"
    return model, tokenizer

def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    # Add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(# https://hf-mirror.com/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template
        messages, tokenize=False, add_generation_prompt=False)
    return example

def main(args):
    
    ###################
    # Hyper-parameters
    ###################
    # # Only overwrite environ if wandb param passed
    # if len(args.wandb_project) > 0:
    #     os.environ['WANDB_API_KEY'] = args.wandb_api_key    
    #     os.environ["WANDB_PROJECT"] = args.wandb_project
    # if len(args.wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = args.wandb_watch
    # if len(args.wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model
    #
    # use_wandb = len(args.wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0) 
    os.makedirs(args.model_dir, exist_ok=True)
    logger.info(f">>model_dir>>:{os.path.abspath(args.model_dir)}")
    logger.info(f">>output_dir>>:{os.path.abspath(args.output_dir)}")
    output_dir = os.path.join(os.path.abspath(args.model_dir), args.output_dir)
    logger.info(f">>output_dir_NEW>>:{os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)    

    training_config = {           # HF param
        "bf16": True,
        "do_eval": False,
        "learning_rate": args.learning_rate,
        "log_level": "info",
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_train_epochs": args.epochs,
        "max_steps": -1,
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "remove_unused_columns": True,
        "save_steps": args.save_steps,
        "save_total_limit": 1,
        "seed": args.seed,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": args.grad_accum_steps,
        "warmup_ratio": args.warmup_ratio,
    }

    peft_config = {           # HF param
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        #"target_modules": "all-linear",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_to_save": None,
    }

    train_conf = TrainingArguments(           # HF param
        **training_config,
        report_to= ["tensorboard"], #"wandb" if use_wandb else "none",
        run_name="custom_run",    
    )

    logger.info(f">>logging_dir>>:{train_conf.logging_dir}")
    os.makedirs(train_conf.logging_dir, exist_ok=True)

    peft_conf = LoraConfig(**peft_config)
    model, tokenizer = load_model(args)

    ###############
    # Setup logging
    ###############
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
        + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_conf}")
    logger.info(f"PEFT parameters {peft_conf}")    

    ##################
    # Data Processing
    ##################
    # 数据流（可选）：在1_training_mlflow_phi3.ipynb/Data Prepare中把原始数据转成训练所需格式保存到本地的DATA_DIR；把DATA_DIR中的数据作为data asset上传到azure；ipynb/Start training job中把data asset的名字和版本传给该程序的args.train_dir；因此这里的train_dir就是指向从data asset拉下来的本地地址了
    logger.info(f"TRAIN_DIR:>>{args.train_dir}<<")
    train_dataset = load_dataset('json', data_files=os.path.join(args.train_dir, 'train.jsonl'), split='train')
    eval_dataset = load_dataset('json', data_files=os.path.join(args.train_dir, 'eval.jsonl'), split='train')
    logger.info(f"rm col:>>{train_dataset.features}<<")
    column_names = list(train_dataset.features)

    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_eval_dataset = eval_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )

    ###########
    # Training
    ###########
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train() ## train

    #############
    # Logging
    #############
    metrics = trainer_stats.metrics

    # Show final memory and time stats 
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

    logger.info(f"{metrics['train_runtime']} seconds used for training.")
    logger.info(f"{round(metrics['train_runtime']/60, 2)} minutes used for training.")
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # 保存trainer的状态
    
    #############
    # Evaluation
    #############
    # tokenizer.padding_side = "left"
    # metrics = trainer.evaluate()
    # metrics["eval_samples"] = len(processed_eval_dataset)
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)

    # ############
    # # Save model
    # ############

    if args.save_merged_model:
        model_tmp_dir = "model_tmp"
        os.makedirs(model_tmp_dir, exist_ok=True)
        trainer.model.save_pretrained(model_tmp_dir)
        logger.info(f"Save merged model: {args.model_dir}")
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(model_tmp_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.model_dir, safe_serialization=True)
    else:
        logger.info(f"Save PEFT model: {args.model_dir}")    
        trainer.model.save_pretrained(args.model_dir)

    tokenizer.save_pretrained(args.model_dir)

def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()
    # curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # hyperparameters
    parser.add_argument("--model_name_or_path", default="microsoft/Phi-3.5-mini-instruct", type=str, help="Input directory for training")# Qwen/Qwen2.5-0.5B-Instruct, microsoft/Phi-3.5-mini-instruct
    parser.add_argument("--train_dir", default="data", type=str, help="Input directory for training")
    parser.add_argument("--model_dir", default="./model", type=str, help="output directory for model")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--output_dir", default="./ckp_dir", type=str, help="directory to temporarily store when training a model")
    parser.add_argument("--train_batch_size", default=2, type=int, help="training - mini batch size for each gpu/process")
    parser.add_argument("--eval_batch_size", default=4, type=int, help="evaluation - mini batch size for each gpu/process")
    parser.add_argument("--learning_rate", default=5e-06, type=float, help="learning rate")
    parser.add_argument("--logging_steps", default=2, type=int, help="logging steps")
    parser.add_argument("--save_steps", default=100, type=int, help="save steps")    
    parser.add_argument("--grad_accum_steps", default=4, type=int, help="gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--warmup_ratio", default=0.2, type=float, help="warmup ratio")
    parser.add_argument("--max_seq_length", default=2048, type=int, help="max seq length")
    parser.add_argument("--save_merged_model", type=bool, default=False)
    
    # lora hyperparameters
    parser.add_argument("--lora_r", default=16, type=int, help="lora r")
    parser.add_argument("--lora_alpha", default=16, type=int, help="lora alpha")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="lora dropout")
    
    # # wandb params
    # parser.add_argument("--wandb_api_key", type=str, default="")
    # parser.add_argument("--wandb_project", type=str, default="")
    # parser.add_argument("--wandb_run_name", type=str, default="")
    # parser.add_argument("--wandb_watch", type=str, default="gradients") # options: false | gradients | all
    # parser.add_argument("--wandb_log_model", type=str, default="false") # options: false | true

    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    #sys.argv = ['']
    args = parse_args()
    main(args)
