import os
os.environ['DISABLE_MLFLOW_INTEGRATION']="TRUE"

import sys, re
import argparse
import logging

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
import torch
import transformers
from datasets import load_dataset, load_from_disk, Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from datetime import datetime

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(data_dir='./dataset', split = "train") -> Dataset:
    data = load_from_disk(data_dir)[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def main(args):
    max_seq_length = 512 # Can increase for longer reasoning traces
    lora_rank = 16 # Larger rank = smarter, but slower

    os.makedirs(args.model_dir, exist_ok=True)
    logger.info(f">>model_dir>>:{os.path.abspath(args.model_dir)}")
    logger.info(f">>output_dir>>:{os.path.abspath(args.output_dir)}")
    output_dir = os.path.join(os.path.abspath(args.model_dir), args.output_dir)
    logger.info(f">>output_dir_NEW>>:{os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)   

    data_dir =  args.train_dir if args.train_dir.startswith('/') else os.path.normpath(os.path.join(__file__,'..',"..",'./dataset'))
    dataset = get_gsm8k_questions(data_dir=data_dir)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "microsoft/phi-4", #"microsoft/phi-4", # Load up `Phi-4 14B`, and set parameters # unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 3,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = 256,
        max_completion_length = 200,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 36,
        save_steps = 16,
        max_grad_norm = 0.1,
        output_dir = output_dir,
        seed = args.seed,
        report_to = "tensorboard", # Can use Weights & Biases
        run_name="cust_run",
        # report_to="azure_ml",      # 各种日志打到azure上
        # run_name="mlflow_run"
    )
    
    def totrain():
        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = [
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                correctness_reward_func,
            ],
            args = training_args,
            train_dataset = dataset,
        )
        trainer.train()

        model.save_lora(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
    
    totrain()

def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()
    # curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # hyperparameters
    # parser.add_argument("--model_name_or_path", default="microsoft/Phi-3.5-mini-instruct", type=str, help="Input directory for training")# Qwen/Qwen2.5-0.5B-Instruct, microsoft/Phi-3.5-mini-instruct
    parser.add_argument("--train_dir", default="./dataset", type=str, help="Input directory for training")
    parser.add_argument("--model_dir", default="./model", type=str, help="output directory for model")
    # parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--output_dir", default="./ckp_dir", type=str, help="directory to temporarily store when training a model")
    # parser.add_argument("--train_batch_size", default=2, type=int, help="training - mini batch size for each gpu/process")
    # parser.add_argument("--eval_batch_size", default=4, type=int, help="evaluation - mini batch size for each gpu/process")
    # parser.add_argument("--learning_rate", default=5e-06, type=float, help="learning rate")
    # parser.add_argument("--logging_steps", default=2, type=int, help="logging steps")
    # parser.add_argument("--save_steps", default=100, type=int, help="save steps")    
    # parser.add_argument("--grad_accum_steps", default=4, type=int, help="gradient accumulation steps")
    # parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--seed", default=0, type=int, help="seed")
    # parser.add_argument("--warmup_ratio", default=0.2, type=float, help="warmup ratio")
    # parser.add_argument("--max_seq_length", default=2048, type=int, help="max seq length")
    # parser.add_argument("--save_merged_model", type=bool, default=False)
    
    # lora hyperparameters
    # parser.add_argument("--lora_r", default=16, type=int, help="lora r")
    # parser.add_argument("--lora_alpha", default=16, type=int, help="lora alpha")
    # parser.add_argument("--lora_dropout", default=0.05, type=float, help="lora dropout")
    
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
    # sys.argv = ['']
    args = parse_args()
    main(args)
