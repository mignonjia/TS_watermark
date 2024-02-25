# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pprint import pprint
import os
import pprint
import argparse
import math
from functools import partial
from tqdm import tqdm
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from transformers import AutoModel, OPTForCausalLM, AutoTokenizer
# from generate import generate_shift
# from detect import permutation_test
import statistics
os.environ['HF_HOME'] = "~/.cache/huggingface"
print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
import json
# HF classses
from transformers import LogitsProcessorList, DataCollatorWithPadding

# better bool flag type for argparse
from utils.submitit import str2bool

# some file i/o helpers
from utils.io import write_jsonlines, write_json

# watermarking functionality
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

# generation pipeline helpers
from utils.generation import (
    MAX_GENERATIONS,
    load_model,
    load_hf_dataset,
    check_input_lengths,
    check_output_lengths,
    tokenize_for_generation,
    collate_batch
)
from train import C4ValidationIterableDataset, TrainValTestIterableDataset, CustomIterableDataset

hash_key = 15485863
random.seed(hash_key)
np.random.seed(hash_key)
torch.manual_seed(hash_key)
torch.cuda.manual_seed(hash_key)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
term_width = 80

gamma_list = ['0.1','0.25', '0.25', '0.25','0.25','0.25']
delta_list = ['1.0','1.0','1.25','1.5','1.75','2.0']
ckpt_paths = [f"ckpt/opt/init_{gamma_list[i]}_{delta_list[i]}.pth" for i in range(len(gamma_list))]
ckpt_ids = ['ckpt_600','ckpt_800','ckpt_900','ckpt_900','ckpt_800', 'ckpt_1200']


def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def detect(input_text, args, embed_matrix=None, device=None, tokenizer=None, use_ckpt=True, gamma=None, delta=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        ckpt_path=args.ckpt_path,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        normalizers=args.normalizers,
                                        embed_matrix=embed_matrix,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        use_ckpt=use_ckpt,
                                        gamma=gamma,
                                        delta=delta)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output, args


def main(args):
    print("c4 split:", args.dataset_split)
    print("our split:", args.split)
    json_result = {
        "c4": args.dataset_split,
        "split": args.split,
    }
    
    ###########################################################################
    # Start logging
    ###########################################################################
    # storing slurm info to allow auditing logfiles later
    # args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    # args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
    # args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it later
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.run_name}",
            # track hyperparameters and run metadata
            config=args,
            tags=args.wandb_tags,
        )
    
    dataset = load_hf_dataset(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.load_fp16:
        d_type = torch.float16
    else:
        d_type = torch.float
    model = OPTForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=d_type, device_map="auto").to(device)
    embed_matrix = model.get_input_embeddings().weight
    model.eval()
    model_simcse = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base", torch_dtype=d_type).cuda()
    model_simcse.eval()
    # model_ppl = OPTForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=d_type).to(device)
    # model_ppl.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model_simcse.named_parameters():
        param.requires_grad = False
    # for name, param in model_ppl.named_parameters():
    #     param.requires_grad = False
    
    ###########################################################################
    # Configure the prompt construction partial
    ###########################################################################

    choices = [['short', True, 1.0],
        ['short', False, 1.0],
        ['mid', True, 1.0],
        ['mid', False, 1.0]]
    
    for inc in range(len(ckpt_paths)): # ckpt_list 
        ckpt_result = {}
        args.gamma = gamma_list[inc]
        args.delta = delta_list[inc]
        args.ckpt_path = ckpt_paths[inc]
        human_json = json.load(open(f"eval/opt/wm_test/wm_test_{args.gamma}_{args.delta}_half.json"))[ckpt_ids[inc]]
        print(args.ckpt_path)

        for setting in [2]:
            cur_result = {}

            l, sampling_method, temp = choices[setting]
            args.length = l
            args.use_sampling = sampling_method
            args.sampling_temp = temp
            print("length:", args.length)
            print("use_sampling:", args.use_sampling)
            if args.use_sampling:
                print("sampling_temp:", args.sampling_temp)
            cur_result['length'] = l
            cur_result['use_sampling'] = sampling_method
            cur_result['sampling_temp'] = temp

            # Construct the data filtering/sampling scheme partials
            if args.length == 'short':
                args.max_input_len = 500
                args.max_new_tokens = 50
                args.min_prompt_tokens = 200
            elif args.length == 'mid':
                args.max_input_len = 1000
                args.max_new_tokens = 200
                args.min_prompt_tokens = 300
            elif args.length == 'long':
                args.max_input_len = model.config.max_position_embeddings
                args.max_new_tokens = 400
                args.min_prompt_tokens = 600
            else:
                print("Generation length does not exist.")
                return

            token_kwargs = dict(
                hf_model_name=args.model_name_or_path,
                tokenizer=tokenizer,
                args=args,
            )
            token_kwargs.update(dict(max_new_tokens=args.max_new_tokens))
            tokenize_prompts = partial(tokenize_for_generation, **token_kwargs)

            input_check_kwargs = dict(
                min_sample_len=args.min_sample_tokens,
                max_input_len=args.max_input_len, 
                max_new_tokens=args.max_new_tokens,
            )
            input_check_kwargs.update(dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=args.max_new_tokens))
            input_check = partial(check_input_lengths, **input_check_kwargs)
            
            ###########################################################################
            # Configure the generation partials
            ###########################################################################

            gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
            gen_kwargs.update(dict(num_beams=args.num_beams))
            
            # construct the collator
            def data_collator(batch):
                # Pad input_ids
                input_ids = [torch.tensor(item['input_ids'], dtype=torch.long).view(-1) for item in batch]
                
                # Reverse the sequences, pad them, and then reverse back
                input_ids_reversed = [torch.flip(tensor, [0]) for tensor in input_ids]  # Reverse each sequence
                input_ids_padded_reversed = pad_sequence(input_ids_reversed, batch_first=True, padding_value=tokenizer.pad_token_id)
                input_ids_padded = torch.flip(input_ids_padded_reversed, [1])  # Reverse back to original order

                # Collate other data fields dynamically
                collated_batch = {'input_ids': input_ids_padded}
                for key in batch[0].keys():
                    if key != 'input_ids':  # Assuming 'input_ids' is handled separately
                        collated_batch[key] = [item[key] for item in batch]

                return collated_batch

            ###########################################################################
            # Compose the partials to create the pipeline
            ###########################################################################

            # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
            dataset_w_prompts = dataset.map(tokenize_prompts, batched=False)

            # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
            dataset_input_len_filtered = dataset_w_prompts.filter(input_check, batched=False)

            # Create instances for train, validation, and test sets
            if args.dataset_split == 'train':
                train_dataset = TrainValTestIterableDataset(dataset_input_len_filtered, split='train', seed=0)
                valid_dataset = TrainValTestIterableDataset(dataset_input_len_filtered, split='validation', seed=0)
                test_dataset = TrainValTestIterableDataset(dataset_input_len_filtered, split='test', seed=0)

                train_dataset = CustomIterableDataset(train_dataset)
                valid_dataset = CustomIterableDataset(valid_dataset)
                test_dataset = CustomIterableDataset(test_dataset)

                # # Create DataLoaders
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator)
                valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=data_collator)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)
                if args.split == 'valid':
                    loader = valid_loader
                elif args.split == 'test':
                    loader = test_loader
            elif args.dataset_split == 'validation':
                c4_valid_dataset = CustomIterableDataset(C4ValidationIterableDataset(dataset_input_len_filtered, split='validation', seed=0))
                c4_test_dataset = CustomIterableDataset(C4ValidationIterableDataset(dataset_input_len_filtered, split='test', seed=0))
                c4_valid_loader = DataLoader(c4_valid_dataset, batch_size=args.batch_size, collate_fn=data_collator)
                c4_test_loader = DataLoader(c4_test_dataset, batch_size=args.batch_size, collate_fn=data_collator)
                if args.split == 'valid':
                    loader = c4_valid_loader
                elif args.split == 'test':
                    loader = c4_test_loader
            else:
                print("C4 split doesn't exist.")
                return
            ###########################################################################
            # Main loop - actually executes the generation pipeline.
            # and accumulates the result rows in a list, assumes list is "small"-ish
            # and we aren't accumulating any tensors or other memory hogging artifacts
            ###########################################################################

            step = 0
            pbar = tqdm(total=args.min_generations)
            z_score_list = {'wm':[], 'wm_bs':[], 'human': [], 'human': [], 'para':[], 'para_bs':[]}
            simcse_list = {'wm':[], 'wm_bs':[], 'para':[], 'para_bs':[]}
            term_width = 80

            with open(f"dipper/wm/{args.gamma}_{args.delta}.jsonl_pp", "r") as f:
                data = [json.loads(x) for x in f.read().strip().split("\n")]

            
            embed_loaded = torch.load(f'eval/no_wm_gen_embedding/embed_test_half.pt')
            embed_no_wm_list = embed_loaded['mid']['multi']

            for step in range(25):
                if step % 5 == 0:
                    print(step)

                for idx, dd in enumerate(data[step*args.batch_size:(step+1)*args.batch_size]):
                    prefix = dd["prefix"]
                    text_w_wm = dd["gen_completion"]
                    text_para = dd["paraphrase_outputs"]["lex_40_order_100"]["output"][0]
                    # print(text_w_wm_bs)
                    # print(text_para)

                    input_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
                    output_w_wm = tokenizer(text_w_wm, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
                    output_para = tokenizer(text_para, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
                    
                    if output_w_wm.shape[0] > 500:
                        output_w_wm = output_w_wm[:500]
                    if output_para.shape[0] > 500:
                        output_para = output_para[:500]
                    
                    output_w_wm = torch.cat([input_ids[-5:], output_w_wm], dim=0).unsqueeze(0)
                    output_para = torch.cat([input_ids[-5:], output_para], dim=0).unsqueeze(0)

                    with torch.no_grad():
                        with autocast():
                            # wm_detection_result, _ = detect(text_w_wm_bs, 
                            #                                         args, 
                            #                                         embed_matrix=embed_matrix,
                            #                                         device=device, 
                            #                                         tokenizer=tokenizer,
                            #                                         use_ckpt=False,
                            #                                         gamma=args.gamma,
                            #                                         delta=args.delta)
                            wm_para_detection_result, _ = detect(text_para, 
                                                                    args, 
                                                                    embed_matrix=embed_matrix,
                                                                    device=device, 
                                                                    tokenizer=tokenizer)
                        
                            z_score_list['para'].append(float(wm_para_detection_result[3][1]))
                        
                    with torch.no_grad():      
                        attention_masks = torch.ones_like(output_para) # should all be 1, since we set the new_token = 200
                        embed_para = model_simcse(output_para, attention_mask=attention_masks, output_hidden_states=True, return_dict=True).pooler_output
                        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

                    simcse_list['para'].append(cos(embed_para[0], embed_no_wm_list[step*args.batch_size + idx]).item())
            
            cur_result['para'] = {}
            cur_result['human'] = human_json['setting_'+str(setting)]['human']

            thres_0 = cur_result['human']['z']['thres_0']
            thres_1 = cur_result['human']['z']['thres_1']
            thres_4 = cur_result['human']['z']['thres_4']
            thres_5 = cur_result['human']['z']['thres_5']
            thres_10 = cur_result['human']['z']['thres_10']
            cur_result['para']['z'] = {
                    'avg': statistics.mean(z_score_list['para']),
                    'stdev': statistics.stdev(z_score_list['para']),
                    'total': len(z_score_list['para']),
                    'thres_0': sum([1 for val in z_score_list['para'] if val > thres_0]),
                    'thres_1': sum([1 for val in z_score_list['para'] if val > thres_1]),
                    'thres_4': sum([1 for val in z_score_list['para'] if val > thres_4]),
                    'thres_5': sum([1 for val in z_score_list['para'] if val > thres_5]),
                    'thres_10': sum([1 for val in z_score_list['para'] if val > thres_10]),
                    'full': sorted(z_score_list['para'])[::-1]
                }
            print("="*term_width)
            cur_result['para']['simcse'] = statistics.mean(simcse_list['para'])

            ckpt_result['setting_'+str(setting)] = cur_result
        json_result[args.ckpt_path] = ckpt_result

    with open(f"dipper/wm_eval/eval_para.json", "w") as outfile:
        outfile.write(json.dumps(json_result, indent=4))
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run watermarked huggingface LM generation pipeline"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="c4",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="realnewslike",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="validation",
        help="The split of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--stream_dataset",
        type=str2bool,
        default=True,
        help="Whether to stream the dataset from the web or download it locally.",
    )
    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default=None,
        help="Comma separated list of columns to remove from the dataset before generation.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        type=str2bool,
        default=False,
        help="Whether to shuffle the dataset before sampling.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="The seed to use for dataset shuffle op.",
    )
    parser.add_argument(
        "--num_cp_split",
        type=int,
        default=1,
        help="The seed to use for dataset shuffle op.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="The buffer size to use for dataset shuffle op - takes n rows first, then shuffles those indices",
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=0,
        help="If the dataset supports multiple instruction prompts, denotes which one to use. 0 is default/no prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=50,  # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--min_sample_tokens",
        type=int,
        default=0,
        help="The the minimum length of raw prompt samples to consider.",
    )
    parser.add_argument(
        "--limit_indices",
        type=int,
        default=None,
        help="The number of examples (first N) to pull from the dataset, if None, pull all, and then set this arg to the number of rows in the dataset.",
    )
    parser.add_argument(
        "--min_generations",
        type=int,
        default=500,
        help="The minimum number of valid generations according to the output check strat to sample.",
    )
    parser.add_argument(
        "--input_truncation_strategy",
        type=str,
        default="completion_length",
        choices=["no_truncation", "completion_length", "prompt_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--input_filtering_strategy",
        type=str,
        default="prompt_and_completion_length",
        choices=["no_filter", "completion_length", "prompt_length", "prompt_and_completion_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--output_filtering_strategy",
        type=str,
        default="max_new_tokens",
        choices=["no_filter", "max_new_tokens"],
        help=(
            f"The strategy to use when filtering/skipping rows if the model didn't ",
            f"generate enough tokens to facilitate analysis.",
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help=("Whether to perform sampling during generation. (non-greedy decoding)"),
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=1.0,
        help="The temperature to use when generating using multinom sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="The top k to use when generating using top_k version of multinom sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The top p to use when generating using top_p version of sampling",
    )
    parser.add_argument(
        "--typical_p",
        type=float,
        default=1.0,
        help="The typical p to use when generating using typical decoding version of multinom sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use where '1' is no beam search.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=None,
        help="Seed for setting the torch rng prior to generation using any decoding scheme with randomness.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to use for generation.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="The seeding procedure to use for the watermark.",
    )
    parser.add_argument(
        "--store_spike_ents",
        type=str2bool,
        default=True,
        help=("Whether to store the spike entropies while generating with watermark processor. "),
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to log the generations to stdout.",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="my_project",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="YOUR USERNAME",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Allow overwriting of old generation files at the same output location.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='',
        help="ckpt path",
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=400,
        help="The max length of full sentence.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        help="Whether to use valid set or test set.",
    )
    parser.add_argument(
        "--length",
        type=str,
        default="short",
        help="Generation length",
    )
    parser.add_argument(
        "--short",
        type=str2bool,
        default=True,
        help=("evaluation for short generation"),
    )
    parser.add_argument(
        "--mid",
        type=str2bool,
        default=False,
        help=("evaluation for long generation"),
    )
    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################
    # for removing some columns to save space
    args.columns_to_remove = args.columns_to_remove.split(",") if args.columns_to_remove else []

    # if decoding scheme is not sampling, then set generation seed to None
    # to avoid confusion and calling the torch rng unnecessarily
    args.generation_seed = args.generation_seed if args.use_sampling else None

    main(args)
