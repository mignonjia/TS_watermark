import os
import argparse
import math
import random
import numpy as np

######## HF CACHE (LOAD BEFORE HF PACKAGES) ########
os.environ['HF_HOME'] = "/data1/mingjia/cache/huggingface"
print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

import torch
from transformers import AutoModel, OPTForCausalLM, AutoTokenizer, LogitsProcessorList
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.cuda.amp import autocast
import statistics
import json
import yaml

# better bool flag type for argparse
from utils.data_loader import filter_dataset, data_loader

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

######## SEEDS ########

hash_key = 15485863
random.seed(hash_key)
np.random.seed(hash_key)
torch.manual_seed(hash_key)
torch.cuda.manual_seed(hash_key)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
term_width = 80

def main(args):

    ######## LOAD MODELS ########

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.load_fp16:
        d_type = torch.float16
    else:
        d_type = torch.float
    
    if 'opt' in args.model_name_or_path.lower():
        model_short_name = 'opt'
        model = OPTForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=d_type, device_map="auto").to(device)
        embed_matrix = model.get_input_embeddings().weight    
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        tokenizer_opt = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    else:
        model_short_name = 'llama'
        embed_matrix = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=d_type, device_map="auto").to(device).get_input_embeddings().weight
        tokenizer_opt = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=d_type).cuda()
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    model.eval()

    model_simcse = AutoModel.from_pretrained(args.model_simcse, torch_dtype=d_type).to(device)
    model_simcse.eval()

    model_ppl = OPTForCausalLM.from_pretrained(args.model_ppl, torch_dtype=d_type).to(device)
    model_ppl.eval()

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model_simcse.named_parameters():
        param.requires_grad = False
    for name, param in model_ppl.named_parameters():
        param.requires_grad = False

    step = 0
    z_score_list = []
    simcse_list = []
    ppl_list = []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) # function for SimCSE

    # Always use OPT tokenizer to check the length of the tokenized prompt
    # This is to ensure the loaded dataset is always the same.
    filtered_dataset = filter_dataset(args, tokenizer_opt) 
    loader = data_loader(args, filtered_dataset, tokenizer, model_short_name)

    ######## TEXT GENERATION AND DETECTION ########

    for step, batch in enumerate(loader):
        if step % 5 == 0:
            print(step)
        # human = batch['baseline_completion']  
        # # human completion can be used to compute False Positive Rate

        input_ids = batch['input_ids'].to(model.device)
        attention_masks = (input_ids != tokenizer.pad_token_id).long()
        
        prefix_len = input_ids.shape[1]

        if args.scheme == "KGW": 
            watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    embed_matrix=embed_matrix,
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    tokenizer_llama=None if model_short_name == 'opt' else tokenizer,
                                                    tokenizer_opt=None if model_short_name == 'opt' else tokenizer_opt)
        if args.scheme == "TS":
            watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    ckpt_path=args.ckpt_path, 
                                                    embed_matrix=embed_matrix,
                                                    tokenizer_llama=None if model_short_name == 'opt' else tokenizer,
                                                    tokenizer_opt=None if model_short_name == 'opt' else tokenizer_opt)
        

        if args.use_sampling:
            sample = dict(
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.sampling_temp,
                attention_mask=attention_masks,
                min_new_tokens=args.max_new_tokens,
                max_new_tokens=args.max_new_tokens
            )
        else:
            sample = dict(
                do_sample=False,
                attention_mask=attention_masks,
                min_new_tokens=args.max_new_tokens,
                max_new_tokens=args.max_new_tokens
            )
        # By setting max_new_tokens = min_new_tokens, we control the output of samples to always be args.max_new_tokens

        ######## GENERATION ########
            
        with torch.no_grad():
            with autocast():
                torch.manual_seed(hash_key)
                output_no_wm = model.generate(input_ids, 
                                **sample)
                
                torch.manual_seed(hash_key) 
                output_w_wm = model.generate(input_ids, 
                                **sample,
                                logits_processor=LogitsProcessorList([watermark_processor]))
                
                decoded_output_no_wm = tokenizer.batch_decode(output_no_wm[:,prefix_len:], skip_special_tokens=True)
                decoded_output_w_wm = tokenizer.batch_decode(output_w_wm[:,prefix_len:], skip_special_tokens=True)                    
        
                if model_short_name == "opt": # SimCSE uses the same tokenizer as OPT
                    attention_masks = torch.ones_like(output_w_wm[:, (prefix_len-5):]) # should all be 1, since we set the new_token = 200
                    embed_wm = model_simcse(output_w_wm[:, (prefix_len-5):], attention_mask=attention_masks, output_hidden_states=True, return_dict=True).pooler_output
                    embed_no_wm = model_simcse(output_no_wm[:, (prefix_len-5):], attention_mask=attention_masks, output_hidden_states=True, return_dict=True).pooler_output
                else: # llama
                    output_wm_opt = tokenizer_opt(decoded_output_w_wm, padding=True, truncation=True, return_tensors="pt").to(device)
                    embed_wm = model_simcse(**output_wm_opt, output_hidden_states=True, return_dict=True).pooler_output
                    output_no_wm_opt = tokenizer_opt(decoded_output_no_wm, padding=True, truncation=True, return_tensors="pt").to(device)
                    embed_no_wm = model_simcse(**output_no_wm_opt, output_hidden_states=True, return_dict=True).pooler_output
        
        
        ######## DETECTION ########

        for idx in range(args.batch_size):
            with torch.no_grad():

                if args.scheme == "KGW": 
                    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        device=device,
                                        embed_matrix=embed_matrix,
                                        tokenizer_llama=None if model_short_name == 'opt' else tokenizer,
                                        tokenizer_opt=tokenizer if model_short_name == 'opt' else tokenizer_opt,
                                        gamma=args.gamma,
                                        delta=args.delta,
                                        seeding_scheme=args.seeding_scheme,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams)
                    with autocast():
                        score_dict = watermark_detector.detect(decoded_output_w_wm[idx])
                    z_score = score_dict['z_score'].item()

                elif args.scheme == "TS": 
                    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        device=device,
                                        embed_matrix=embed_matrix,
                                        tokenizer_llama=None if model_short_name == 'opt' else tokenizer,
                                        tokenizer_opt=tokenizer if model_short_name == 'opt' else tokenizer_opt,
                                        ckpt_path=args.ckpt_path,
                                        seeding_scheme=args.seeding_scheme,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams)
                    with autocast():
                        score_dict = watermark_detector.detect(decoded_output_w_wm[idx])
                    z_score = score_dict['z_score'].item()

            z_score_list.append(z_score)  
            simcse_list.append(cos(embed_wm[idx], embed_no_wm[idx]).item())

            if args.log_generated_text:
                dd = {
                    "prefix": batch['truncated_input'][idx],
                    "gold_completion": batch['baseline_completion'][idx], 
                    "no_wm_completion": decoded_output_no_wm[idx], 
                    "gen_completion": decoded_output_w_wm[idx],
                    "z_wm": z_score,
                    "simcse": cos(embed_wm[idx], embed_no_wm[idx]).item(),
                }
                with open(f"{args.output_dir}/text_{args.scheme}_{model_short_name}_1.json_pp", "a") as f:
                    f.write(json.dumps(dd) + "\n")
    

    ######## OUTPUTS ########

    results = {
        'gamma': args.gamma,
        'delta':  args.delta,
        'ckpt_path': args.ckpt_path,
        'max_new_tokens': args.max_new_tokens,
        'min_prompt_tokens': args.min_prompt_tokens,
        'max_input_len': args.max_input_len,
        'use_sampling': args.use_sampling,
        'sampling_temp': args.sampling_temp,
        'model': args.model_name_or_path,
        'z':{
            'avg': statistics.mean(z_score_list),
            'stdev': statistics.stdev(z_score_list),
            'total': len(z_score_list),
            'detected': sum([1 for val in z_score_list if val > args.detection_z_threshold]), 
        },
        'simcse': statistics.mean(simcse_list),
    }

    with open(f"{args.output_dir}/{args.scheme}_{model_short_name}_1.json", "w") as outfile:
        outfile.write(json.dumps(results, indent=4))

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")
    parser.add_argument("--config_file", type=str, default="TS.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    flat_config = {f"{group}.{key}": value for group, params in config.items() for key, value in params.items()}
    
    for key, value in flat_config.items():
        group, param = key.split('.')
        setattr(args, param, value)

    # for removing some columns to save space
    args.columns_to_remove = args.columns_to_remove.split(",") if args.columns_to_remove else []

    # if decoding scheme is not sampling, then set generation seed to None
    # to avoid confusion and calling the torch rng unnecessarily
    args.generation_seed = args.generation_seed if args.use_sampling else None

    main(args)
