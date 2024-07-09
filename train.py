import os
import argparse
# os.environ['HF_HOME'] = "/data1/mingjia/cache/huggingface"
# print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

from functools import partial
from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from transformers import AutoModel, OPTForCausalLM, AutoTokenizer
import itertools
from torch.cuda.amp import autocast, GradScaler
import statistics
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.data_loader import TrainValTestIterableDataset, CustomIterableDataset

# better bool flag type for argparse
from utils.submitit import str2bool

# some file i/o helpers
from utils.TS_networks import GammaNetwork, DeltaNetwork

# generation pipeline helpers
from utils.generation import (
    load_hf_dataset,
    check_input_lengths,
    check_output_lengths,
    tokenize_for_generation
)

# Reproducible
hash_key = 15485863
torch.manual_seed(hash_key)
torch.cuda.manual_seed(hash_key)

def get_green_mask(scores, gamma, tau):
    # Use Gumbel Softmax to Generate Green/Red Tokens
     
    device = scores.device
    green_tokens_mask = torch.zeros_like(scores, dtype=torch.float)
    red_tokens_mask = torch.zeros_like(scores, dtype=torch.float)
   
    logits_bernoulli = torch.zeros(scores.shape[0], scores.shape[1], 2)
    
    logits_bernoulli[:,:,0] = (1-gamma).expand_as(logits_bernoulli[:,:,0])
    logits_bernoulli[:,:,1] = gamma.expand_as(logits_bernoulli[:,:,1])

    logits_bernoulli = torch.log(logits_bernoulli) # [batch_size, vocab_size, 2]
    red_green_prob = F.gumbel_softmax(logits_bernoulli, tau=tau, hard=False, dim=-1).to(device)  # [batch_size, vocab_size, 2]

    for b_id in range(scores.shape[0]):
        green_tokens_mask[b_id] = red_green_prob[b_id,:,1].squeeze(0) # [vocab_size]
        red_tokens_mask[b_id] = red_green_prob[b_id,:,0].squeeze(0)

    return green_tokens_mask, red_tokens_mask # [batch_size, vocab_size]

def differentiable_decode(model, model_simcse, delta_network, gamma_network, input_ids, attention_masks, max_length, tau=0.1):
    device = input_ids.device
    batch_size = input_ids.shape[0]
    delta_list = torch.empty((batch_size, 0), dtype=torch.float).to(device)
    gamma_list = torch.empty((batch_size, 0), dtype=torch.float).to(device)
    p_green_list = torch.empty((batch_size, 0), dtype=torch.float).to(device)

    embedding_matrix = model.get_input_embeddings().weight
    embedding_matrix_simcse = model_simcse.get_input_embeddings().weight
    vocab_simcse = embedding_matrix_simcse.shape[0]

    input_embeddings_wm = model.get_input_embeddings()(input_ids).to(device)
    input_embeddings_wm_simcse = model_simcse.get_input_embeddings()(input_ids).to(device)
    input_embeddings_no_wm = model.get_input_embeddings()(input_ids).to(device)
    input_embeddings_no_wm_simcse = model_simcse.get_input_embeddings()(input_ids).to(device)

    output_ids_no_wm = torch.empty((batch_size, 0), dtype=torch.int).to(device)
    output_ids_no_wm = torch.cat([input_ids, output_ids_no_wm], dim=1)

    output_ids_wm = torch.empty((batch_size, 0), dtype=torch.int).to(device)
    output_ids_wm = torch.cat([input_ids, output_ids_wm], dim=1)

    for _ in range(max_length):
        ########### Generate wm embeddings ###########
        with torch.no_grad():
            logits = model(inputs_embeds=input_embeddings_wm, attention_mask=attention_masks).logits
            
        # Get llm logits 
        last_logits = logits[:, -1, :].squeeze(dim=1)

        # Get delta/gamma based on embedding of preceding token
        delta = delta_network(input_embeddings_wm[:, -1, :].squeeze(dim=1)) 
        gamma = gamma_network(input_embeddings_wm[:, -1, :].squeeze(dim=1)) 

        # Get green tokens using Gumbel Softmax
        green_mask, red_mask = get_green_mask(last_logits, gamma, tau) # [batch_size, vocab_size]
        last_logits = last_logits + delta * green_mask # [batch_size, vocab_size] both for last_logits and (delta * green_mask)

        # compute soft_one_hot (contains gradient) and hard_one_hot (no gradient), and get mix_ont_hot 
        soft_one_hot = F.softmax(last_logits, dim=-1).to(embedding_matrix.dtype)
        max_val, max_index = torch.max(soft_one_hot, dim=-1, keepdim=True)

        hard_one_hot = torch.zeros_like(soft_one_hot)
        hard_one_hot[soft_one_hot == max_val] = 1 #[batch_size]
        mix_one_hot = hard_one_hot * 0.5 + soft_one_hot * 0.5
        output_ids_wm = torch.cat([output_ids_wm, max_index], dim=1)
        
        # Get next token embedding
        predicted_embedding_wm = torch.matmul(mix_one_hot, embedding_matrix)
        input_embeddings_wm = torch.cat([input_embeddings_wm, predicted_embedding_wm.unsqueeze(1)], dim=1)

        predicted_embedding_simcse = torch.matmul(mix_one_hot[:,:vocab_simcse], embedding_matrix_simcse)
        input_embeddings_wm_simcse = torch.cat([input_embeddings_wm_simcse, predicted_embedding_simcse.unsqueeze(1)], dim=1)

        # Probability of green token to compute detection loss
        p_green = torch.sum(mix_one_hot * green_mask, dim=-1, keepdim=True)
   
        delta_list = torch.cat([delta_list, delta.expand(batch_size, 1)], dim=-1)
        gamma_list = torch.cat([gamma_list, gamma.expand(batch_size, 1)], dim=-1)

        p_green_list = torch.cat([p_green_list, p_green], dim=-1)

        ########### Generate no wm embeddings ###########
        with torch.no_grad():
            logits_no_wm = model(inputs_embeds=input_embeddings_no_wm, attention_mask=attention_masks).logits

        last_logits_no_wm = logits_no_wm[:, -1, :].squeeze(dim=1)

        # Get soft_one_hot_no_wm and hard_one_hot_no_wm, and compute mix_ont_hot_no_wm. 
        # None of these contain gradient. 
        # This is just to make sure the generating no_wm embeddings follows the same procedure as the above wm embeddings.

        soft_one_hot_no_wm = F.softmax(last_logits_no_wm, dim=-1).to(embedding_matrix.dtype)

        max_val, max_index = torch.max(soft_one_hot_no_wm, dim=-1, keepdim=True)

        hard_one_hot_no_wm = torch.zeros_like(soft_one_hot_no_wm)
        hard_one_hot_no_wm[soft_one_hot_no_wm == max_val] = 1 

        mix_one_hot_no_wm = hard_one_hot_no_wm * 0.5 + soft_one_hot_no_wm * 0.5

        # Embedding of the next token
        predicted_embedding_no_wm = torch.matmul(mix_one_hot_no_wm, embedding_matrix)
        input_embeddings_no_wm = torch.cat([input_embeddings_no_wm, predicted_embedding_no_wm.unsqueeze(1)], dim=1)
        
        predicted_embedding_no_wm_simcse = torch.matmul(mix_one_hot_no_wm[:,:vocab_simcse], embedding_matrix_simcse)
        input_embeddings_no_wm_simcse = torch.cat([input_embeddings_no_wm_simcse, predicted_embedding_no_wm_simcse.unsqueeze(1)], dim=1) 
        
        _, max_index_no_wm = torch.max(logits_no_wm[:, -1, :].squeeze(dim=1), dim=-1, keepdim=True)
        output_ids_no_wm = torch.cat([output_ids_no_wm, max_index_no_wm], dim=1)

        attention_masks = torch.cat((attention_masks, torch.ones(batch_size, 1).cuda()), dim=1)

    return input_embeddings_wm_simcse, output_ids_wm, input_embeddings_no_wm_simcse, output_ids_no_wm, delta_list, gamma_list, p_green_list


def main(args):
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
    scaler = GradScaler()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto").cuda()
    model.eval()
    hidden_size = model.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")
    
    model_simcse = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base", torch_dtype=torch.float16).cuda()
    model_simcse.eval()

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model_simcse.named_parameters():
        param.requires_grad = False
    
    delta_network = DeltaNetwork(input_dim=hidden_size, layers=args.layer_delta, init_val=args.init_val_delta).cuda()
    gamma_network = GammaNetwork(input_dim=hidden_size, layers=args.layer_gamma, init_val=args.init_val_gamma).cuda()
    
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(list(gamma_network.parameters()) + list(delta_network.parameters()), lr=args.lr, weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.Adam(list(gamma_network.parameters()) + list(delta_network.parameters()), lr=args.lr, weight_decay=args.wdecay)
        
    scheduler_gamma = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.lr_step_size, gamma = args.lr_scheduler)
   
    ###########################################################################
    # Configure the prompt construction partial
    ###########################################################################

    # Construct the data filtering/sampling scheme partials

    token_kwargs = dict(
        hf_model_name="facebook/opt-1.3b",
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
    input_check_kwargs.update(dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=args.max_new_tokens)) #
    input_check = partial(check_input_lengths, **input_check_kwargs)

    output_kwargs = dict(min_output_len=args.max_new_tokens)
    output_check = partial(check_output_lengths, **output_kwargs)

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

    dataset = load_hf_dataset(args)

    # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
    dataset_w_prompts = dataset.map(tokenize_prompts, batched=False)

    # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
    dataset_input_len_filtered = dataset_w_prompts.filter(input_check, batched=False)

    # Create instances for train, validation, and test sets
    train_dataset = TrainValTestIterableDataset(dataset_input_len_filtered, split='train', seed=0)
    train_dataset = CustomIterableDataset(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    


    ###########################################################################
    # Main loop - actually executes the generation pipeline.
    # and accumulates the result rows in a list, assumes list is "small"-ish
    # and we aren't accumulating any tensors or other memory hogging artifacts
    ###########################################################################

    pbar = tqdm(total=args.min_generations)

    factor_list = []
    
    # Record all the L_S and L_D during training for analysis
    L_S_list, L_D_list = [], [] # empty after 1 epoch
    L_S_list_100, L_D_list_100 = [], [] # empty after 100 steps
    step_cnter = 0

    for epoch in range(args.epochs):

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(model.device)
            attention_masks = (input_ids != tokenizer.pad_token_id).long()
            
            # full_ids = prompt['untruncated_inputs'].to(device) 
            prefix_len = input_ids.shape[1]

            with autocast():
                # Get embedding of wm_gen (with gradient) and no_wm_gen (no gradient)
                ret = differentiable_decode(model, model_simcse, delta_network, gamma_network, input_ids, attention_masks, args.max_new_tokens, tau=args.tau)
                input_embeddings_simcse, output_ids_wm, input_embeddings_no_wm, output_ids_no_wm, delta_list, gamma_list, p_green_list = ret
                
                # simcse embedding
                attention_masks = torch.ones_like(output_ids_wm[:, (prefix_len-5):])
                embed_wm = model_simcse(inputs_embeds=input_embeddings_simcse[:, (prefix_len-5):], attention_mask=attention_masks, output_hidden_states=True, return_dict=True).pooler_output
                embed_no_wm = model_simcse(inputs_embeds=input_embeddings_no_wm[:, (prefix_len-5):], attention_mask=attention_masks, output_hidden_states=True, return_dict=True).pooler_output
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

                # Get L_D (detection loss)
                var = torch.sum(gamma_list * (1 - gamma_list), dim = -1) # [batch_size]
                true_z_score = torch.sum(p_green_list - gamma_list, dim = -1)/torch.sqrt(var) # [batch_size]
                true_z_score = torch.mean(true_z_score)
                L_D_list_100.append(true_z_score.item())
                detect_loss = (torch.log10(torch.mean(true_z_score)) if args.log_z_score else torch.mean(true_z_score))
                detect_loss *= (- args.z_score_factor)

            if args.method == "Pareto": 
                # Get L_S (semantic loss)
                simcse_loss = - torch.mean(cos(embed_wm, embed_no_wm))
                L_S_list_100.append(-simcse_loss.item())

                scaler.scale(detect_loss).backward(retain_graph=True)
                vec_d = []
                for param in itertools.chain(gamma_network.parameters(), delta_network.parameters()):
                    vec_d.append(param.grad.view(-1))  
                vec_d = torch.cat(vec_d)
                
                optimizer.zero_grad()
                scaler.scale(simcse_loss).backward(retain_graph=True)
                vec_s = []
                for param in itertools.chain(gamma_network.parameters(), delta_network.parameters()):
                    vec_s.append(param.grad.view(-1))  
                vec_s = torch.cat(vec_s)

                # Multiple-Gradient Descent Algorithm
                if torch.dot(vec_d, vec_s) >= torch.dot(vec_d, vec_d):
                    factor = 1.0
                elif torch.dot(vec_d, vec_s) >= torch.dot(vec_s, vec_s):
                    factor = 0.0
                else:
                    factor = (torch.dot(vec_s - vec_d, vec_s)/torch.dot(vec_s - vec_d, vec_s - vec_d)).item()
                
                factor = min(factor, 0.01) # ensure the weight for L_D is not too high

                factor_list.append(factor)
            
                vec = factor * vec_d + (1 - factor) * vec_s

                # Assign the gradients from MGDA
                grad_position = 0
                for param in itertools.chain(gamma_network.parameters(), delta_network.parameters()):
                    param_numel = param.numel()
                    param_grad = vec[grad_position:grad_position + param_numel]
                    param_grad = param_grad.view_as(param)
                    param.grad = param_grad
                    grad_position += param_numel

                # Step
                scaler.step(optimizer)
                scaler.update()
                scheduler_gamma.step()
                
            elif args.method == "Weighted": # weighted sum
                # Get L_S (semantic loss)
                simcse_loss = - torch.mean(cos(embed_wm, embed_no_wm))
                L_S_list_100.append(-simcse_loss.item())
                loss = detect_loss + simcse_loss

                # Step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler_gamma.step()
            else:
                print("optimization method doesn't exist")
                return 
            
            if args.wandb:
                log_dict = {
                    "semantic loss": simcse_loss,
                    "detect_loss": detect_loss,
                    "true_z_score": true_z_score,
                    "gamma_average": torch.mean(gamma_list),
                    "delta_average": torch.mean(delta_list)
                }
                if args.method == "Pareto":
                    log_dict["factor"] = factor
                run.log(log_dict, step=step_cnter) 
            step_cnter += 1
            
            if step_cnter % 100 == 0: 
                
                # check the value range of delta/gamma by printing an example
                print("epoch", epoch)
                print("step", step_cnter)
                print("delta_list")
                print(delta_list[0]) 
                print("gamma_list")
                print(gamma_list[0]) 

                print("simcse", statistics.mean(L_S_list_100))
                print("z-score", statistics.mean(L_D_list_100))

                L_S_list_100, L_D_list_100 = [], []
                L_S_list.append(statistics.mean(L_S_list_100))
                L_D_list.append(statistics.mean(L_D_list_100))

                checkpoint_path = f'{args.ckpt_dir}/checkpoint_len_{args.max_new_tokens}_{step_cnter}.pth'

                checkpoint = {
                    'delta_state_dict': delta_network.state_dict(),
                    'gamma_state_dict': gamma_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
    
        print("Summrize for epoch ", epoch)
        print("simcse", statistics.mean(L_S_list))
        print("z_score", statistics.mean(L_D_list))
        L_S_list, L_D_list = [], []
    pbar.close()

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
        default="train",
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
        default=200,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=200,  # 500
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
        "--max_input_len",
        type=int,
        default=500,
        help="The max length of full sentence.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs",
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
        default="completion_length",
        choices=["no_filter", "completion_length", "prompt_length", "prompt_and_completion_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--output_filtering_strategy",
        type=str,
        default="no_filter",
        choices=["no_filter", "max_new_tokens"],
        help=(
            f"The strategy to use when filtering/skipping rows if the model didn't ",
            f"generate enough tokens to facilitate analysis.",
        ),
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
        default=0,
        help="The top k to use when generating using top_k version of multinom sampling. 0 for whole vocab_size.",
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
        "--ckpt_dir",
        type=str,
        default="ckpt",
        help="ckpt directory.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Allow overwriting of old generation files at the same output location.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for gamma-generator.",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=50,
        help="Learning rate schedule step size.",
    )
    parser.add_argument(
        "--layer_gamma",
        type=int,
        default=2,
        help="number of layers for gamma network (MLP).",
    )
    parser.add_argument(
        "--layer_delta",
        type=int,
        default=2,
        help="number of layers for delta network (MLP).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=float,
        default=1.0,
        help="Learning rate schedule gamma (decay factor).",
    )
    parser.add_argument(
        "--init_val_gamma",
        type=float,
        default=0.25,
        help="init gamma.",
    )
    parser.add_argument(
        "--init_val_delta",
        type=float,
        default=2.0,
        help="init delta.",
    )
    parser.add_argument(
        "--wdecay",
        type=float,
        default=1e-5,
        help="Weight decay.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Temperature for Gumbel Softmax.",
    )
    parser.add_argument(
        "--accum_iter",
        type=int,
        default=1,
        help="Gradient Accumulation",
    )
    parser.add_argument(
        "--log_z_score",
        type=str2bool,
        default=False,
        help="computer log z score",
    )
    parser.add_argument(
        "--z_score_factor",
        type=float,
        default=1.0,
        help="factor on detection loss compared with semantic loss",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="SGD or Adam",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="Pareto",
        help="The optimization method:Pareto, Weighted",
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
