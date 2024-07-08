from __future__ import annotations
import collections
from math import sqrt

import scipy.stats

import torch
from torch import nn
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from utils.normalizers import normalization_strategy_lookup

from utils.TS_networks import DeltaNetwork, GammaNetwork

class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        embed_matrix: torch.FloatTensor = torch.empty(1).cuda(),
        ckpt_path: str = '',
        gamma: float = 0.5,
        delta: float = 2.0,
    ):
        # watermarking parameters
        self.vocab = vocab
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.ckpt_path = ckpt_path
        device = embed_matrix.device
        self.vocab_size = 50272

        if len(ckpt_path) > 0:
            checkpoint = torch.load(ckpt_path)
            layer_delta = sum(1 for key in checkpoint['delta_state_dict'] if "weight" in key)  # Counting only weight keys as layers
            layer_gamma = sum(1 for key in checkpoint['gamma_state_dict'] if "weight" in key)  # Counting only weight keys as layers

            self.gamma_network = GammaNetwork(input_dim=embed_matrix.shape[1], layers=layer_gamma).to(device)
            self.delta_network = DeltaNetwork(input_dim=embed_matrix.shape[1], layers=layer_delta).to(device)

            self.delta_network.load_state_dict(checkpoint['delta_state_dict'])
            self.gamma_network.load_state_dict(checkpoint['gamma_state_dict'])
            
            for name, param in self.delta_network.named_parameters():
                param.requires_grad = False        
            for name, param in self.gamma_network.named_parameters():
                param.requires_grad = False
            self.delta_network.eval()
            self.gamma_network.eval()
            
        else:
            self.gamma = torch.tensor([gamma]).to(device)
            self.delta = torch.tensor([delta]).to(device)

        self.gamma_list = torch.empty(0, dtype=torch.float).to(device)
        self.delta_list = torch.empty(0, dtype=torch.float).to(device)
        self.embed_matrix = embed_matrix

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # Always use ids given by OPT model
        # since our gamma/delta network is trained on the embedding matrix of OPT model
       
        # seed the rng using the previous tokens/prefix according to the seeding_scheme
        self._seed_rng(input_ids)
        
        if len(self.ckpt_path) > 0:
            gamma = self.gamma_network(self.embed_matrix[input_ids[-1].item()])
            delta = self.delta_network(self.embed_matrix[input_ids[-1].item()])
        else:
            delta = self.delta
            gamma = self.gamma
        
        self.gamma_list = torch.cat([self.gamma_list, gamma])
        self.delta_list = torch.cat([self.delta_list, delta])

        greenlist_size = int(self.vocab_size * gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        
        greenlist_ids = vocab_permutation[:greenlist_size] 
        
        return greenlist_ids, gamma, delta


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, 
                tokenizer_llama: Tokenizer=None,
                tokenizer_opt: Tokenizer=None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_llama = tokenizer_llama
        self.tokenizer_opt = tokenizer_opt
        if self.tokenizer_llama is not None:
            self.vocab_size = len(self.tokenizer_llama)

    def _calc_greenlist_mask(self, logits: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(logits)
        green_tokens_mask[greenlist_token_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        device = input_ids.device
        if self.rng is None:
            self.rng = torch.Generator(device=device)

        if self.tokenizer_llama is not None: # only run this part if test on LLAMA model.
            llama_str = self.tokenizer_llama.batch_decode(input_ids[:,-5:], add_special_tokens=False)
            ids_opt = self.tokenizer_opt(llama_str, add_special_tokens=False)['input_ids']

        for b_idx in range(input_ids.shape[0]):
            if self.tokenizer_llama is not None:
                greenlist_ids, gamma, delta = self._get_greenlist_ids(torch.tensor(ids_opt[b_idx]).to(device))
            else:                
                greenlist_ids, gamma, delta = self._get_greenlist_ids(input_ids[b_idx])

            green_tokens_mask = self._calc_greenlist_mask(logits=scores[b_idx], greenlist_token_ids=greenlist_ids)
            scores[b_idx][green_tokens_mask] = scores[b_idx][green_tokens_mask] + delta.half()
        
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer_opt: Tokenizer = None,
        tokenizer_llama: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer_opt, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer_opt = tokenizer_opt
        self.tokenizer_llama = tokenizer_llama
        
        if self.tokenizer_llama is not None:
            self.vocab_size = len(self.tokenizer_llama)

        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        var = torch.sum(self.gamma_list * (1 - self.gamma_list))
        mean = torch.sum(self.gamma_list)
        z = (observed_count - mean)/torch.sqrt(var)            
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z.cpu())
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        if self.ignore_repeated_bigrams:
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                greenlist_ids, gamma, delta = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError(
                    (
                        f"Must have at least {1} token to score after "
                        f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                    )
                )
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                # greenlist_ids, gamma, delta = self._get_greenlist_ids(input_ids[:idx])
                if self.tokenizer_llama is None:
                    greenlist_ids, gamma, delta = self._get_greenlist_ids(input_ids[:idx])
                else:
                    llama_str = self.tokenizer_llama.decode(input_ids[max(idx-5, 0):idx], add_special_tokens=False)
                    ids_opt = self.tokenizer_opt(llama_str, add_special_tokens=False)['input_ids']
                    if len(ids_opt) == 0:
                        continue
                    greenlist_ids, gamma, delta = self._get_greenlist_ids(torch.tensor(ids_opt).to(self.device))
                
                # print(input_ids, curr_token, greenlist_ids)
                # exit()
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            if self.tokenizer_llama is not None:
                tokenized_text = self.tokenizer_llama(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
                if tokenized_text[0] == self.tokenizer_llama.bos_token_id:
                    tokenized_text = tokenized_text[1:]
            else:
                assert self.tokenizer_opt is not None, (
                    "Watermark detection on raw string ",
                    "requires an instance of the tokenizer_opt ",
                    "that was used at generation time.",
                )
                tokenized_text = self.tokenizer_opt(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
                if tokenized_text[0] == self.tokenizer_opt.bos_token_id:
                    tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer_llama is not None) and (tokenized_text[0] == self.tokenizer_llama.bos_token_id):
                tokenized_text = tokenized_text[1:]
            elif (self.tokenizer_opt is not None) and (tokenized_text[0] == self.tokenizer_opt.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict
