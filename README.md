Code for **Token-Specific Watermarking with Enhanced Detectability and Semantic Coherence for Large Language Models**

## Environment Setup
* Our code follows the setup of [KGW](https://github.com/jwkirchenbauer/lm-watermarking/tree/main/). 
* Make sure the packages in `requirements.txt` are installed in your environment.
* Also, in the training and inference files, change the model paths (`facebook/opt-1.3b`, `princeton-nlp/sup-simcse-roberta-base`, `meta-llama/Llama-2-7b-hf`) to your desired paths.

## Train 
  * `bash run_pipeline.sh`
  * Choose from MOO or Weighted Sum to train the network
    * If MOO, `log_z_score=False, z_score_factor=1.0`
    * If Weighted Sum, `log_z_score=False, z_score_factor=4e-4`

## Evaluation
### Default setting for all evaluation
  * Multinomial sampling with temp=1.0
  * Dataset: C4 realnewslike official validation split from Hugging Face, and we loaded it and split it into our validation and test set. The default split is the test split.
  * 500 samples
  * generation length = 200 tokens

### OPT-1.3b evaluation
* Results stored in `eval/opt` by default
* KGW
  * `CUDA_VISIBLE_DEVICES=0 python inference_bs.py --split=test --batch_size=20`
* Ours
  * `CUDA_VISIBLE_DEVICES=0 python inference.py --split=test --batch_size=20`
* SWEET
  * `CUDA_VISIBLE_DEVICES=0 python inference_sweet.py --split=test --batch_size=20`

### Ablation Study: Weighted Sum
* Ours
  * `CUDA_VISIBLE_DEVICES=0 python inference_weighted.py --split=test --batch_size=20`

### Dipper Attack
* KGW
  * `CUDA_VISIBLE_DEVICES=0 python inference_bs_dipper_get_text.py --split=test --batch_size=20`
  Get baseline generation text
  * Then use [Dipper](https://github.com/martiansideofthemoon/ai-detection-paraphrases0) to get paraphrase text
  * `CUDA_VISIBLE_DEVICES=0 python inference_bs_dipper_text_eval.py --split=test --batch_size=20`
  Evaluation on the paraphase text
* Ours
  * `CUDA_VISIBLE_DEVICES=0 python inference_dipper_get_text.py --split=test --batch_size=20`
  Get baseline generation text
  * Then use [Dipper](https://github.com/martiansideofthemoon/ai-detection-paraphrases0) to get paraphrase text
  * `CUDA_VISIBLE_DEVICES=0 python inference_dipper_text_eval.py --split=test --batch_size=20`
  Evaluation on the paraphase text

### Copy_Paste Attack
* change `num_cp_split=1` or `num_cp_split=3` for two settings
* KGW
  * `CUDA_VISIBLE_DEVICES=0 python inference_bs_cp_att.py --split=test --batch_size=20 --num_cp_split=1`
* Ours
  * `CUDA_VISIBLE_DEVICES=0 python inference_cp_att.py --split=test --batch_size=20 --num_cp_split=1`

### LLAMA-2 evaluation
* Results stored in `eval/llama` by default
* KGW
  * `CUDA_VISIBLE_DEVICES=0 python inference_bs_llama.py --split=test --batch_size=20`
* Ours
  * `CUDA_VISIBLE_DEVICES=0 python inference_llama.py --split=test --batch_size=20`

### Plotting the figures
* Figure 2, 3, 7, 8, 9: [result_figures.ipynb](result_figures.ipynb)
* Figure 4: [pos_tag/delta_gamma_analysis.ipynb](pos_tag/delta_gamma_analysis.ipynb)
* Figure 5: [dipper/result.ipynb](dipper/result.ipynb)
* Figure 6: [cp_att/result.ipynb](cp_att/result.ipynb)


