# Watermarked Huggingface LM Generation Pipeline

This repository contains a script to run a watermarked Huggingface Language Model (LM) generation pipeline. The script parameters can be configured via a YAML file.

## Configuration

The YAML configuration file (`config.yaml`) defines the parameters for running the pipeline. The parameters are grouped into sections for better organization.

### General Settings

- **model_name_or_path**: Main model, path to pretrained model or model identifier from huggingface.co/models.
  - Type: `str`
  - Example: `facebook/opt-1.3b`

- **model_name_or_path**: A larger model to compute PPL.
  - Type: `str`
  - Example: `facebook/opt-2.7b`
  
- **model_simcse**: Model to compute SimCSE.
  - Type: `str`
  - Example: `princeton-nlp/sup-simcse-roberta-base`

- **load_fp16**: Whether to run the model in float16 precision.
  - Type: `bool`
  - Example: `True`

- **use_gpu**: Whether to run inference and watermark hashing/seeding/permutation on GPU.
  - Type: `bool`
  - Example: `True`

### Dataset Settings

- **dataset_name**: The name of the dataset to use (via the datasets library).
  - Type: `str`
  - Example: `c4`

- **dataset_config_name**: The configuration name of the dataset to use (via the datasets library).
  - Type: `str`
  - Example: `realnewslike`

- **dataset_split**: The split of the dataset to use (via the official datasets library).
  - Type: `str`
  - Example: `validation`

- **split**: The split defined by us.
  - Type: `str`
  - Example: `test`

- **stream_dataset**: Whether to stream the dataset from the web or download it locally.
  - Type: `bool`
  - Example: `True`

- **columns_to_remove**: Comma-separated list of columns to remove from the dataset before generation.
  - Type: `str`
  - Example: `""`

- **shuffle_dataset**: Whether to shuffle the dataset before sampling.
  - Type: `bool`
  - Example: `False`

- **shuffle_seed**: The seed to use for dataset shuffle operation.
  - Type: `int`
  - Example: `1234`

- **shuffle_buffer_size**: The buffer size to use for dataset shuffle operation.
  - Type: `int`
  - Example: `10000`

### Generation Settings

- **max_new_tokens**: The number of tokens to generate using the model, and the number of tokens removed from the real text sample.
  - Type: `int`
  - Example: `200`

- **min_prompt_tokens**: The minimum number of tokens for the prompt.
  - Type: `int`
  - Example: `300`

- **max_input_len**: The maximum length of original text.
  - Type: `int`
  - Example: `1000`

- **limit_indices**: The number of examples (first N) to pull from the dataset, if `None`, pull all, and then set this argument to the number of rows in the dataset.
  - Type: `int`
  - Example: `None`

- **input_truncation_strategy**: The strategy to use when tokenizing and truncating raw inputs to make prompts.
  - Type: `str`
  - Choices: `["no_truncation", "completion_length", "prompt_length"]`
  - Example: `completion_length`

- **input_filtering_strategy**: The strategy to use when filtering raw inputs to make prompts.
  - Type: `str`
  - Choices: `["no_filter", "completion_length", "prompt_length", "prompt_and_completion_length"]`
  - Example: `prompt_and_completion_length`

- **output_filtering_strategy**: The strategy to use when filtering/skipping rows if the model didn't generate enough tokens to facilitate analysis.
  - Type: `str`
  - Choices: `["no_filter", "max_new_tokens"]`
  - Example: `max_new_tokens`

- **use_sampling**: Whether to perform sampling during generation (non-greedy decoding).
  - Type: `bool`
  - Example: `False`

- **sampling_temp**: The temperature to use when generating using multinomial sampling.
  - Type: `float`
  - Example: `1.0`

- **top_k**: The top K to use when generating using top_k version of multinomial sampling.
  - Type: `int`
  - Example: `50`

- **top_p**: The top P to use when generating using top_p version of sampling.
  - Type: `float`
  - Example: `1.0`

- **generation_seed**: Seed for setting the torch rng prior to generation using any decoding scheme with randomness.
  - Type: `int`
  - Example: `None`

- **batch_size**: The batch size to use for generation.
  - Type: `int`
  - Example: `1`

- **seeding_scheme**: The seeding procedure to use for the watermark.
  - Type: `str`
  - Example: `simple_1`

- **store_spike_ents**: Whether to store the spike entropies while generating with watermark processor.
  - Type: `bool`
  - Example: `True`

### Evaluation Settings

- **scheme**: Watermark Scheme.
  - Type: `str`
  - Example: `TS` or `KGW`

- **ignore_repeated_bigrams**: Whether to use the detection method that only counts each unique bigram once as either a green or red hit.
  - Type: `bool`
  - Example: `False`

- **detection_z_threshold**: The test statistic threshold for the detection hypothesis test.
  - Type: `float`
  - Example: `4.0`

- **ckpt_path**: Path to checkpoint.
  - Type: `str`
  - Example: `""`

- **gamma**: The fraction of the vocabulary to partition into the greenlist at each generation and verification step.
  - Type: `float`
  - Example: `0.5`

- **delta**: The amount/bias to add to each of the greenlist token logits before each token sampling step.
  - Type: `float`
  - Example: `2.0`

- Other keywords specificed by the watermark scheme.

### Logging Settings

- **output_dir**: The unique name for the run.
  - Type: `str`
  - Example: `eval`

- **log_generated_text**: Whether to log the generated text.
  - Type: `bool`
  - Example: `True`


## Usage

1. Create a YAML configuration file (`config.yaml`) with the necessary parameters. An example configuration file is provided below.

2. Run the script with the configuration file as an argument:
   ```bash
   python your_script.py --config_file config.yaml
