# The code of RICO
## Usage Instructions

### 1. Run a Single Experiment

Using a config file:

```bash
python experiment_runner.py --config configs/lora_experiment.yaml --mode eval
```

Using command-line arguments:

```bash
python experiment_runner.py \
    --base-model-path "/mnt/data1/TC/TextClassDemo/llama3.1-8b" \
    --adapter-path "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/llama3.1-8b_ohsumed_lora_english_zeroshotCoT" \
    --data-path "/mnt/data1/TC/TextClassDemo/data/ohsumed_Test_alpaca_noCoT_updated.json" \
    --dataset-name "ohsumed" \
    --use-lora \
    --mode eval
```

### 2. Create a New Prompt Template

```bash
python run_experiments.py --create-prompt my_dataset --prompt-content "Your prompt template here"
```

### 3. Create a New Experiment Configuration

```bash
python run_experiments.py --create my_experiment \
    --base-model-path "/path/to/model" \
    --adapter-path "/path/to/adapter" \
    --data-path "/path/to/data" \
    --dataset-name "my_dataset" \
    --use-lora
```

### 4. Run All Experiments in Batch

```bash
python run_experiments.py --all --mode eval
```

---

## Configuration Parameters

### Model Configuration

* `base_model_path`: Path to the base model
* `adapter_path`: Path to the LoRA adapter
* `use_lora`: Whether to use LoRA

### Data Configuration

* `data_path`: Path to the data file
* `dataset_name`: Dataset name (used to select the prompt template, matching `configs/prompts/{dataset_name}_prompt.txt`)

### Generation Parameters

* `max_new_tokens`: Maximum number of generated tokens
* `temperature`: Temperature for sampling
* `top_p`: Top-p sampling value
* `do_sample`: Whether to enable sampling (should be enabled for majority voting, otherwise greedy decoding will be used)

### Training Configuration

* `batch_size`: Batch size
* `vote_count`: Number of votes per sample (run multiple inferences per sample and aggregate via majority vote; set to 1 to disable voting)

### Output Configuration

* `base_output_dir`: Output directory
* `experiment_name`: Name of the experiment

---

## Prompt Templates

### Creating a Prompt Template

Create a file named `{dataset_name}_prompt.txt` under the `configs/prompts/` directory. Format:

```
You are a text classification expert. Your task is to classify the given text into one of the specified categories.
Category mapping:
{category_mapping}
Text: {text}
Please analyze the text and provide your classification step by step:
```

### Template Variables

* `{text}`: The input text
* `{category_mapping}`: (Optional) The mapping of categories

---

## Output Files

Each experiment will generate outputs under the directory:

```
outputs/{experiment_name}_{adapter_name}_{dataset_name}/
```

The directory includes:

* `eval_results.json`: Evaluation results
* `confusion_matrix.png`: Confusion matrix plot
* `error_analysis.json`: Misclassified sample analysis
* `experiment.log`: Experiment log

---
