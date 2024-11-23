# Datasets
We move from historical document OCR to modern document OCR through SynthDoG and Tesseract processing of different corpuses. 

# Dataset Groups
We grouped these datasets into a few "Dataset groups" which in the code just combine the train-test splits of each dataset. The splits are preprocessed by `python pipeline.py --config_file <...> --prepare_dataset`, so we are able to group these datasets in whatever combination desired without corrupting the original train-test splits. Note that the size of each dataset does modify how much the fine-tuned LM will focus on examples from that dataset! This could mess us up since there's so much non-English data in the groups described further down. As seen in `yaml_builder.py`, here are the original dataset groups we defined for our Original Testing Pipeline:
```python
datasets_groups = {"base": ['bln600'], # base dataset style
  "expanded": ['europarl', 'plainwiki', 'plusone', 'iam_tesseract', 'bln600'], # expanded context with 80k-100k entries
  "expanded-10k": ['europarl_10k', 'plainwiki_10k', 'plusone_10k', 'iam_tesseract', 'bln600'], # expanded context but larger datasets reduced by 1/10th size
}
```

# Original Testing Pipeline
Our pipeline allows for many language models and unique model versions to be run with different dataset groups. Running all permutations of these combos results in 243 tests to be queued on the HPC, which we knew would be unreasonable to complete and analyze in time to find the "optimal" combination of LM+LM version+dataset group+prompt (for non-BART models). We knew, therefore, that we had to test these independent variables in a smarter way.

Orginally, our testing pipeline was as follows:

1. Baseline test -- This test served to eliminate lackluster models. It would test all model versions on prompts 1 and 2, with the expanded-10k dataset group. This dataset group was chosen because it was the cleanest amount of data we have, so we expected models would perform the best. This gives models a fighting chance on lackluster prompts. 

The `yaml_builder` code for it was as follows:
```python
keys_to_print = ['expanded-10k'] # this file will make all yaml files, but only print the keys you ask for.
config_path = "model_configs.yaml"
prompt_patterns = range(1, 8)
prompts_to_print = [1, 2] # just like keys_to_print
for key in datasets_groups:
    print(f"sbatch phi3_model_pipeline.slurm --config_file models/bart-base_{key}_prompt_1 --prepare_dataset") # prepare dataset once


model_type = "Phi_3"
model_versions = ['phi-3-mini-4k', 'phi-3-mini-128k']
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups, keys_to_print, prompts_to_print)

model_type = "Llama_2"
# model_versions = ['llama-2-7b', 'llama-2-13b', 'llama-2-70b']
model_versions = ['llama-2-7b', 'llama-2-13b']
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups, keys_to_print, prompts_to_print)

model_type = "BART"
model_versions = ['bart-base', 'bart-large']
format_patterns(model_type, config_path, model_versions, [1], datasets_groups, keys_to_print, prompts_to_print)

model_type = "Llama_3_1"
# model_versions = ['llama-3.1-7b', 'llama-3.1-13b', 'llama-3.1-70b', 'llama-3.1-405b-instruct']
model_versions = ['llama-3.1-7b', 'llama-3.1-13b']
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups, keys_to_print, prompts_to_print)
```

This made for (1 dataset * (6 models * 2 prompts) + 2 bart models) = 14 tests to run. 

2. Refinement -- This test served to eliminate lackluster "extra" prompts. The best-performing version of each LM would be taken and prompts 3-7 would be used for finetuning step. Results would be compared by prompts. 

This made for (1 dataset * (3 models * 5 prompts each)) = 15 tests to run.

3. Sanity Check -- This test served to verify that the expanded-10k dataset group was good. All 3 dataset groups would be tested to see if they improved upon the original base model. We would use all models from Stage 2 and only the best 2 prompts from Stage 2. 

This made for (3 dataset * (3 models * 2 prompts each + 1 Bart run)) = 21 tests to run.

4. Confirmation -- This would take our best-trained model with best prompt and replace the dataset group with the `base` one, to see how our non-dataset changes improve upon the original Historical Documents BLN600 paper. 

Even though this was far better than the original 243 tests, we ran into a major issue while queueing our first Stage: our team had run out of compute time on Newton! Here is an example:
```bash
(base) [cap6614.student2@evuser1 llms_post-ocr_correction-main]$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            399265    normal   LM_OCR cap6614. PD       0:00      1 (QOSGrpCPUMinutesLimit)
            399264    normal   LM_OCR cap6614. PD       0:00      1 (QOSGrpCPUMinutesLimit)
            399263    normal   LM_OCR cap6614. PD       0:00      1 (QOSGrpCPUMinutesLimit)
            399262    normal   LM_OCR cap6614. PD       0:00      1 (QOSGrpCPUMinutesLimit)
            399261    normal   LM_OCR cap6614. PD       0:00      1 (QOSGrpCPUMinutesLimit)
```
As you can see, we were totally out of CPU time for the month through our many submissions of test runs for the pipline, dataset preparations, etc. This wasn't helped by the fact that a few of our test runs had stalled for many many hours, eating up those precious CPU minutes (every submission had a 72-hour maximum window attached, so stalled code would hang for 3 days before being killed by SLURM).

We reached out to Dr. Bedi to resolve this, but knew at the same time that we would need to minify how many tests we did. So now we needed a solution. 

# New Testing Pipeline 
Because resources were limited, we wanted to compare against only a few model versions, unlike our historical Stage 1. We will split our new test into two categories, BART and Other. 

## BART
Because BART is a Sequence-to-Sequence model, there is no prompt engineering to do. 
The former paper used `bart-large`, and `bart-base` on BLN600 dataset, which includes old English and some old German texts. We aim to measure how modern OCR datasets can improve foreign language and English OCR. BART was published in 2019 and is trained primarily on English data. Non-English data involves the plugin of multiple transformer layers to "translate" foreign language tokens to noisy English tokens, which BART denoises and works with. The foreign language mapping is learned as a geometric transformation between language spaces. [Source](https://arxiv.org/pdf/1910.13461). With this in mind, we compromised to only use English datasets for our BART dataset improvement. Since BLN600 is text extracted from printed newspapers, we also opted to abstain from use of `iam_tesseract` to reduce potential noise differences between handwriting and printed text OCR errors. To keep training time low, we used the _10k versions of our expanded datasets, so BART will only use new:
```python
"bart-compromise": ['plainwiki_10k', 'bln600']
```
CER-reduction results between our `bart-large` and `bart-base` runs will be compared to the original paper's results. We will separate this between our new BART's performance on PlainWiki data and BLN600 data. This reduces us to two tests for BART:
```
| Dataset Group   | Model Version | Prompt |
|-----------------|---------------|--------|
| bart-compromise | bart-large    | N/A    |
| bart-compromise | bart-base     | N/A    |
```

## Others
### LLaMA 2
We originally had 7 prompts, including the original prompt, but we will reduce down to two, including the original prompt. The 6 engineered prompts we wrote will be given to a state-of-the-art LLM such as GPT-4o to summarize/combine the prompts for an optimized approach. THis is a known approach to prompt engineering, as seen [here](https://cameronrwolfe.substack.com/p/automatic-prompt-optimization). We will give this to [BPO](https://huggingface.co/spaces/CCCCCC/BPO_demo/tree/main) to further optimize. 

The preceding paper group used `LLaMA2-7b`, `LLaMA2-13b` on the BLN600 dataset. We believe these models can be improved drastically with a larger dataset. Because LLaMA is familiar with foreign languages, including Spanish and Russian, we include our datasets attached to these languages. This gives us:
```python
"llama2-compromise": ['europarl_10k', 'plainwiki_10k', 'plusone_10k', 'bln600']
```
```
| Dataset Group    | Model Version | Prompt |
|------------------|---------------|--------|
| llama2-compromise| llama-2-7b    | 1      |
| llama2-compromise| llama-2-13b   | 1      |
| llama2-compromise| llama-2-7b    | 2      |
| llama2-compromise| llama-2-13b   | 2      |
```
The first two rows allow us to confirm that our dataset group performs better than the preceding work. The second two rows confirm that our prompt does perform better than theirs.

### LLaMA 3.1
We also believe that CER-reduction will be improved significantly between LLaMA2 and LLaMA3.1, so we will test on these. 
We use our `llama2-compromise` dataset group as the base of our work here, but we also include `iam-tesseract` as we hypothesize LLaMA3.1 is significant enough in "knowledge" to pick up on the differences between handwriting OCR error patterns and printed text OCR error patterns. This brings us back to our original `expanded-10k` dataset group. 

We have `llama-3.1-7b` and `llama-3.1-13b` as well as two prompts, meaning four test permutations, which we want to reduce. 

We will compare the preceding paper's CER-reduction to LLaMA3.1's reduction using the same hyperparameters, BLN600 dataset, and prompt that their LLaMA2 used. 
```
| Dataset Group | Model Version | Prompt |
|---------------|---------------|--------|
| base          | llama-3.1-7b  | 1      |
| base          | llama-3.1-13b | 1      |
```

We can use the LLaMA2 tests to affirm that our dataset group is better than the base. As a corollary to this, we can switch to testing our other prompt without issue. Due to resource restrictions, we can not test both model versions, so we will only run a test using the 13B model since in the preceding paper, `llama-2-13b` outperformed `llama-2-13b` by 11.25% CER-reduction (see Table 3 of the paper). 
```
| Dataset Group | Model Version | Prompt |
|---------------|---------------|--------|
| expanded_10k  | llama-3.1-13b | 2      |
```

### Phi-3
For the same reason we only used `expanded_10k` on prompt 2 for `llama-3.1-13b`, we will do the same for both `phi-3-mini-4k` and `phi-3-mini-128k`. These models are very small, so running multiple tests is less resource-intensive combined than even one LLaMA 2 training. 
```
| Dataset Group | Model Version   | Prompt |
|---------------|-----------------|--------|
| expanded_10k  | phi-3-mini-4k   | 2      |
| expanded_10k  | phi-3-mini-128k | 2      |
```
These results will be compared to the LLaMA 2 and 3.1 results. 

This reduces from 50 tests in the original plan from last week's meeting to 11 tests, very minimal. 