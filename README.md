# Distilling Step-by-Step!

Code for paper [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301)

## Environment Setup
- Setup Conda environment:
```
conda create --name distill python=3.10.6 -y
conda activate distill
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/huggingface/transformers@v4.24.0 datasets sentencepiece protobuf==3.20.* tensorboardX
```
- Extract datasets to `datasets/`:
```
unzip datasets.zip
```

## Command Usages
#### Args usages
- `--from_pretrained`: `google/t5-v1_1-small`, `google/t5-v1_1-base`, `google/t5-v1_1-large`, `google/t5-v1_1-xxl`
- `--dataset`: `esnli`, `anli1`, `cqa`, `svamp`
- `--label_type`:
  - `--label_type gt`: Use GT label for training
  - `--label_type llm`: Use LLM predicted label for training
- `--alpha`: Task weight for multi-task training. Loss = alpha * label_prediction_loss + (1 - alpha) * rationale_generation_loss
  - `--alpha 0.5`: recommended
- `--batch_size`: Batch size
- `--grad_steps`: Gradient accumulation step
- `--max_input_length`: Maximum input length
- `--eval_steps`: How many steps to evaluate the model during training
- `--max_steps`: Maximum steps for training
- `--run`: Random seed to use
- `--model_type`:
  - `standard`: Standard finetuning (`--label_type gt`) or distillation (`--label_type llm`)
  - `task_prefix`: Distilling step-by-step
- `--parallelize`: Model parallelism


#### Example usages
- Standard finetuning:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type standard --label_type gt --batch_size 64
```


- Distilling step-by-step with `GT label` and `PaLM rationale`:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 64
```


- Standard distillation:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type standard --label_type llm --batch_size 64
```


- Distilling step-by-step with `PaLM label` and `PaLM rationale`:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type task_prefix --label_type llm --llm palm --alpha 0.5 --batch_size 64
```

## Cite
If you find this repository useful, please consider citing:
```bibtex
@article{hsieh2023distilling,
  title={Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes},
  author={Hsieh, Cheng-Yu and Li, Chun-Liang and Yeh, Chih-Kuan and Nakhost, Hootan and Fujii, Yasuhisa and Ratner, Alexander and Krishna, Ranjay and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2305.02301},
  year={2023}
}
```
