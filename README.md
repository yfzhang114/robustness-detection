# Assaying on the Robustness of Zero-Shot Machine-Generated Text Detectors

This codebase is the official implementation of [`Assaying on the Robustness of Zero-Shot Machine-Generated Text Detectors`]() (**AAAI Workshop on Responsible Language Models**)


This codebase is mainly based on [DetectGPT](https://github.com/eric-mitchell/detect-gpt).
## Introduction

Existing zero-shot detectors, typically designed for specific tasks or topics, often assume uniform testing scenarios, limiting their practicality. In our research, we explore various advanced Large Language Models (LLMs) and their specialized variants, contributing to this field in several ways. In empirical studies, we uncover a significant correlation between topics and detection performance. Secondly, we delve into the influence of topic shifts on zero-shot detectors. In this regard, we make three key contributions to the research community.

1. Across various models, a correlation between topics and detection performance emerges. It becomes evident that documents centered around low-entropy topics generally pose a greater challenge for detection. To deepen our understanding, we delve into the score distributions and conduct a thorough analysis to uncover the underlying reasons. Additionally, it's noteworthy that different methods exhibit considerable variability in their performance across diverse topics. 

2. We investigate the impact of different types of topic shifts on zero-shot detectors. These shifts include transitions from low to high-entropy topic (Low entropy to high entropy topics refer to instances where documents authored by humans pertain to low-entropy topics and we aim to detect high-entropy documents generated by machines.), shifts from high to low-entropy topics, combinations of documents from different topics, and more.

3. We explore zero-shot detectors to various state-of-the-art LLMs, such as GPT-2, LLaMA, LLaMA-2, and their variants fine-tuned for instructions or dialogues, such as Alpaca. This exploration provides valuable insights into the dynamic landscape of AI-generated text and its evolving capabilities.

## Instructions

First, install the Python dependencies:

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt


If you'd like to run the WritingPrompts experiments, you'll need to download the WritingPrompts data from [here](https://www.kaggle.com/datasets/ratthachat/writing-prompts). Save the data into a directory `data/writingPrompts`.

**Note: Intermediate results are saved in `tmp_results/`. If your experiment completes successfully, the results will be moved into the `results/` directory.**

Second,  run `run.sh`

## Citing the paper
If our work is useful for your own, you can cite us with the following BibTex entry:

    @misc{mitchell2023detectgpt,
        url = {https://arxiv.org/abs/2301.11305},
        author = {Mitchell, Eric and Lee, Yoonho and Khazatsky, Alexander and Manning, Christopher D. and Finn, Chelsea},
        title = {DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature},
        publisher = {arXiv},
        year = {2023},
    }