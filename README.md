<h1 align="center"><b>awesome-large-graph-model</b></h1>
<p align="center">
    <a href="https://github.com/THUMNLab/awesome-large-graph-model/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-green" alt="PRs"></a>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="awesome"></a>
    <!-- <a href="https://"><img src="https://img.shields.io/badge/-Website-grey?logo=svelte&logoColor=white" alt="Website"></a> -->
    <img src="https://img.shields.io/github/stars/THUMNLab/awesome-large-graph-model?color=yellow&label=Star" alt="Stars" >
    <img src="https://img.shields.io/github/forks/THUMNLab/awesome-large-graph-model?color=blue&label=Fork" alt="Forks" >
</p>

This repository contains a paper list related to **Large Graph Models**. Similar to Large Language Models (LLMs) for natural languages, we believe large graph models will revolutionaize graph machine learning with exciting opportunities for both researchers and practioners! For more details, please refer to our perspective paper: [Large Graph Models: A Perspective](https://arxiv.org/pdf/2308.14522)

We will try our best to make this paper list updated. If you notice some related papers missing or have any suggestion, do not hesitate to contact us via pull requests at our repo.

# Papers

## Perspective and Survey
- [arXiv 2023.08] Large Graph Models: A Perspective [[paper]](https://arxiv.org/pdf/2308.14522)
- [arXiv 2023.10] Integrating Graphs with Large Language Models: Methods and Prospects [[paper]](https://arxiv.org/pdf/2310.05499)
- [arXiv 2023.03] A Survey of Graph Prompting Methods: Techniques, Applications, and Challenges [[paper]](https://arxiv.org/pdf/2303.07275.pdf)

## Model

#### LLMs as Graph Models
- [NeurIPS 2023] Can Language Models Solve Graph Problems in Natural Language? [[paper]](https://arxiv.org/pdf/2305.10037) [[code]](https://github.com/Arthur-Heng/NLGraph)
- [arXiv 2023.10] GraphLLM: Boosting Graph Reasoning Ability of Large Language Model [[paper]](https://arxiv.org/pdf/2310.05845) [[code]](https://github.com/mistyreed63849/Graph-LLM)
- [arXiv 2023.10] Beyond Text: A Deep Dive into Large Language Models’ Ability on Understanding Graph Data [[paper]](https://arxiv.org/pdf/2310.04944)
- [arXiv 2023.10] Talk like a Graph: Encoding Graphs for Large Language Models [[paper]](https://arxiv.org/pdf/2310.04560)
- [arXiv 2023.10] GraphText Graph Reasoning in Text Space [[paper]](https://arxiv.org/pdf/2310.01089)
- [arXiv 2023.10] One for All Towards Training One Graph Model for All Classification Tasks [[paper]](https://arxiv.org/pdf/2310.00149) [[code]](https://github.com/LechengKong/OneForAll)
- [arXiv 2023.09] Can LLMs Effectively Leverage Structural Information for Graph Learning: When and Why [[paper]](https://arxiv.org/pdf/2309.16595) [[code]](https://github.com/TRAIS-Lab/LLM-Structured-Data)
- [arXiv 2023.08] Evaluating Large Language Models on Graphs: Performance Insights and Comparative Analysis [[paper]](https://arxiv.org/pdf/2308.11224) [[code]](https://github.com/Ayame1006/LLMtoGraph)
- [arXiv 2023.08] Natural Language is All a Graph Needs [[paper]](https://arxiv.org/pdf/2308.07134) 
- [arXiv 2023.07] Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs [[paper]](https://arxiv.org/pdf/2307.03393) [[code]](https://github.com/CurryTang/Graph-LLM)
- [arXiv 2023.05] GPT4Graph: Can Large Language Models Understand Graph Structured Data ? An Empirical Evaluation and Benchmarking [[paper]](https://arxiv.org/pdf/2305.15066) 


#### LLM for GNNs
- [arXiv 2023.10] Label-free Node Classification on Graphs with Large Language Models (LLMS) [[paper]](https://arxiv.org/pdf/2310.04668)
- [arXiv 2023.08] SimTeG: A Frustratingly Simple Approach Improves Textual Graph Learning [[paper]](https://arxiv.org/pdf/2308.02565) [[code]](https://github.com/vermouthdky/SimTeG)
- [arXiv 2023.05] Explanations as Features LLM-Based Features for Text-Attributed Graphs [[paper]](https://arxiv.org/pdf/2305.19523) [[code]](https://github.com/XiaoxinHe/TAPE)

#### Graph Prompts
- [NeurIPS 2023] PRODIGY: Enabling In-context Learning Over Graphs [[paper]](https://arxiv.org/pdf/2305.12600.pdf) [[code]](https://github.com/snap-stanford/prodigy)
- [NeurIPS 2023] Universal Prompt Tuning for Graph Neural Networks [[paper]](https://arxiv.org/pdf/2209.15240.pdf) 
- [CVPR 2023] Deep Graph Reprogramming [[paper]](https://arxiv.org/pdf/2304.14593.pdf)
- [KDD 2023] All in One: Multi-Task Prompting for Graph Neural Networks [[paper]](https://arxiv.org/pdf/2307.01504) [[code]](https://github.com/sheldonresearch/ProG)
- [WWW 2023] GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks [[paper]](https://arxiv.org/pdf/2302.08043) [[code]](https://github.com/Starlien95/GraphPrompt)
- [WWW 2023] Structure Pre-training and Prompt Tuning for Knowledge Graph Transfer [[paper]](https://arxiv.org/pdf/2303.03922.pdf) [[code]](https://github.com/zjukg/KGTransformer)
- [KDD 2022] GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks [[paper]](https://dl.acm.org/doi/10.1145/3534678.3539249)
- [arXiv 2023.09] Prompt-based Node Feature Extractor for Few-shot Learning on Text-Attributed Graphs [[paper]](https://arxiv.org/pdf/2309.02848.pdf)
- [arXiv 2023.09] Deep Prompt Tuning for Graph Transformers [[paper]](https://arxiv.org/pdf/2309.10131.pdf)
- [arXiv 2023.02] SGL-PT: A Strong Graph Learner with Graph Prompt Tuning [[paper]](https://arxiv.org/pdf/2302.12449.pdf)

#### Graph Parameter-efficient Fine-tuning
- [arXiv 2023.08] Search to Fine-tune Pre-trained Graph Neural Networks for Graph-level Tasks [[paper]](https://arxiv.org/pdf/2308.06960)
- [arXiv 2023.05] G-Adapter: Towards Structure-Aware Parameter-Efficient Transfer Learning for Graph Transformer Networks [[paper]](https://arxiv.org/pdf/2305.10329)
- [arXiv 2023.04] AdapterGNN: Efficient Delta Tuning Improves Generalization Ability in Graph Neural Networks [[paper]](https://arxiv.org/pdf/2304.09595)

## Applications
#### Knowledge Graph
- [arXiv 2023.08] Large Language Models and Knowledge Graphs: Opportunities and Challenges [[paper]](https://arxiv.org/pdf/2308.0637)
- [arXiv 2023.06] Unifying Large Language Models and Knowledge Graphs: A Roadmap [[paper]](https://arxiv.org/pdf/2306.08302)

#### Molecules
- [arXiv 2023.09] DrugChat: Towards Enabling ChatGPT-Like Capabilities on Drug Molecule Graphs [[paper]](https://arxiv.org/pdf/2309.03907) [[code]](https://github.com/UCSD-AI4H/drugchat)
- [arXiv 2023.08] GIT-Mol A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text [[paper]](https://arxiv.org/pdf/2308.069)
- [arXiv 2023.07] Can Large Language Models Empower Molecular Property Prediction? [[paper]](https://arxiv.org/pdf/2307.07443) [[code]](https://github.com/ChnQ/LLM4Mol)

#### Neural Architecture Search
- [arXiv 2023.10] Graph Neural Architecture Search with GPT-4 [[paper]](https://arxiv.org/pdf/2310.01436)
- [arXiv 2023.02] EvoPrompting: Language Models for Code-Level Neural Architecture Search [[paper]](https://arxiv.org/pdf/2302.14838) [[code]](https://github.com/algopapi/EvoPrompting_Reinforcement_learning)

#### Miscellaneous
- [arXiv 2023.10] AUTOPARLLM: GNN-Guided Automatic Code Parallelization using Large Language Models [[paper]](https://arxiv.org/pdf/2310.04047)
- [arXiv 2023.09] VulnSense: Efficient Vulnerability Detection in Ethereum Smart Contracts by Multimodal Learning with Graph Neural Network and Language Model [[paper]](https://arxiv.org/pdf/2309.08474)
- [arXiv 2023.08] FoodGPT A Large Language Model in Food Testing Domain with Incremental Pre-training and Knowledge Graph Prompt [[paper]](https://arxiv.org/pdf/2308.10173)
- [arXiv 2023.06] ChatGPT Informed Graph Neural Network for Stock Movement Prediction [[paper]](https://arxiv.org/pdf/2306.03763)
- [arXiv 2023.05] Graph Meets LLM A Novel Approach to Collaborative Filtering for Robust Conversational Understanding [[paper]](https://arxiv.org/pdf/2305.14449)

## Graphs for LLMs
#### Graph of Thoughts
- [arXiv 2023.08] MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models [[paper]](https://arxiv.org/pdf/2308.09729)
- [arXiv 2023.08] Graph of Thoughts: Solving Elaborate Problems with Large Language Models [[paper]](https://arxiv.org/pdf/2308.09687)  [[code]](https://github.com/spcl/graph-of-thoughts)
- [arXiv 2023.08] Enhancing Reasoning Capabilities of Large Language Models: A Graph-Based Verification Approach [[paper]](https://arxiv.org/pdf/2308.09267)
- [arXiv 2023.08] Boosting Logical Reasoning in Large Language Models through a New Framework: The Graph of Thought [[paper]](https://arxiv.org/pdf/2308.08614)
- [arXiv 2023.08] Thinking Like an Expert: Multimodal Hypergraph-of-Thought (HoT) Reasoning to boost Foundation Modals [[paper]](https://arxiv.org/pdf/2308.06207)
- [arXiv 2023.07] Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph [[paper]](https://arxiv.org/pdf/2307.07697)
- [arXiv 2023.05] Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Large Language Model [[paper]](https://arxiv.org/pdf/2305.16582)

#### Graph as Tools
- [arXiv 2023.05] StructGPT: A General Framework for Large Language Model to Reason over Structured Data [[paper]](https://arxiv.org/pdf/2305.09645)  [[code]](https://github.com/RUCAIBox/StructGPT)
- [arXiv 2023.04] Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT [[paper]](https://arxiv.org/pdf/2304.11116)  [[code]](https://github.com/jwzhanggy/Graph_Toolformer)


## Cite

Please consider citing our [perspective paper](https://arxiv.org/pdf/2308.14522) if you find this repository helpful:
```
@article{zhang2023large,
  title={Large Graph Models: A Perspective},
  author={Zhang, Ziwei and Li, Haoyang and Zhang, Zeyang and Qin, Yijian and Wang, Xin and Zhu, Wenwu},
  journal={arXiv preprint arXiv:2308.14522},
  year={2023}
}
``` 
