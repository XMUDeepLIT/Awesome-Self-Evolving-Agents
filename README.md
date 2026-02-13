# Awesome-Agentic-Self-Evolution (Self-Evolution Survey)

<!-- <div align="center">
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg"/></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-green.svg"/></a>
    <a href="https://arxiv.org/abs/2501.13958" target="_blank"><img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&style=flat-square" alt="arXiv:2506.08938"></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/last-commit/DEEP-PolyU/Awesome-GraphRAG?color=blue"/></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/DEEP-PolyU/Awesome-GraphRAG"/></a>
</div> -->


This repository contains a curated list of resources on Agentic Self-Evolution, which are classified according to "[**Agentic Self-Evolution for Large Language Models**](https://arxiv.org/abs/2501.13958)". Continuously updating, stay tuned!

# 🍀 Citation
If you find this survey helpful, please cite our paper:



# 🎉 News
- **[2026-02]** We release the [TTCS](https://github.com/XMUDeepLIT/TTCS), a self-evolving framework.
- **[2026-02]** We release the [Agentic Self-Evolution survey](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

---

<div>
<h3 align="left">
       <p align="center"><img width="100%" src="figs/trend.png" /></p>
    <p align="center"><em>Overview of Agentic Self-Evolution Framework. </em></p>
</div>

Agentic Self-Evolution represents a paradigm shift in AI development, enabling systems to autonomously improve through three key dimensions:

- **Model-Centric Self-Evolving**: Focuses on improving the model itself through inference-based evolution (parallel sampling, sequential self-correction, structured reasoning) and training-based evolution (synthesis-driven offline and exploration-driven online self-evolving).

- **Environment-Centric Self-Evolving**: Enhances the agent's interaction with external knowledge and experience through static knowledge evolution, dynamic experience evolution, modular architecture evolution, and agentic topology evolution.

- **Model-Environment Co-Evolving**: Enables simultaneous evolution of both the model and its environment through environment training and multi-agent policy co-evolution.

---

# Agentic Self-Evolution
Agentic Self-Evolution represents a paradigm of autonomous continuous improvement, where an agent progressively enhances its capabilities through self-driven learning. It is characterized by two essential properties: **(i) Strong autonomy with minimal human supervision**, enabling the agent to generate learning signals without relying on external annotations. **(ii) Actively exploration through interaction**, where the agent actively interacts with itself and the external environment to discover feedback and learning opportunities.

<h3 align="center">
   <p align="center"><img width="100%" src="figs/intro.png" /></p>
    <p align="center"><em>The three dimensions of Agentic Self-Evolution.</em></p>



# 📫 Contact Us
We welcome researchers to share related work to enrich this list or provide insightful comments on our survey on Agentic Self-Evolution. Feel free to reach out to the corresponding authors: [Zhishang Xiang](xiangzhishang@stu.xmu.edu.cn), [Chengyi Yang](yangchengyi@stu.xmu.edu.cn).


## Table of Content
- [🍀 Citation](#-citation)
- [📫 Contact Us](#-contact-us)
- [📈 Trend of Agentic Self-Evolution Research](#-trend-of-agentic-self-evolution-research)
- [📜 Research Papers](#-research-papers)
    - [Model-Centric Self-Evolving](#model-centric-self-evolving)
        - [Inference-Based Evolution](#inference-based-evolution)
            - [Parallel Sampling](#parallel-sampling)
            - [Sequential Self-Correction](#sequential-self-correction)
            - [Structured Reasoning](#structured-reasoning)
        - [Training-Based Evolution](#training-based-evolution)
            - [Synthesis-Driven Offline Self-Evolving](#synthesis-driven-offline-self-evolving)
            - [Exploration-Driven Online Self-Evolving](#exploration-driven-online-self-evolving)
    - [Environment-Centric Self-Evolving](#environment-centric-self-evolving)
        - [Static Knowledge Evolution](#static-knowledge-evolution)
            - [Agentic Retrieval-Augmented Generation](#agentic-retrieval-augmented-generation)
            - [Reasoning-Driven Deep Research](#reasoning-driven-deep-research)
        - [Dynamic Experience Evolution](#dynamic-experience-evolution)
            - [Offline Experience Compilation](#offline-experience-compilation)
            - [Online Experience Adaptation](#online-experience-adaptation)
            - [Lifelong Experience Evolution](#lifelong-experience-evolution)
        - [Modular Architecture Evolution](#modular-architecture-evolution)
            - [Interaction Protocol Evolution](#interaction-protocol-evolution)
            - [Memory Topology Evolution](#memory-topology-evolution)
            - [Tool-Augmented Evolution](#tool-augmented-evolution)
        - [Agentic Topology Evolution](#agentic-topology-evolution)
            - [Offline Architecture Search](#offline-architecture-search)
            - [Runtime Dynamic Adaptation](#runtime-dynamic-adaptation)
            - [Structural Memory Evolution](#structural-memory-evolution)
    - [Model-Environment Co-Evolving](#model-environment-co-evolving)
        - [Environment Training](#environment-training)
            - [Adaptive Curriculum Evolution](#adaptive-curriculum-evolution)
            - [Scalable Environment Evolution](#scalable-environment-evolution)
        - [Multi-Agent Policy Co-Evolution](#multi-agent-policy-co-evolution)
- [📚 Related Survey Papers](#-related-survey-papers)
- [🏆 Benchmarks](#-benchmarks)
- [💻 Open Source Libraries](#-open-source-libraries)
- [⭐ Star History](#-star-history)


# 📈 Trend of Agentic Self-Evolution Research

<h3 align="center">
   <p align="center"><img width="100%" src="figs/trend.png" /></p>
    <p align="center"><em>The development trends in the field of Agentic Self-Evolution with representative works.</em></p>

# 📜 Research Papers

## Model-Centric Self-Evolving


### Inference-Based Evolution

#### Parallel Sampling
- [ICLR'23] **Self-Consistency Improves Chain of Thought Reasoning in Language Models** [[Paper]](https://arxiv.org/abs/2203.11171)
- LLM Calls [[Paper]](https://arxiv.org/abs/2403.02419)
- LLM-Blender [[Paper]](https://arxiv.org/abs/2306.02561)
- Scaling LLM Test-Time Compute [[Paper]](https://arxiv.org/abs/2408.03314)
- SelfCheckGPT [[Paper]](https://arxiv.org/abs/2303.08896)
- PlanSearch [[Paper]](https://arxiv.org/abs/2409.03733)
- Crowd Comparative Reasoning [[Paper]](https://arxiv.org/abs/2502.12501)

#### Sequential Self-Correction
- Self-Refine [[Paper]](https://arxiv.org/abs/2303.17651)
- Self-Debugging [[Paper]](https://arxiv.org/abs/2304.05128)
- Reflexion [[Paper]](https://arxiv.org/abs/2303.11366)
- CRITIC [[Paper]](https://arxiv.org/abs/2305.11738)
- SCORE [[Paper]](https://arxiv.org/abs/2404.17140)
- Mind Evolution [[Paper]](https://arxiv.org/abs/2501.09891)
- Meta-CoT [[Paper]](https://arxiv.org/abs/2501.04682)
- Planning Tokens [[Paper]](https://arxiv.org/abs/2409.03733)
- RaLU [[Paper]](https://arxiv.org/abs/2502.07803)

#### Structured Reasoning
- Tree of Thoughts (ToT) [[Paper]](https://arxiv.org/abs/2305.10601)
- LATS [[Paper]](https://arxiv.org/abs/2310.04406)
- TS-LLM [[Paper]](https://arxiv.org/abs/2309.17179)
- Graph of Thoughts (GoT) [[Paper]](http://dx.doi.org/10.1609/aaai.v38i16.29720)
- Planner-Centric Framework [[Paper]](https://arxiv.org/abs/2511.10037)
- Think-on-Graph (ToG) [[Paper]](https://arxiv.org/abs/2307.07697)
- Think-on-Graph 2.0 [[Paper]](https://arxiv.org/abs/2407.10805)
- Graph Chain-of-Thought (Graph-CoT) [[Paper]](https://arxiv.org/abs/2404.07103)
- Reasoning on Graphs (ROG) [[Paper]](https://arxiv.org/abs/2310.01061)

### Training-Based Evolution

#### Synthesis-Driven Offline Self-Evolving
- Self-Instruct [[Paper]](https://arxiv.org/abs/2212.10560)
- Self-Guide [[Paper]](https://arxiv.org/abs/2407.12874)
- SEAL [[Paper]](https://arxiv.org/abs/2506.10943)
- SPIN [[Paper]](https://arxiv.org/abs/2401.01335)
- SPPO [[Paper]](https://arxiv.org/abs/2405.00675)
- STaR [[Paper]](https://arxiv.org/abs/2203.14465)
- LMSI [[Paper]](https://aclanthology.org/2023.emnlp-main.67/)
- ReST-MCTS* [[Paper]](https://arxiv.org/abs/2406.03816)
- SELF [[Paper]](https://arxiv.org/abs/2310.00533)
- Sirius [[Paper]](https://arxiv.org/abs/2502.04780)
- RAGEN [[Paper]](https://arxiv.org/abs/2504.20073)
- SAMULE [[Paper]](https://arxiv.org/abs/2509.20562)

#### Exploration-Driven Online Self-Evolving
- R-Zero [[Paper]](https://arxiv.org/abs/2508.05004)
- Absolute Zero [[Paper]](https://arxiv.org/abs/2505.03335)
- Language Self-Play (LSP) [[Paper]](https://arxiv.org/abs/2509.07414)
- Self-Questioning LM [[Paper]](https://arxiv.org/abs/2508.03682)
- SeRL [[Paper]](https://arxiv.org/abs/2505.20347)
- Socratic-Zero [[Paper]](https://arxiv.org/abs/2509.24726)
- Agent0 [[Paper]](https://arxiv.org/abs/2511.16043)
- Self-Challenging [[Paper]](https://arxiv.org/abs/2506.01716)
- SPIRAL [[Paper]](https://arxiv.org/abs/2506.24119)
- CURE [[Paper]](https://arxiv.org/abs/2506.03136)
- SPICE [[Paper]](https://arxiv.org/abs/2510.24684)
- LADDER [[Paper]](https://arxiv.org/abs/2503.00735)
- WebRL [[Paper]](https://arxiv.org/abs/2411.02337)
- SPELL [[Paper]](https://arxiv.org/abs/2509.23863)
- R-FEW [[Paper]](https://arxiv.org/abs/2512.02472)
- SKE-Learn [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34590)
- SPC (Self-Play Critic) [[Paper]](https://arxiv.org/abs/2504.19162)
- Misevolution [[Paper]](https://arxiv.org/abs/2509.26354)

## Environment-Centric Self-Evolving

### Static Knowledge Evolution

#### Agentic Retrieval-Augmented Generation
- Self-RAG [[Paper]](https://arxiv.org/abs/2310.11511)
- RAG-Critic [[Paper]](https://aclanthology.org/2025.acl-long.179/)
- Search-o1 [[Paper]](https://arxiv.org/abs/2501.05366)
- Search-R1 [[Paper]](https://arxiv.org/abs/2503.09516)
- MindSearch [[Paper]](https://arxiv.org/abs/2407.20183)
- ODS [[Paper]](https://arxiv.org/abs/2503.20201)

#### Reasoning-Driven Deep Research
- DeepDive [[Paper]](https://arxiv.org/abs/2509.10446)
- DeepSearch [[Paper]](https://arxiv.org/abs/2509.25454)
- DeepResearcher [[Paper]](https://arxiv.org/abs/2504.03160)
- SFR-DeepResearch [[Paper]](https://arxiv.org/abs/2509.06283)
- WebWeaver [[Paper]](https://arxiv.org/abs/2509.13312)
- Tongyi DeepResearch [[Paper]](https://arxiv.org/abs/2510.24701)
- WebThinker [[Paper]](https://arxiv.org/abs/2504.21776)
- SurveyX [[Paper]](https://arxiv.org/abs/2502.14776)
- FINSIGHT [[Paper]](https://arxiv.org/abs/2510.16844)

### Dynamic Experience Evolution

#### Offline Experience Compilation
- AgentRR [[Paper]](https://arxiv.org/abs/2505.17716)
- Agent Workflow Memory (AWM) [[Paper]](https://arxiv.org/abs/2409.07429)
- SkillWeaver [[Paper]](https://arxiv.org/abs/2504.07079)
- Agent KB [[Paper]](https://arxiv.org/abs/2507.06229)
- CoPS [[Paper]](https://arxiv.org/abs/2410.16670)
- Trainable Graph Memory [[Paper]](https://arxiv.org/abs/2511.07800)

#### Online Experience Adaptation
- Dynamic Cheatsheet [[Paper]](https://arxiv.org/abs/2504.07952)
- GEPA [[Paper]](https://arxiv.org/abs/2507.19457)
- Agentic Context Engineering [[Paper]](https://arxiv.org/abs/2510.04618)
- Memento [[Paper]](https://arxiv.org/abs/2508.16153)

#### Lifelong Experience Evolution
- ReasoningBank [[Paper]](https://arxiv.org/abs/2509.25140)
- EVOLVER [[Paper]](https://arxiv.org/abs/2510.16079)
- FLEX [[Paper]](https://arxiv.org/abs/2511.06449)
- Early Experience [[Paper]](https://arxiv.org/abs/2510.08558)
- Training-Free GRPO [[Paper]](https://arxiv.org/abs/2510.08191)
- LatentEvolve [[Paper]](https://arxiv.org/abs/2509.24771)
- ASI [[Paper]](https://arxiv.org/abs/2504.06821)
- AccelOpt [[Paper]](https://arxiv.org/abs/2511.15915)
- Xolver [[Paper]](https://arxiv.org/abs/2506.14234)
- MemGen [[Paper]](https://arxiv.org/abs/2509.24704)
- SAGE [[Paper]](https://arxiv.org/abs/2512.17102)
- AgentEvolver [[Paper]](https://arxiv.org/abs/2511.10395)

### Modular Architecture Evolution

#### Interaction Protocol Evolution
- Think-in-Memory (TiM) [[Paper]](https://arxiv.org/abs/2311.08719)
- Memory-of-Thought (MoT) [[Paper]](https://arxiv.org/abs/2305.05181)
- ReadAgent [[Paper]](https://arxiv.org/abs/2402.09727)
- MemoryBank [[Paper]](https://arxiv.org/abs/2305.10250)
- General Agentic Memory (GAM) [[Paper]](https://arxiv.org/abs/2511.18423)
- MemGPT [[Paper]](https://arxiv.org/abs/2310.08560)
- LightMem [[Paper]](https://arxiv.org/abs/2510.18866)
- AgentFold [[Paper]](https://arxiv.org/abs/2510.24699)
- HierSearch [[Paper]](https://arxiv.org/abs/2508.08088)

#### Memory Topology Evolution
- A-MEM [[Paper]](https://arxiv.org/abs/2502.12110)
- Mem0 [[Paper]](https://arxiv.org/abs/2504.19413)
- CAM [[Paper]](https://arxiv.org/abs/2510.05520)
- MemAct [[Paper]](https://arxiv.org/abs/2510.12635)
- Mem-α [[Paper]](https://arxiv.org/abs/2509.25911)
- EvoRoute [[Paper]](https://arxiv.org/abs/2601.02695)
- MemEvolve [[Paper]](https://arxiv.org/abs/2512.18746)

#### Tool-Augmented Evolution
- ReAct [[Paper]](https://arxiv.org/abs/2210.03629)
- WebGPT [[Paper]](https://arxiv.org/abs/2112.09332)
- PAL [[Paper]](https://arxiv.org/abs/2211.10435)
- LATM [[Paper]](https://arxiv.org/abs/2305.17126)
- CREATOR [[Paper]](https://arxiv.org/abs/2305.14318)
- CRAFT [[Paper]](https://arxiv.org/abs/2309.17428)
- TOOLMAKER [[Paper]](https://arxiv.org/abs/2502.11705)
- VOYAGER [[Paper]](https://arxiv.org/abs/2305.16291)

### Agentic Topology Evolution

#### Offline Architecture Search
- GPTSwarm [[Paper]](https://arxiv.org/abs/2402.16823)
- AutoFlow [[Paper]](https://arxiv.org/abs/2407.12821)
- ADAS [[Paper]](https://arxiv.org/abs/2408.08435)
- AFLOW [[Paper]](https://arxiv.org/abs/2410.10762)
- MAS-GPT [[Paper]](https://arxiv.org/abs/2503.03686)

#### Runtime Dynamic Adaptation
- AutoAgents [[Paper]](https://arxiv.org/abs/2309.17288)
- EVOAGENT [[Paper]](https://arxiv.org/abs/2406.14228)
- MASS [[Paper]](https://arxiv.org/abs/2502.02533)
- AGP [[Paper]](https://arxiv.org/abs/2506.02951)
- G-Designer [[Paper]](https://arxiv.org/abs/2410.11782)
- MaAS [[Paper]](https://arxiv.org/abs/2502.04180)
- ReMA [[Paper]](https://arxiv.org/abs/2503.09501)

#### Structural Memory Evolution
- SEDM [[Paper]](https://arxiv.org/abs/2509.09498)
- G-Memory [[Paper]](https://arxiv.org/abs/2506.07398)
- Collaborative Memory [[Paper]](https://arxiv.org/abs/2505.18279)
- LatentMAS [[Paper]](https://arxiv.org/abs/2511.20639)

## Model-Environment Co-Evolving

### Environment Training

#### Adaptive Curriculum Evolution
- GenEnv [[Paper]](https://arxiv.org/abs/2512.19682)
- Environment Tuning [[Paper]](https://arxiv.org/abs/2510.10197)
- RLVE [[Paper]](https://arxiv.org/abs/2511.07317)

#### Scalable Environment Evolution
- DreamGym [[Paper]](https://arxiv.org/abs/2511.03773)
- AutoEnv [[Paper]](https://arxiv.org/abs/2511.19304)
- Endless Terminals [[Paper]](https://arxiv.org/abs/2601.16443)
- Reasoning Gym [[Paper]](https://arxiv.org/abs/2505.24760)
- GEM [[Paper]](https://arxiv.org/abs/2510.01051)
- AgentGym [[Paper]](https://arxiv.org/abs/2406.04151)

### Multi-Agent Policy Co-Evolution
- OPTIMA [[Paper]](https://arxiv.org/abs/2410.08115)
- MAPoRL [[Paper]](https://arxiv.org/abs/2502.18439)
- MARFT [[Paper]](https://arxiv.org/abs/2504.16129)
- CoMAS [[Paper]](https://arxiv.org/abs/2510.08529)

# 📚 Related Survey Papers
- (arXiv 2025) **Retrieval-Augmented Generation with Graphs (GraphRAG)** [[Paper]](https://arxiv.org/abs/2501.00309)
- (arXiv 2024) **Graph Retrieval-Augmented Generation: A Survey** [[Paper]](https://arXiv.org/pdf/2408.08921)
- (AIxSET 2024) **Graph Retrieval-Augmented Generation for Large Language Models: A Survey** [[Paper]](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=4895062)


# 🏆 Benchmarks

## Intrinsic Capabilities

### General Knowledge
| Name | Domain | Modality | Feature | Link | Paper |
| --- | --- | --- | --- | --- | --- |
| MMLU-Pro | General Knowledge | Text | Robust Reasoning, 10-Choice | [🤗 HF](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | [[Paper]](https://arxiv.org/abs/2406.01574) |
| HotpotQA | General Knowledge | Text | Multi-hop Reasoning, Wiki-based | [🤗 HF](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | [[Paper]](https://arxiv.org/abs/1809.09600) |
| MMLU | General Knowledge | Text | Massive Multitask, 57 Disciplines | [🤗 HF](https://huggingface.co/datasets/cais/mmlu) | [[Paper]](https://arxiv.org/abs/2009.03300) |
| MuSiQue | General Knowledge | Text | Connected Multi-hop, Robustness | [💻 GitHub](https://github.com/StonyBrookNLP/musique) | [[Paper]](https://aclanthology.org/2022.tacl-1.31/) |
| NQ | General Knowledge | Text | Real User Queries, Open-Domain | [💻 GitHub](https://github.com/google-research-datasets/natural-questions) | [[Paper]](https://aclanthology.org/Q19-1026/) |
| TriviaQA | General Knowledge | Text | Reading Comprehension, Triples | [🤗 HF](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | [[Paper]](https://arxiv.org/abs/1705.03551) |
| PopQA | General Knowledge | Text | Long-Tail Knowledge, RAG Focus | [🤗 HF](https://huggingface.co/datasets/akariasai/PopQA) | [[Paper]](https://aclanthology.org/2023.acl-long.546/) |
| 2WikiMultiHopQA | General Knowledge | Text | Structured Multi-hop, Explanations | [🤗 HF](https://huggingface.co/datasets/xanhho/2WikiMultihopQA) | [[Paper]](https://arxiv.org/abs/2011.01060) |
| BBH | General Knowledge | Text | Challenging Tasks, CoT Focus | [💻 GitHub](https://github.com/suzgunmirac/BIG-Bench-Hard) | [[Paper]](https://aclanthology.org/2023.findings-acl.824/) |
| AGIEval | General Knowledge | Text | Human-Centric Exams, General | [💻 GitHub](https://github.com/ruixiangcui/AGIEval) | [[Paper]](https://aclanthology.org/2024.findings-naacl.149/) |
| ARC | General Knowledge | Visual | Abstraction, Few-Shot Reasoning | [💻 GitHub](https://github.com/fchollet/ARC-AGI) | [[Paper]](https://arxiv.org/abs/1911.01547) |
| NarrativeQA | General Knowledge | Text | Long Context, Story Understanding | [💻 GitHub](https://github.com/google-deepmind/narrativeqa) | [[Paper]](https://aclanthology.org/Q18-1023/) |
| LongBench | General Knowledge | Text | Long Context, Multi-Task Eval | [💻 GitHub](https://github.com/THUDM/LongBench) | [[Paper]](https://aclanthology.org/2024.acl-long.170/) |

### Scientific Reasoning
| Name | Domain | Modality | Feature | Link | Paper |
| --- | --- | --- | --- | --- | --- |
| GPQA | Scientific Reasoning | Text | Google-Proof, PhD-Level Experts | [💻 GitHub](https://github.com/idavidrein/gpqa) | [[Paper]](https://arxiv.org/abs/2311.12022) |
| SciBench | Scientific Reasoning | Text | College Science, Calculation | [💻 GitHub](https://github.com/mandyyyyii/scibench) | [[Paper]](https://arxiv.org/abs/2307.10635) |
| ChemBench | Scientific Reasoning | Text | Chemistry, Autonomous Labs | [💻 GitHub](https://github.com/lamalab-org/chembench) | [[Paper]](https://arxiv.org/abs/2404.01475) |
| SciQA | Scientific Reasoning | Text | Scientific QA, Knowledge Graph | [🤗 HF](https://huggingface.co/datasets/orkg/SciQA) | [[Paper]](https://www.nature.com/articles/s41598-023-33607-z) |

### Mathematical Reasoning
| Name | Domain | Modality | Feature | Link | Paper |
| --- | --- | --- | --- | --- | --- |
| AIME | Mathematical Reasoning | Text | Competition Math, Hard Difficulty | [🤗 HF](https://huggingface.co/datasets/math-ai/aime25) | [[Paper]](https://huggingface.co/datasets/math-ai/aime25) |
| OlympiadBench | Mathematical Reasoning | Multimodal | Visual Reasoning, Olympiad-Level | [💻 GitHub](https://github.com/OpenBMB/OlympiadBench) | [[Paper]](https://aclanthology.org/2024.acl-long.210/) |
| GSM8K | Mathematical Reasoning | Text | Grade School Math, Chain-of-Thought | [🤗 HF](https://huggingface.co/datasets/openai/gsm8k) | [[Paper]](https://arxiv.org/abs/2110.14168) |
| MATH | Mathematical Reasoning | Text | Challenging Math, Diverse Topics | [🤗 HF](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark) | [[Paper]](https://arxiv.org/abs/2103.03874) |
| AMC | Mathematical Reasoning | Text | Pre-Olympiad, Competition Math | [🤗 HF](https://huggingface.co/datasets/zwhe99/amc23) | [[Paper]](https://huggingface.co/datasets/zwhe99/amc23) |

### Code Generation
| Name | Domain | Modality | Feature | Link | Paper |
| --- | --- | --- | --- | --- | --- |
| LiveCodeBench | Code Generation | Text | Contamination-Free, Dynamic | [💻 GitHub](https://github.com/LiveCodeBench/LiveCodeBench) | [[Paper]](https://openreview.net/forum?id=chfJJYC3iL) |
| BigCodeBench | Code Generation | Text | Complex Libraries, Instruction | [💻 GitHub](https://github.com/bigcode-project/bigcodebench) | [[Paper]](https://openreview.net/forum?id=YrycTjllL0) |
| HumanEval | Code Generation | Text | Functional Correctness, Synthesis | [💻 GitHub](https://github.com/openai/human-eval) | [[Paper]](https://arxiv.org/abs/2107.03374) |
| MBPP | Code Generation | Text | Basic Programming, Semantic | [💻 GitHub](https://github.com/google-research/google-research/blob/master/mbpp/README.md) | [[Paper]](https://arxiv.org/abs/2108.07732) |

## Agentic Reasoning Capabilities

### Web Navigation
| Name | Domain | Modality | Feature | Link | Paper |
| --- | --- | --- | --- | --- | --- |
| WebArena | Web Navigation | Text | Realistic Tasks, Long-Horizon | [💻 GitHub](https://github.com/web-arena-x/webarena) | [[Paper]](https://openreview.net/forum?id=oKn9c6ytLx) |
| WebShop | Web Navigation | Text | E-commerce, Decision Making | [💻 GitHub](https://github.com/princeton-nlp/WebShop) | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html) |
| MT-Mind2Web | Web Navigation | Text | Multi-Turn, Generalization | [🤗 HF](https://huggingface.co/datasets/magicgh/MT-Mind2Web) | [[Paper]](https://aclanthology.org/2024.acl-long.475/) |
| Mind2Web | Web Navigation | Text | Generalist Agent, Real Websites | [💻 GitHub](https://github.com/OSU-NLP-Group/Mind2Web) | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5950bf290a1570ea401bf98882128160-Abstract-Datasets_and_Benchmarks.html) |
| WebVoyager | Web Navigation | Multimodal | End-to-End, Visual Navigation | [💻 GitHub](https://github.com/MinorJerry/WebVoyager) | [[Paper]](https://arxiv.org/abs/2401.13919) |
| VisualWebArena | Web Navigation | Multimodal | Visual/HTML, Interactive | [💻 GitHub](https://github.com/web-arena-x/visualwebarena) | [[Paper]](https://aclanthology.org/2024.acl-long.49/) |

### Tool Usage
| Name | Domain | Modality | Feature | Link | Paper |
| --- | --- | --- | --- | --- | --- |
| ToolLLM | Tool Usage | Text | Large-Scale APIs, Instruction Tuning | [💻 GitHub](https://github.com/OpenBMB/ToolBench) | [[Paper]](https://openreview.net/forum?id=dHng2O0Jjr) |

### Unified Frameworks
| Name | Domain | Modality | Feature | Link | Paper |
| --- | --- | --- | --- | --- | --- |
| AgentGym | Unified Frameworks | Multimodal | Interactive Learning, Diversity | [💻 GitHub](https://github.com/WooooDyy/AgentGym) | [[Paper]](https://arxiv.org/abs/2406.04151) |
| AgentBoard | Unified Frameworks | Multimodal | Analytic Dashboard, Unified | [💻 GitHub](https://github.com/hkust-nlp/AgentBoard) | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/51ac6936ab73c83e65fb25ed28ccb2b4-Abstract-Datasets_and_Benchmarks.html) |
| Reasoning Gym | Unified Frameworks | Text | Algorithmic, Dynamic Tasks | [💻 GitHub](https://github.com/open-thought/reasoning-gym) | [[Paper]](https://arxiv.org/abs/2505.24760) |
| ALFWorld | Unified Frameworks | Text | Text-World, Household Tasks | [💻 GitHub](https://github.com/alfworld/alfworld) | [[Paper]](https://arxiv.org/abs/2010.03768) |
| AgentBench | Unified Frameworks | Text | Comprehensive, Multi-Environment | [💻 GitHub](https://github.com/THUDM/AgentBench) | [[Paper]](https://arxiv.org/abs/2308.03688) |
| GAIA | Unified Frameworks | Multimodal | General Assistant, Hard Tasks | [🤗 HF](https://huggingface.co/gaia-benchmark) | [[Paper]](https://openreview.net/forum?id=oOte_397Q4) |

### Software Engineering & OS Operations
| Name | Domain | Modality | Feature | Link | Paper |
| --- | --- | --- | --- | --- | --- |
| SWE-bench | Software Engineering | Text | Real GitHub Issues, Patch Gen | [💻 GitHub](https://github.com/SWE-bench/SWE-bench) | [[Paper]](https://openreview.net/forum?id=VTF8yNQM66) |
| Terminal-Bench | OS Operations | Text | Linux Command Line, Security | [💻 GitHub](https://github.com/laude-institute/terminal-bench) | [[Paper]](https://github.com/laude-institute/terminal-bench) |
| OSWorld | OS Operations | Multimodal | GUI/Desktop, Cross-App | [💻 GitHub](https://github.com/xlang-ai/OSWorld) | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/82df44e3d94a6a3ff7e3e6edd8f4caa7-Abstract-Datasets_and_Benchmarks.html) |

# 💻 Open Source Libraries

## Foundational Agent Orchestration
| Library | Key Features | Link | Paper |
| --- | --- | --- | --- |
| **LangGraph** | Enables multi-actor applications with cyclic graphs for complex looping logic | [💻 GitHub](https://github.com/langchain-ai/langgraph) | [[Paper]](https://github.com/langchain-ai/langchain) |
| **LlamaIndex** | Integrates private data with LLMs via robust connectors and query engines | [💻 GitHub](https://github.com/run-llama/llama_index) | [[Paper]](https://github.com/jerryjliu/llama_index) |
| **AutoGen** | Automates tasks via customizable agents using conversation and tool integration | [💻 GitHub](https://github.com/microsoft/autogen) | [[Paper]](https://openreview.net/forum?id=uAjxFFing2) |
| **MetaGPT** | Encodes SOPs into LLMs for role-based software development | [💻 GitHub](https://github.com/FoundationAgents/MetaGPT) | [[Paper]](https://openreview.net/forum?id=VtmBAGCN7o) |

## Distributed Training
| Library | Key Features | Link | Paper |
| --- | --- | --- | --- |
| **Megatron-LM** | Facilitates high-performance training utilizing multi-dimensional parallelism | [💻 GitHub](https://github.com/NVIDIA/Megatron-LM) | [[Paper]](https://arxiv.org/abs/1909.08053) |
| **DeepSpeed** | Optimizes memory efficiency featuring ZeRO technology | [💻 GitHub](https://github.com/deepspeedai/DeepSpeed) | [[Paper]](https://dl.acm.org/doi/10.1145/3394486.3406703) |

## Post-training & Alignment
| Library | Key Features | Link | Paper |
| --- | --- | --- | --- |
| **VeRL** | Provides a HybridFlow-based RL library with 3D-HybridEngine | [💻 GitHub](https://github.com/verl-project/verl) | [[Paper]](https://arxiv.org/abs/2409.19256) |
| **OpenRLHF** | Supports distributed RLHF based on Ray and vLLM frameworks | [💻 GitHub](https://github.com/OpenRLHF/OpenRLHF) | [[Paper]](https://arxiv.org/abs/2405.11143) |
| **TRL** | Offers a full-stack library for SFT, Reward Modeling, and RL alignment | [💻 GitHub](https://github.com/huggingface/trl) | [[Paper]](https://github.com/huggingface/trl) |

## Efficient Fine-tuning
| Library | Key Features | Link | Paper |
| --- | --- | --- | --- |
| **LLaMA Factory** | Provides a unified "code-free" WebUI supporting 100+ models | [💻 GitHub](https://github.com/hiyouga/LlamaFactory) | [[Paper]](https://aclanthology.org/2024.acl-demos.41/) |
| **Unsloth** | Accelerates training via manually derived backpropagation and Triton kernels | [💻 GitHub](https://github.com/unslothai/unsloth) | [[Paper]](http://github.com/unslothai/unsloth) |

## Inference & Serving
| Library | Key Features | Link | Paper |
| --- | --- | --- | --- |
| **vLLM** | Serves models with high throughput utilizing PagedAttention | [💻 GitHub](https://github.com/vllm-project/vllm) | [[Paper]](https://dl.acm.org/doi/10.1145/3600006.3613165) |
| **SGLang** | Manages structured generation using RadixAttention for aggressive cache reuse | [💻 GitHub](https://github.com/sgl-project/sglang) | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c7dc7d1dd443c422b8a4063853de4e23-Abstract-Conference.html) | 

# 🍀 Citation
If you find this survey helpful, please cite our paper:



