# G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems

## 🌎 Setup
```
conda create -n GMemory python=3.12
conda activate GMemory
pip install -r requirements.txt
```

## 🚀 Quick Start

### 🔑 Add API keys in template.env and change its name to .env
```
OPENAI_API_BASE = "" # the BASE_URL of OpenAI LLM backend
OPENAI_API_KEY = "" # for OpenAI LLM backend
```

### 🔎 Getting Started
- Available memories: ***Empty, ChatDev, MetaGPT, Voyager, Generative, MemoryBank, G-Memory***
- Available MAS: ***AutoGen, DyLAN, MacNet***

### ▶️ How to Run
- Option 1: Run with Shell Script. Simply execute the following script:
    ```
    ./run_mas.sh
    ```
- Option 2: Run with Python Command. You can also launch specific tasks via command-line:
    ```
    python tasks/run.py --task alfworld --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <your model here>
    python tasks/run.py --task pddl --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <your model here>
    python tasks/run.py --task fever --reasoning io --mas_memory g-memory --mas_trials 15 --mas_type autogen --model <your model here>
    ```

## 🙏 Acknowledgement
- We sincerely thank [ExpeL](https://github.com/LeapLabTHU/ExpeL) for providing their prompt designs.
- We also extend our heartfelt thanks to [AgentSquare](https://github.com/tsinghua-fib-lab/AgentSquare) for their dataset environments and baseline implementations.
