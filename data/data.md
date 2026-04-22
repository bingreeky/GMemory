# Dataset
## ALFWorld

For the tasks suffix json
```
curl -o alfworld_tasks_suffix.json https://raw.githubusercontent.com/LeapLabTHU/ExpeL/e41ec9a24823e7b560c561ab191441b56d9bcefc/data/alfworld/alfworld_tasks_suffix.json
```

For PPDL and game files

```
curl -L -o alfworld.zip https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip
```

## PDDL

```
curl -L -o data.tar.gz https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz

tar -zxvf data.tar.gz
```

Get the test.jsonl from data/pddl/test.jsonl

## FEVER
curl -L -o train.jsonl https://fever.ai/download/fever/train.jsonl
