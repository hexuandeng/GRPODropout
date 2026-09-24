# Data preparation guide (data/)

This directory holds the preprocessing scripts for each dataset, which convert raw
datasets into the parquet / jsonl formats required for verl RL training.

## Important notes

- **Raw dataset files are not distributed with the repository.** This directory does
  not contain the actual data files for each dataset. You need to download the raw
  data from the sources below, place it in the corresponding subdirectory, and then
  convert it with the preprocessing scripts.
- The preprocessing scripts look for each dataset subdirectory in the **script's own
  directory** (i.e. this `data/` directory) by default.
- Each dataset lives in its own subdirectory under `data/` (e.g. `data/AMC23/`,
  `data/AIME24/`, `data/dapo17k/`, `data/Olympiad/`).

## Directory layout

The default layout expected by the scripts is as follows (rooted at `DATA_ROOT`,
which defaults to this `data/` directory):

```
data/
├── AMC23/
│   └── AMC23.parquet            # Raw input (download and place it yourself)
├── AIME24/
│   └── AIME24.parquet           # Raw input (download and place it yourself)
├── dapo17k/
│   └── dapo17k.parquet          # Raw input (download and place it yourself)
├── Olympiad/                    # OlympiadBench output directory
├── olympiadbench.jsonl          # OlympiadBench raw input (download and place it yourself)
└── MMLU-pro/                    # MMLU-Pro data directory
```

The converted verl train/validation files (`*_verl_train.parquet` /
`*_verl_val.parquet`) are written into the corresponding subdirectory.

## Custom data root directory

All preprocessing scripts support overriding the default data root directory via the
`DATA_ROOT` environment variable:

```bash
DATA_ROOT=/path/to/your/data python3 prepare_amc_aime_for_verl.py
```

When `DATA_ROOT` is not set, the script's own directory (i.e. this `data/` directory)
is used by default.

## Retained datasets and their preprocessing scripts

| Dataset | Type | Preprocessing script / loading method |
| --- | --- | --- |
| AMC23 & AIME24 | MATH | `prepare_amc_aime_for_verl.py` |
| DAPO-17k | MATH | `prepare_dapo17k_for_verl.py` |
| OlympiadBench | MATH | `prepare_olympiad_for_verl.py` |
| MMLU-Pro | Multiple choice | Loaded via `verl/examples/grpo_dropout/mmlu_pro_dataset.py` |

## Helper utility scripts

The following scripts are not required preprocessing steps; they are just helper
utilities:

- `convert.py`: Convert cleaned JSON / JSONL files into verl-compatible parquet files.
- `data_view.py`: Convert a parquet file into JSON for easier content inspection.

Both scripts also support the `DATA_ROOT` environment variable.
