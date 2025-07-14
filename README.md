# Probabilistic LTSF: Investigating a DMS-IMS Trade-off

This repo contains the code for my master thesis. The codebase is based on [BasicTS+](https://github.com/GestaltCogTeam/BasicTS), but expands it by adding probabilistic functionalities to the underlying models.
I opted to not use [ProbTS](https://github.com/microsoft/ProbTS), since pytorch-lightning abstracts the implementation too much for my liking compared to easytorch where pipeline changes are made more quickly. Additionally, BasicTS+ support more relevant LTSF models. Nonetheless, comparing standard probabilistic models, such as those implemented in ProbTS could be an interesting future avenue.

Let's start by introducing the recommended file structure, then discuss important pieces and why it's included. Below, files are shown first and folders after denoted with a `/` appended such as `folder/`.

## File structure
```
├── README.md              <- The top-level README for developers using this project
├── environment.yml        <- The requirements file for reproducing the environment
├── requirements.txt       <- Alternative, requirements .txt file for reproducing the environment
├── .gitignore             <- Git ignore file
├── BasicTS/               <- The cloned repository from BasicTS+
    ├── baselines/         <- The source code and configs of all models
    │
    ├── basicts/           <- Contains the pipeline
    │   ├── data/          <- Dataset loaders
    │   ├── metrics/       <- Evaluation and Training metrics
    │   ├── runners/       <- Pipeline runners
    │   ├── scaler/        <- Scalers
    │   ├── utils/         
    │   └── launcher.py    <- Entry point for training, hpo and testing
    │
    ├── examples/          <- Example configs from BasicTS
    │
    ├── experiments/       <- .py scripts for training, hpo and testing
    │
    ├── final_weights/     <- Config files from the best model runs after hpo
    │   ├── ETTh1/         
    │   └── ETTm1/
    │
    ├── hp_tuning/         <- .yaml files for the hyperparameter optimization
    │   ├── ETTh1/         
    │   └── ETTm1/ 
    │
    ├── prob/              <- Added probabilistic heads
    │
    ├── scripts/           <- Initial scripts to get BasicTS+ project started
    │
    ├── slurm/             <- Slurm scripts used throughout this project
    │   ├── ETTh1/         
    │   └── ETTm1/ 
    │
    ├── tests/             <- Test from BasicTS+
    │
    └── tutorial/          <- .md Tutorials from BasicTS+
│
├── ProbTS/                <- Initally used ProbTS, but switched to BasicTS due to more models/datasets
│
├── notebooks/             <- Jupyter notebooks used in this project
    ├── Clustering/        <- notebooks used to detect multi-world scenarios in the BasicTS+ data sets
    │
    ├── Evaluation/        <- Evaluation of final models
    │
    ├── SyntheticTS/       <- Initial exploration of creating synthetic time series
    │
    ├── multi_worls/       <- Our synthetic multi-world experiment
    │
    ├── old/               <- Old notebooks not relevant anymore
    │
    ├── data.ipynb         <- Analysis of the data
    │
    └── weights.yml        <- file that tracks the paths to all final model runs
│
├── notes/                 <- Generated analysis as .md and PDF
│
└── figures/               <- Figures used in report
```

## Our additions to `BasicTS`

- `BasicTS/baselines`: modified the architectures for PatchTST, DLinear, DeepAR and iTransformer to support proabilistic heads. Each is found under the respective `model_name/arch/model_name_arch.py`
- `BasicTS/basicts/data`: added the `one_step_tsf_dataset.py` training set to support the correct training of IMS models.
- `BasicTS/basicts/metrics`: added the `prob_metrics.py` and `probts.py` files, which enable the evaluation of our models, later on, however, I instead used the metrics in the respective evaluation notebooks (`notebooks/Evaluation`). Moreover, the metrics in these notebooks (e.g. `evaluation_ETTm1.ipynb` under 'eval') are GPU optimized versions of the metrics provided by [scoringrules](https://frazane.github.io/scoringrules/).
- `BasicTS/basicts/runners`: implemented the probabilistic runner (`runner_zoo/simple_prob_tsf_runner.py`), which takes a few extra steps/arguments depending on the probabilistic head used.
- `BasicTS/basicts/scaler`: made changes to the `inverse_transform` of each scaler to support probabilistic distributional heads.

## `notebooks/Evaluation`

Contains the scripts and notebooks used to qualitatively and quantitatively assess our final models.

## `notebooks/multi_world`

Contains all details for the synthetic multi-world experiment. The final results were created using the `KLL.ipynb` file, which accesses the respective .py files on the same directory level.

## `figures/`

Here, all figures used in the report are displayed as PDF files.

## `notes/`

Here, all previously important information is gathered. For instane, my seminar report on LTSF or the master thesis proposal.

