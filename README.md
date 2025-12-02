# Trailblazer

![Trailblazer](./images/trailblazer.png)

Hi! This is the official repository of *Trailblazer* paper "*Large Language Models as Generalist Policies for 
Network Optimization*".

**Abstract**: Designing control policies to ensure robust network services is essential to modern digital infrastructure. However, the dominant paradigm for network optimization relies on designing specialist policies based on handcrafted rules or deep learning models, leading to poor generalization across diverse tasks and environments. In contrast, large language models (LLMs), pretrained on Internet-scale corpora, provide a rich and unified knowledge base that encodes fundamental networking principles. Combined with their emergent abilities in generalization to unseen scenarios, LLMs offer a transformative foundation for generalist network policies that can generalize across diverse tasks and environments with minimal adaptation. In this paper, we present Trailblazer, the first systematic framework to realize such a generalist policy for networking. Trailblazer incorporates a network alignment scheme to ground the LLM in specific networking tasks, and an adaptive policy collaboration mechanism that offloads simple control cases from the LLM to a lightweight policy for computational efficiency. Through extensive simulations and large-scale real-world online evaluation on Douyin (the Chinese version of TikTok), Trailblazer, powered by a single LLM, demonstrates stronger cross-task and cross-environment generalization than conventional specialist policies. Our results validate LLMs as the foundation for generalist network policies, and position Trailblazer as the first step toward the generalist-driven paradigm that enables strong generalization with minimal efforts in policy design.

**What is Trailblazer?** 😄

Trailblazer is the pioneering framework that grounds LLMs as generalist policies for network optimization. 

Through extensive experiments on two representative networking tasks of broad social and industrial importance-adaptive bitrate streaming (ABR) and cluster job scheduling (CJS)-we show that Trailblazer powered by a single LLM significantly outperforms state-of-the-art baselines, demonstrating stronger cross-task and cross-environment generalization. 

To validate its real-world applicability, we deploy Trailblazer in Douyin’s congestion control (CC) service for large-scale online A/B tests for three weeks, serving 150,000+ users across 100+ cities and accumulating over 1,200 days of video playback time. Results show that Trailblazer outperformed VICC, a strong and mature baseline currently used by Douyin, across all key industrial performance metrics. 

> *Trailblazer (开拓者)* signifies our goal of forging the first path in LLM-driven generalist network policies, establishing a framework for both academia and industry to advance the integration of LLMs into real-world network services. 😊

**What does this repo provide?** 😄

This repository contains the implementation of Trailblazer for the ABR and CJS tasks. Please note that due to Douyin’s data security requirements, we are unable to release the code and data for the CC task deployed in production.

# Adaptive Bitrate Streaming
## Preface
The codes for adaptive bitrate streaming (ABR) are implemented based on the repository of [Genet](https://github.com/GenetProject/Genet/tree/main). Thanks for Genet's authors for their open source codes!

What is ABR?
> The volume of video streaming has reached almost 60% of all the Internet traffic. Streaming video over variable-bandwidth networks (e.g., cellular network) requires the client to adapt the video bitrate to optimize the user experience. In industrial DASH standard, videos are
divided into multiple chunks, each of which represents a few seconds of the overall video playback. Each chunk is encoded at several discrete bitrates, where a higher bitrate implies a higher resolution and thus a larger chunk size. For this problem, each MDP episode is a video playback with a particular network trace (i.e., a time series of network throughput). At each step, the agent observes the past network throughput measurement, the current video buffer size, and the remaining portion of the video. The action is the bitrate for the next video chunk. The objective is to maximize the video resolution and minimize the stall (which occurs when download time of a chunk is larger than the current buffer size) and the reward
is structured to be a linear combination of selected bitrate and the stall when downloading the corresponding chunk.

## Code Structure
The codes for ABR are stored in `adaptive_bitrate_streaming`.

- `artifacts`: This directory stores some artifacts, e.g., result files.
   - `exp_pool`: This directory stores the experience pool files, which will be used for LLM adaptation.
   - `results`: This directory stores the result files.

- `data`: This directory stores datasets and pre-trained model checkpoints of baselines.
   - `traces`: This directory stores the bandwidth trace datasets.
   - `videos`: This directory stores the video specifications.
   - `ft_plms`: This directory stores the fine-tuned (adapted) LLMs.
   - `all_models`: This directory stores the model checkpoints of baselines.

- `baseline_specical`: This directory stores the codes for runing baselines. Most of the codes are from the Genet's repository.
- `plm_special`: This directory stores the codes for running Trailblazer.
   - `data`: This directory stores the codes related to the training datasets for LLM adaptation.
      - `exp_pool.py`: Implements the experience pool for collecting trajectories.
      - `dataset.py`: Implements a dataset class that wraps the experience pool.
    - `models`: This directory stores the codes related to Trailblazer.
      - `state_encoder.py`: Implements the feature encoder for encoding states.
      - `gpt2.py, llama.py, opt.py, mistral.py, t5.py`: Customized LLMs.
      - `low_rank.py`: Implements the low rank matrices.
      - `rl_policy.py`: Implements the Transformer-based offline RL policy.
    - `utils`: This directory stores some utilization codes.
      - `plm_utils.py`: Some codes for loading LLMs.
      - `utils.py`: Some codes for data processing.
    - `trainer.py`: Some codes for training (adapting) LLMs. 
    - `evaluate.py`: Some codes for evaluting the performance of adapted-LLMs.
    - `test.py`: Some codes for testing the performance of adapted LLMs.
- `generate_exp_pool.py`: Implements the generation of experience pool (i.e., training dataset for LLM).
- `run_baseline.py`: The main file for running baselines. 
- `run_plm.py`: The main file for running Trailblazer.

## Environment Setup
### Environment for Trailblazer
1. Create a conda environment for Trailblazer:

   `conda create -n abr_trailblazer python>=3.8.10`

2. Then install the following depdendencies:

   ```
   python==3.8.10
   torch==2.1.0
   numpy==1.24.4
   munch==4.0.0
   openprompt==1.0.1
   transformers==4.34.1
   peft==0.6.2
   ```

### Environment for baselines
To run baselines, we need a different environment, since they are mainly written in tensforflow v1.

1. First, create a conda environment with `python=3.7`. Please note that you must install `python=3.7`, since the greater versions of python do not support installing tensorflow 1.x any more.

   `conda create -n abr_tf python=3.7`

2. Next, install the following dependencies.
   ```sh
   conda activate abr_tf
   pip install tensorflow-gpu==1.15
   pip install tensorboard==1.15.0
   pip install tensorboard-plugin-wit==1.8.0
   pip install tflearn==0.5.0
   pip install numba==0.53.1
   pip install gym==0.18.0
   pip install stable-baselines[mpi]==2.10.1
   pip install pandas==1.1.5
   pip install tqdm==4.62.2
   ```
## Usage
## Usage of Trailblazer
To run Trailblazer, first we need to download some LLMs. For example, if you want to use Llama2-7b as the foundation model, please download Llama2-7b in the directory: `../downloaded_plms/llama2/base`. In the following, we will use the Llama2-7b as the example to illustrate the usage of Trailblazer.

**Finetune LLM**

If you want to finetune LLM, please run the following command:
```sh
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 
```
This command will finetune Llama2 on the default experience pool we provided at `artifacts/exp_pools/exp_pool.pkl`.
If you want to use your own experience pool, first use the `generate_exp_pool.py` to generate a new experience pool.
```sh
conda activate abr_tf  # since we need to use baselines to interact with environments, we need to activate the baseline environment first.
python generate_exp_pool.py --models genet --trace fcc-train --video video1 --trace-num 100 --cuda-id 0
```
Next, specify the path to your own experience pool with argument `--exp-pool-path` and run the following command:
```sh
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2--exp-pool-path your_exp_pool_path
```

**Test LLM**

If you want to test the performance of the finetuned LLM, please run the following command:
```sh
python run_plm.py --test --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2
```
You can also specify the path to the finetuned LLM with argument `--model-dir`:
```sh
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir you_finetune_llm_dir
```

### Run baselines

To run baselines, please run:
```sh
conda activate abr_tf
python run_baseline.py --model genet --cuda-id 0
python run_baseline.py --model mpc 
python run_baseline.py --model bba 
```

# Cluster Job Scheduling
## Preface

The codes for cluster job scheduling (CJS) are implemented based on the repository of  [spark-sched-sim](https://github.com/ArchieGertsman/spark-sched-sim). Thanks for spark-sched-sim's authors for their open source codes! 

What is CJS? 

> Cluster job scheduling (CJS) plays a critical role in optimizing the allocation of computational resources in distributed computing environments, where multiple jobs need to be processed simultaneously. It usually need a policy to schedule incoming jobs within the cluster. Each job is represented as a directed acyclic graph (DAG), which describes the dependencies between each execution stage and the resource requirements of each stage. The primary task of the policy is to select the next stage of a job to execute and allocate a set of executors (computing resources) to that stage. The objective is to minimize the average job completion time, thereby optimizing the system-level utilization of computing resources within the cluster.

## Code structure
The codes for CJS are stored in `cluster_job_scheduling`.

- `artifacts`: Stores some artifacts, e.g., result files.
    - `exp_pool`: Stores the experience pool files, which will be used for LLM adaptation.
    - `results`: Stores the result files.
- `stdout`: Output information generated when training baselines. (not necessary)
- `checkpoints`: Output information generated when training baselines. (not necessary)
- `config`: Stores configuration files for baselines, which are used during the running/training of baselines.
- `data`: Stores datasets. Note that this dataset is not the same as the experience pool. It is used to generate the simulation environment.
- `models`: Stores pre-trained model weights for baselines. 

- `spark_sched_sim`: Code directory for implementing the simulation environment, which is from [spark-sched-sim](https://github.com/ArchieGertsman/spark-sched-sim).
- `trainers`: Files related to training baselines.

- `plm_special`: Stores core codes for running Trailblazer on cluster job scheduling.
    - `data`: Stores codes related to the training dataset.
      - `exp_pool.py`: Implements the experience pool for collecting trajectories.
      - `dataset.py`: Implements a dataset class that wraps the experience pool.
    - `models`: Stores codes related to models.
      - `gpt2.py, llama.py, t5.py`: Customized LLMs.
      - `low_rank.py`: Implements the low rank matrices.
      - `rl_policy.py`: Implements the Transformer-based offline RL policy.
      - `state_encoder.py`: Implements the feature encoder for encoding DAGs.
    - `utils`: Stores some utilization codes.
      - `plm_utils.py`: Some codes for loading LLMs.
      - `utils.py`: Some codes for data processing.
    - `trainer.py`: Implements a wrapper class for the training process.
    - `evaluate.py`: Some codes for validation process.
    - `test.py`: Some codes for testing process.

- `cfg_loader.py`: Some codes for loading configuration files for baselines from the `config` directory.
- `train_baseline.py`: A file provided by [spark-sched-sim](https://github.com/ArchieGertsman/spark-sched-sim) for training baselines.

- `generate_exp_pool.py`: Implements the generation of experience pool (i.e., training dataset for LLM).
- `run_baseline.py`: The main file for running baselines. 
- `run_plm.py`: The main file for running Trailblazer.

## Environment Setup

1. Create a conda environment with `python=3.11.9` and activate it. Other versions of python might be okay as well.

   ```sh
   conda create -n cjs_trailblazer python==3.11.9 -y
   conda activate cjs_trailblazer
   ```

2. Install the following dependencies (the package versions are provided just for reference): 
   ```shell
   # For Pytorch
   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
   
   # For PyG
   pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
   pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
   pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
   pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
   pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric
   
   # For Gymnasium
   conda install swig -y
   pip install "Gymnasium[all]"
   
   # For other packages
   pip install numpy==1.26.4
   pip install transformers==4.37.1
   pip install munch==4.0.0
   pip install openprompt==1.0.1
   pip install peft==0.13.2
   ```

## Usage

### Usage of Trailblazer
To run Trailblazer, first we need to download the pretrained weights of some LLMs. For example, if you want to use Llama2-7b as the foundation model, please download Llama2-7b in the directory: `../downloaded_plms/llama2/base`. In the following, we will use the Llama2-7b as the example to illustrate the usage of Trailblazer.

**Finetune LLM**

- If you want to finetune LLM, you can run the following command:

    ```sh
    python run_plm.py \
        --train \
        --test \
        --seed 666 \
        --plm-type llama \
        --plm-size base \
        --peft-rank 128 \
        --device cuda:0 \
        --device-out cuda:0 \
        --state-feature-dim 256 \
        --K 20 \
        --gamma 1.0 \
        --lr 0.0001 \
        --num-iters 40 \
        --freeze-encoder
    ```
    
    This command will finetune Llama2 on the default experience pool we provided at `artifacts/exp_pools/exp_pool.pkl`.
    
- If you want to use your own experience pool, first use the `generate_exp_pool.py` to generate a new experience pool.

    ```sh
    python generate_exp_pool.py \
        --scheds decima \
        --pool-size 1000 \
        --complete-episode \
        --num-executors 50 \
        --job-arrival-cap 200 \
        --job-arrival-rate 4e-5 \
        --moving-delay 2000 \
        --warmup-delay 1000 \
        --render-mode None \
        --dataset tpch \
        --seed 1 \
        --device cuda:0
    ```

    Next, specify the path to your own experience pool with argument `--exp-pool-path` and run the following command:

    ```sh
    python run_plm.py \
        --train \
        --test \
        --seed 666 \
        --plm-type llama \
        --plm-size base \
        --peft-rank 128 \
        --device cuda:0 \
        --device-out cuda:0 \
        --state-feature-dim 256 \
        --K 20 \
        --gamma 1.0 \
        --lr 0.0001 \
        --num-iters 40 \
        --freeze-encoder \
        --exp-pool-path "your_exp_pool_path"
    ```

**Test LLM**

If you want to test the performance of the finetuned LLM, please run the following command:
```sh
python run_plm.py \
    --test \
    --plm-type llama \
    --plm-size base \
    --peft-rank 128 \
    --state-feature-dim 256 \
    --device cuda:0
```
You can also specify the path to the finetuned LLM with argument `--model-dir`:
```sh
python run_plm.py \
    --test \
    --plm-type llama \
    --plm-size base \
    --peft-rank 128 \
    --state-feature-dim 256 \
    --device cuda:0 \
    --model-dir you_finetune_llm_dir
```

### Run baselines

To run baselines, you can run the following command:
```sh
python run_baseline.py \
    --sched decima \
    --num-executors 50 \
    --job-arrival-cap 200 \
    --seed 666 \
    --device cuda:0

python run_baseline.py \
    --sched fair \
    --num-executors 50 \
    --job-arrival-cap 200 \
    --seed 666

python run_baseline.py \
    --sched fifo \
    --num-executors 50 \
    --job-arrival-cap 200 \
    --seed 666
```


