## NeuronX Nemo Megatron on Slurm with Trn1

## 1. Bulid Custom Neuron AMI for ParallelCluster

```
packer build -only 'aws-pcluster-neuron.*' -var parallel_cluster_version=3.6.1 -var aws_region=us-west-2 -var "ami_version=1" packer-ami.pkr.hcl | tee aws-pcluster-neuron_ami.log
```

you will see 


```

==> Wait completed after 13 minutes 4 seconds

==> Builds finished. The artifacts of successful builds are:
--> aws-pcluster-neuron.amazon-ebs.aws-pcluster-ami: AMIs were created:
us-west-2: ami-0f5ed52f73351d999
```

once AMI creation completed.

## 3. Setup infrastrcutre

```
export REGION=us-west-2
export AZ=us-west-2d
VPC_NAME=vpc-neuronx-nemomegatron
aws cloudformation create-stack --stack-name ${VPC_NAME} \
    --template-body file://../1.architectures/1.vpc_network/2.vpc-one-az.yaml \
	--parameters ParameterKey=VPCName,ParameterValue=${VPC_NAME} ParameterKey=SubnetsAZ,ParameterValue=${AZ} \
	--region ${REGION} --capabilities=CAPABILITY_IAM
```



## 2. Launch ParallelCluster with AMI

Fill out the following configs
```
REGION=us-west-2
AZ=us-west-2d
PLACEHOLDER_CUSTOM_AMI_ID=ami-0ef5eac7c4cd4b22e
PLACEHOLDER_PUBLIC_SUBNET=subnet-0a910e572266c13bd
PLACEHOLDER_PRIVATE_SUBNET=subnet-031418ea04ed99194
PLACEHOLDER_SSH_KEY=dev-machine-us-west-2
PLACEHOLDER_CAPACITY_RESERVATION_ID=cr-06d73238916b3c7a8
PLACEHOLDER_PLACEMENT_GROUP=trn1-placement-group
PLACEHOLDER_NUM_INSTANCES=32
```
in ``../1.architectures/2.aws-parallelcluster/distributed-training-trn1_custom_ami.yaml` and save as `pcluster.yaml`.

Then create trn1 cluster with:

```
pcluster create-cluster --cluster-configuration pcluster.yaml --cluster-name pcluster-neuronx-nemomegatron --region us-west-2
```


## 3. Connect to Cluster

```
pcluster ssh --dryrun true --cluster-name pcluster-neuronx-nemomegatron --region us-west-2 
```
{
  "command": "ssh ec2-user@xx.xx.xx.xx "
}

## Convert weights


## Tokenize dataset
This tutorial makes use of a Red pyjama dataset. The dataset can be downloaded to your cluster by running the following commands on the head node:

```
mkdir -p /fsx/data/llama2
wget https://data.together.xyz/redpajama-data-1T/v1.0.0/book/book.jsonl
or
wget https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/book_sample.jsonl -O /fsx/data/llama2/book.jsonl
```

Once you have the Tokenizer and the dataset. You can tokenize the dataset following the below command : 
```
sbatch 02-tokenize.sbatch
```

Use of the Llama 2 model is governed by the Meta license and must be downloaded and converted to the standard Hugging Face format prior to running this sample. Assuming you had downloaded the model and tokenizer as shown below.

```
/fsx/Llama2-meta/
├── 7B/
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── tokenizer.model
└── tokenizer_checklist.chk
```

To conver the model to the standard Hugging Face format, the following script in transformers can be called with the following (example) command:

```
sbatch 01-convert-weight.sbatch
```
Note: For the purposes of this sample we assume you have saved the Llama-2-7b model in a directory called `Llama2-7b-hf` with the following format:

```
Llama2-7b-hf/
├── config.json
├── generation_config.json
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00002-of-00002.bin
├── pytorch_model.bin.index.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer.model
└── tokenizer_config.json
```
Post tokenizing the dataset, you will have a path to the tokenizer and the dataset which will be used for pretraining. 

## Llama2 training configurations
We tested with the following model sizes: 7B
### Llama2 7B

- Model configuration
    - Attention heads: 32
    - Layers: 32
    - Sequence length: 4096
    - Hidden size: 4096
    - Hidden FFN size: 11008
    - Microbatch size: 1
    - Global batch size: 256

- Distributed training configuration
    - Number of nodes: 4
    - Tensor parallel degree: 8
    - Pipeline parallel degree: 1
    - Data parallel degree: 16

## Pre-compile the model
By default, PyTorch Neuron uses a just in time (JIT) compilation flow that sequentially compiles all of the neural network compute graphs as they are encountered during a training job. The compiled graphs are cached in a local compiler cache so that subsequent training jobs can leverage the compiled graphs and avoid compilation (so long as the graph signatures and Neuron version have not changed).

An alternative to the JIT flow is to use the included [neuron_parallel_compile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html?highlight=neuron_parallel_compile) command to perform ahead of time (AOT) compilation. In the AOT compilation flow, the compute graphs are first identified and extracted during a short simulated training run, and the extracted graphs are then compiled and cached using parallel compilation, which is considerably faster than the JIT flow.

Before starting the compilation you need to update your path to the dataset and tokenizer in the llama_7b script as below : 

```
vi llama_7b.sh

# Update the below lines
# For tokenizer
model.tokenizer.type='PATH_TO_LLAMA_TOKENIZER/llamav2_weights/7b-hf' \

# For Dataset
model.data.data_prefix=[1.0,PATH_TO_TOKENIZED_DATASET/books/book.jsonl-processed_text_document] \
```
Run the following commands to launch an AOT pre-compilation job on your ParallelCluster:
```
cd ~/neuronx-nemo-megatron/nemo/examples/nlp/language_modeling
sbatch --nodes 4 compile.slurm ./llama_7b.sh
```

Once you have launched the precompilation job, run the `squeue` command to view the SLURM job queue on your cluster. If you have not recently run a job on your cluster, it may take 4-5 minutes for the requested trn1.32xlarge nodes to be launched and initialized. Once the job is running, `squeue` should show output similar to the following:
```
    JOBID  PARTITION  NAME           USER    ST  TIME  NODES NODELIST(REASON)
    10     compute1   compile.slurm  ubuntu  R   5:11  4     compute1-dy-queue1-i1-[1-4]
```

You can view the output of the precompilation job by examining the file named `slurm-compile.slurm-ZZ.out` where ZZ represents the JOBID of your job in the `squeue` output, above. Ex:
```
tail -f slurm-compile.slurm-10.out
```

Once the precompilation job is complete, you should see a message similar to the following in the logs:
```
2023-06-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total graphs: 22
2023-06-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total successful compilations: 22
2023-06-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total failed compilations: 0
```

At this point, you can press `CTRL-C` to exit the tail command.

## Launch a pretraining job
The Llama2 pretraining job can be launched in the same manner as the precompilation job described above. In this case, we change the SLURM script from `compile.slurm` to `run.slurm`, but the other parameters remain the same:
```
cd ~/neuronx-nemo-megatron/nemo/examples/nlp/language_modeling
sbatch --nodes 4 run.slurm ./llama_7b.sh
```


As outlined above, you can again use the `squeue` command to view the job queue. Once you see that your pretraining job is running, you can view the output of the training job by examining the file named `slurm-run.slurm-ZZ.out` where ZZ represents the JOBID of your job:
```
tail -f slurm-run.slurm-11.out
```

Once the model is loaded onto the Trainium accelerators and training has commenced, you will begin to see output indicating the job progress:
```
Epoch 0:  22%|██▏       | 4499/20101 [22:26:14<77:48:37, 17.95s/it, loss=2.43, v_num=5563, reduced_train_loss=2.470, gradient_norm=0.121, parameter_norm=1864.0, global_step=4512.0, consumed_samples=1.16e+6, iteration_time=16.40]
Epoch 0:  22%|██▏       | 4500/20101 [22:26:32<77:48:18, 17.95s/it, loss=2.43, v_num=5563, reduced_train_loss=2.470, gradient_norm=0.121, parameter_norm=1864.0, global_step=4512.0, consumed_samples=1.16e+6, iteration_time=16.40]
Epoch 0:  22%|██▏       | 4500/20101 [22:26:32<77:48:18, 17.95s/it, loss=2.44, v_num=5563, reduced_train_loss=2.450, gradient_norm=0.120, parameter_norm=1864.0, global_step=4512.0, consumed_samples=1.16e+6, iteration_time=16.50]
```
## 4. Clean up

```
 pcluster delete-cluster  --cluster-name pcluster-neuronx-nemomegatron --region us-west-2      
```