run_commands = []
runs_per_config = 2
run_configurations = [
    # {"nodes": 1, "wall_time": "02:00:00", "n_gpu": 16, "cpu_cores": 16, "n_cpu": 1},
    # {"nodes": 1, "wall_time": "02:00:00", "n_gpu": 8, "cpu_cores": 8, "n_cpu": 1},
    # {"nodes": 1, "wall_time": "02:00:00", "n_gpu": 4, "cpu_cores": 4, "n_cpu": 1},
    {"nodes": 1, "wall_time": "02:00:00", "n_gpu": 2, "cpu_cores": 2, "n_cpu": 1},
    # {"nodes": 1, "wall_time": "02:00:00", "n_gpu": 1, "cpu_cores": 1, "n_cpu": 1},
]

batch_size = 208
num_epochs = 72
for run_configuration in run_configurations:
    nodes, wall_time, n_gpu, cpu_cores, n_cpu = run_configuration.get('nodes'), \
                                                run_configuration.get('wall_time'), \
                                                run_configuration.get('n_gpu'), \
                                                run_configuration.get('cpu_cores'), \
                                                run_configuration.get('n_cpu')

    file_name = "job_{n_gpu}GPUs".format(n_gpu=n_gpu)
    batch_job = """#!/bin/bash
#SBATCH -N {nodes}
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:volta32:{n_gpu}
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=EGRESS
#SBATCH --reservation=dgx2
#SBATCH --time={wall_time}

# interact -N {nodes} -p GPU-AI --gres=gpu:volta32:{n_gpu} -n 48 --egress -R dgx2 -t {wall_time}

LOCAL=/local/imagenet
cd $LOCAL/
# time rsync -a /pylon5/datasets/community/imagenet/processed_mxnet* . --progress
# mv processed_mxnet data
# git clone git@github.com:mlperf/training_results_v0.6.git
# time rsync -a $SCRATCH/mlperf/imagenet/production*.sif . --progress
cd $LOCAL/training_results_v0.6/NVIDIA/benchmarks/resnet/implementations/mxnet/

echo '#!/bin/bash
echo NODES: '{nodes}' WALLTIME: '{wall_time}' NGPU: '{n_gpu}' CORES: '{cpu_cores}' CPUS '{n_cpu}' BATCHSIZE: '{batch_size}' NUMEPOCHS: '{num_epochs}'

export CONT="<docker/registry>/mlperf-nvidia:image_classification" 
export DATADIR=$LOCAL/data/
export LOGDIR=$LOCAL/log_dir 
export DGXSYSTEM=imagenet 

## DL params
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="'{batch_size}'"
export KVSTORE="horovod"
export LR="10"
export WARMUP_EPOCHS="5"
export EVAL_OFFSET="3" # Targeting epoch no. 60
export EVAL_PERIOD="4"
export WD="0.0002"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS="'{num_epochs}'"

export NETWORK="resnet-v1b-normconv-fl"
export MXNET_CUDNN_SUPPLY_NORMCONV_CONSTANTS=1

export DALI_PREFETCH_QUEUE="5"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="0"
export DALI_ROI_DECODE="1"  #needs to be set to 1 as default and proof perf uplift

## Environment variables for multi node runs
## TODO: These are settings for large scale runs that
## may need to be adjusted for single node.
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=20
export NHWC_BATCHNORM_LAUNCH_MARGIN=32
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=25

## System run parms
export DGXNNODES='{nodes}'
export DGXSYSTEM=imagenet
export WALLTIME='{wall_time}'

## System config params
export DGXNGPU='{n_gpu}'
export DGXSOCKETCORES='{cpu_cores}'
export DGXHT=2         # HT is on is 2, HT off is 1
export DGXIBDEVICES=""' > config_imagenet.sh

cp ompi_bind_DGX2.sh ompi_bind_imagenet.sh
export DGXSYSTEM=imagenet
source /etc/profile.d/modules.sh
module unload intel
module load singularity

singularity exec --nv -B $LOCAL/data:/data $LOCAL/production_root_mxnet_19.07-py3.sif ./run_and_time.sh 
""".format(nodes=nodes, wall_time=wall_time, n_gpu=n_gpu, cpu_cores=cpu_cores, n_cpu=n_cpu, file_name=file_name,
           batch_size=batch_size, num_epochs=num_epochs)

    with open(file_name + ".imagenet.sbatch.sh", "w") as file:
        file.write(batch_job)
    sbatch_path = "/opt/packages/slurm/default/bin/"
    run_commands.append(sbatch_path + "sbatch {file_name}.imagenet.sbatch.sh".format(file_name=file_name))

results_dict = {}
for run_command in run_commands:
    for run_number in range(runs_per_config):
        # process_handle = subprocess.run(run_command.split(" "), capture_output=True)
        # run_output = process_handle.stdout.decode("utf-8")[:-1].split("\n")
        print(run_command)

        # file_name = run_command.split(" ")[1]
        #
        # with open(file_name, "w+") as file:
        #     content = json.dumps(results_dict)
        #     file.write(content)

print('pidstat -C "python|singularity" -h -d -r -u -U julian 5 >> '
      '/pylon5/pscstaff/julian/mlperf/results/2020-05-01/julian.cpu.log &')
