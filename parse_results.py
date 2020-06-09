import subprocess
from datetime import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 05-01
# file_list = [
#     "slurm-8919210.out",
#     "slurm-8919280.out",
#     "slurm-8919307.out",
#     "slurm-8919368.out",
#     "slurm-8919402.out",
#     "slurm-8919498.out",
#     "slurm-8919503.out",
#     "slurm-8919593.out",
#     "slurm-8919594.out",
#     "slurm-8919614.out",
#     "slurm-8919617.out",
#     "slurm-8919630.out",
#     "slurm-8919631.out",
#     "slurm-8919632.out",
#     "slurm-8919640.out",
#     "slurm-8919641.out",
#     "slurm-8919642.out",
#     "slurm-8919643.out",
#     "slurm-8919644.out",
#     "slurm-8919647.out",
#     "slurm-8919649.out",
#     "slurm-8919650.out",
#     "slurm-8919651.out",
#     "slurm-8919652.out",
#     "slurm-8920069.out",
# ]

# 05-06
# file_list = [
#     "slurm-8979280.out",
#     "slurm-8979281.out",
#     "slurm-8979282.out",
#     "slurm-8979283.out",
#     "slurm-8979284.out",
#     "slurm-8979285.out",
#     "slurm-8979286.out",
#     "slurm-8979287.out",
#     "slurm-8979288.out",
#     "slurm-8979289.out"
# ]

# 05-12
# file_list = [
#     "slurm-9091180.out",
#     "slurm-9091181.out",
#     "slurm-9091182.out",
#     "slurm-9091183.out",
#     "slurm-9091184.out",
#     "slurm-9091185.out",
#     "slurm-9091186.out",
#     "slurm-9091187.out",
#     "slurm-9091188.out",
#     "slurm-9091189.out",
# ]

# ls -1 slurm*out
file_list = [
    "slurm-9136383.out",
    "slurm-9136385.out",
    "slurm-9136394.out",
    "slurm-9136405.out",
    "slurm-9136406.out",
    "slurm-9136418.out",
    "slurm-9136419.out",
    "slurm-9136421.out",
    "slurm-9136431.out",
    "slurm-9136432.out",
    "slurm-9136435.out",
    "slurm-9136445.out",
    "slurm-9136446.out",
    "slurm-9136447.out",
    "slurm-9136448.out",
    "slurm-9136457.out",
    "slurm-9136458.out",
    "slurm-9136459.out",
    "slurm-9136462.out",
    "slurm-9136473.out",
    "slurm-9136474.out",
    "slurm-9136477.out",
    "slurm-9136478.out",
    "slurm-9136487.out",
    "slurm-9136488.out",
]

results = []
results_dict = {}
elapsed_dict = {}

for file_name in file_list:
    base_path = "input_files/2020-05-16-ml" + "/"
    with open(f"{base_path}/{file_name}") as f:
        ngpu = int(f.readline().split()[5])

    process_handle = subprocess.run(
        ["tail", "-n", "2", base_path + file_name], capture_output=True
    )
    lines = process_handle.stdout.decode("utf-8")[:-1].split("\n")

    start = datetime.strptime(lines[1][-22:], "%Y-%m-%d %H:%M:%S %p")
    end = datetime.strptime(lines[0][-22:], "%Y-%m-%d %H:%M:%S %p")
    delta = end - start

    process_handle = subprocess.run(
        ["grep", "_time", base_path + file_name], capture_output=True
    )
    epochs = process_handle.stdout.decode("utf-8")[:-1].split("\n")
    # Epoch 0: HR@10 = 0.4211, NDCG@10 = 0.2385, train_time = 2.64, val_time = 0.15
    for epoch in epochs:
        epoch_values = epoch.split(",")
        for value in epoch_values:
            value = value.strip()
            if value.startswith("Epoch "):
                epoch_index, hr_accuracy = value.split(":")
                epoch_index = epoch_index.split(" ")[1].strip()
                hr_accuracy = hr_accuracy.split("=")[1].strip()
            elif value.startswith("train_time"):
                train_time = float(value.split("=")[1].strip())
            elif value.startswith("val_time"):
                val_time = float(value.split("=")[1].strip())

        results.append(
            [
                file_name,
                int(epoch_index),
                float(hr_accuracy),
                train_time,
                val_time,
                ngpu,
            ]
        )
        elapsed_dict[file_name] = [delta, delta.seconds/60]

# for i in slurm*out; do grep  eval_accuracy ${i} | tail -n 1; done
for key, value in elapsed_dict.items():
    print(key, value)

for result in results:
    if results_dict.get(result[0]):
        results_dict[result[0]].append(result[1:])
    else:
        results_dict[result[0]] = [result[1:]]

sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")


counter = 0
for key, values in results_dict.items():
    epochs_list = []
    hr_accuracy_list = []
    train_time_list = []
    val_time_list = []
    ngpu_list = []

    keys_list = []
    for value in values:
        epochs_list.append(value[0])
        hr_accuracy_list.append(value[1])

        train_time_list.append(value[2])
        val_time_list.append(value[3])

        ngpu_list.append(value[-1])

        keys_list.append(key)

    # print(key, len(epochs_list), hr_accuracy_list[-1], train_time_list[-1], val_time_list[-1])
#
# sns.set_context("notebook", font_scale=1.1)
# sns.set_style("ticks")
# sns.set(rc={"figure.figsize": (11.7, 8.27)})
#
# df = pd.DataFrame()
#
# # Add columns
# df["epochs"] = epochs_list
# df["train_time"] = train_time_list
# df["accuracy"] = hr_accuracy_list
# df["valid_time"] = val_time_list
# df["ngpu"] = ngpu_list
# df["keys"] = keys_list
#
#
# def create_lineplot(df, y: str, output: str, title: str, y_label: str, label: str):
#     if label == 1:
#         label = str(label) + " GPU"
#     else:
#         label = str(label) + " GPUs"
#
#     sns.lineplot(
#         x="epochs", y=y, data=df, estimator=None, lw=1, sort=True, dashes=False, label=label
#     )
#
#     # Set title
#     plt.title(title)
#
#     # Set x-axis label
#     plt.xlabel("Epochs")
#
#     # Set y-axis label
#     plt.ylabel(y_label)
#
#     plt.savefig(output)
#
#
# # plot the training and validation times across epochs.
# for ngpu in set(ngpu_list):
#     create_lineplot(
#         df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).mean(),
#         y="train_time",
#         output="1_training_epochs.png",
#         title="Training time across epochs",
#         y_label="Time (secs)",
#         label=ngpu
#     )
# plt.clf()
# for ngpu in set(ngpu_list):
#     create_lineplot(
#         df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).mean(),
#         y="valid_time",
#         output="1_valid_epochs.png",
#         title="Validation time across epochs",
#         y_label="Time (secs)",
#         label=ngpu
#     )
# plt.clf()
# # TODO plot the training and validation times sum across runs.
# for ngpu in set(ngpu_list):
#     create_lineplot(
#         df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).sum(),
#         y="train_time",
#         output="2_time_across_runs.png",
#         title="Training time across runs",
#         y_label="Time (secs)",
#         label=ngpu
#     )
# plt.clf()
# for ngpu in set(ngpu_list):
#     create_lineplot(
#         df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).sum(),
#         y="valid_time",
#         output="2_valid_across_runs.png",
#         title="Valid time across runs",
#         y_label="Time (secs)",
#         label=ngpu
#     )
#
# plt.clf()
# # plot progression of accuracy across epochs and across GPU configurations.
# for ngpu in set(ngpu_list):
#     create_lineplot(
#         df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).mean(),
#         y="accuracy",
#         output="3_accuracy.png",
#         title="Accuracy across GPU configurations.",
#         y_label="Accuracy",
#         label=ngpu
#     )
#
# plt.clf()
