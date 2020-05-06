import subprocess
from datetime import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


file_list = [
    "slurm-8919210.out",
    "slurm-8919280.out",
    "slurm-8919307.out",
    "slurm-8919368.out",
    "slurm-8919402.out",
    "slurm-8919498.out",
    "slurm-8919503.out",
    "slurm-8919593.out",
    "slurm-8919594.out",
    "slurm-8919614.out",
    "slurm-8919617.out",
    "slurm-8919630.out",
    "slurm-8919631.out",
    "slurm-8919632.out",
    "slurm-8919640.out",
    "slurm-8919641.out",
    "slurm-8919642.out",
    "slurm-8919643.out",
    "slurm-8919644.out",
    "slurm-8919647.out",
    "slurm-8919649.out",
    "slurm-8919650.out",
    "slurm-8919651.out",
    "slurm-8919652.out",
    "slurm-8920069.out",
]

results = []
results_dict = {}
elapsed_dict = {}

for file_name in file_list:
    with open(f"input_files/{file_name}") as f:
        ngpu = int(f.readline().split()[5])

    process_handle = subprocess.run(
        ["tail", "-n", "2", "input_files/" + file_name], capture_output=True
    )
    lines = process_handle.stdout.decode("utf-8")[:-1].split("\n")

    start = datetime.strptime(lines[1][-22:], "%Y-%m-%d %H:%M:%S %p")
    end = datetime.strptime(lines[0][-22:], "%Y-%m-%d %H:%M:%S %p")
    delta = end - start

    process_handle = subprocess.run(
        ["grep", "_time", "input_files/" + file_name], capture_output=True
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
        elapsed_dict[file_name] = [delta, delta.seconds]

for result in results:
    if results_dict.get(result[0]):
        results_dict[result[0]].append(result[1:])
    else:
        results_dict[result[0]] = [result[1:]]

sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

epochs_list = []
hr_accuracy_list = []
train_time_list = []
val_time_list = []
ngpu_list = []

keys_list = []

counter = 0
for key, values in results_dict.items():
    for value in values:
        epochs_list.append(value[0])
        hr_accuracy_list.append(value[1])

        train_time_list.append(value[2])
        val_time_list.append(value[3])

        ngpu_list.append(value[-1])

        keys_list.append(key)

sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.set(rc={"figure.figsize": (11.7, 8.27)})

df = pd.DataFrame()

# Add columns
df["epochs"] = epochs_list
df["train_time"] = train_time_list
df["accuracy"] = hr_accuracy_list
df["valid_time"] = val_time_list
df["ngpu"] = ngpu_list
df["keys"] = keys_list


def create_lineplot(df, y: str, output: str, title: str, y_label: str, label: str):
    sns.lineplot(
        x="epochs", y=y, data=df, estimator=None, lw=1, sort=True, dashes=False, label=label
    )

    # Set title
    plt.title(title)

    # Set x-axis label
    plt.xlabel("epochs")

    # Set y-axis label
    plt.ylabel(y_label)

    plt.savefig(output)

# plot the training and validation times across epochs.
for ngpu in set(ngpu_list):
    create_lineplot(
        df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).mean(),
        y="train_time",
        output="1_training_epochs.png",
        title="Training time accross epochs",
        y_label="time",
        label=ngpu
    )
plt.clf()
for ngpu in set(ngpu_list):
    create_lineplot(
        df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).mean(),
        y="valid_time",
        output="1_valid_epochs.png",
        title="Valid time accross epochs",
        y_label="time",
        label=ngpu
    )
plt.clf()
# TODO plot the training and validation times sum across runs.
for ngpu in set(ngpu_list):
    create_lineplot(
        df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).sum(),
        y="train_time",
        output="2_time_across_runs.png",
        title="Training time accross runs",
        y_label="time",
        label=ngpu
    )
plt.clf()
for ngpu in set(ngpu_list):
    create_lineplot(
        df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).sum(),
        y="valid_time",
        output="2_valid_across_runs.png",
        title="Valid time accross runs",
        y_label="time",
        label=ngpu
    )

plt.clf()
# plot progression of accuracy across epochs and across GPU configurations.
for ngpu in set(ngpu_list):
    create_lineplot(
        df.query(f"ngpu=={ngpu}").groupby("epochs", as_index=False).mean(),
        y="accuracy",
        output="3_accuracy.png",
        title="Accuracy accross gpu configurations.",
        y_label="accuracy",
        label=ngpu
    )

plt.clf()
