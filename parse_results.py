import subprocess
from datetime import datetime

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
print("file_name;epoch_index;hr_accuracy;train_time;val_time")
for file_name in file_list:
    process_handle = subprocess.run(["tail", "-n", "2", "input_files/" + file_name], capture_output=True)
    lines = process_handle.stdout.decode("utf-8")[:-1].split("\n")

    start = datetime.strptime(lines[1][-22:], "%Y-%m-%d %H:%M:%S %p")
    end = datetime.strptime(lines[0][-22:], "%Y-%m-%d %H:%M:%S %p")
    delta = end - start

    process_handle = subprocess.run(["grep", "_time", "input_files/" + file_name], capture_output=True)
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
                train_time = value.split("=")[1].strip()
            elif value.startswith("val_time"):
                val_time = value.split("=")[1].strip()

        results.append([file_name, epoch_index, hr_accuracy, train_time, val_time])
        elapsed_dict[file_name] = [delta, delta.seconds]

for result in results:
    if results_dict.get(result[0]):
        results_dict[result[0]].append(result[1:])
    else:
        results_dict[result[0]] = [result[1:]]

# print("file_name;delta;delta.seconds")
# for key, value in elapsed_dict.items():
#     print("%s;%s;%d" % (key, value[0], value[1]))

for key, values in results_dict.items():
    hr_accuracy_list = []
    train_time_list = []
    val_time_list = []
    for value in values:
        hr_accuracy_list.append(value[1])
        train_time_list.append(value[2])
        val_time_list.append(value[3])

    print(key,hr_accuracy_list ,train_time_list, val_time_list)
