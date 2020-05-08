import datetime as dt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from plot_helpers import create_lineplot


times = []
gpu_percs = []
gpus = []
pids = []

with open("results/gpu.log") as f:
    f.readline()  # drop first line

    for line in f.readlines():
        if not line.strip():  # ignore blanks
            continue
        (
            timestamp,
            index,
            usage,
            memory,
            pstate,
            temp_gpu,
            mem_used,
            mem_free
        ) = line.strip().split(",")
        print(timestamp)
        times.append(dt.datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S.%f"))

        gpus.append(int(index))
        gpu_percs.append(float(usage.strip(" %")))


sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.set(rc={"figure.figsize": (11.7, 8.27)})

df = pd.DataFrame()

df["gpus"] = gpus
df["gpu_percs"] = gpu_percs
df["times"] = times

print(len(set(gpus)))

for gpu in set(gpus):
    create_lineplot(
        df.query(f"gpus=={gpu}").groupby("times", as_index=False).mean(),
        y="gpu_percs",
        output=f"gpu_graphs/gpu_graph_{gpu}.png",
        title="CPU usage",
        y_label="gpu",
        label=gpu,
        x="times",
    )

    plt.clf()
