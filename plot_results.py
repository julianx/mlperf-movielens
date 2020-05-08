import datetime as dt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from plot_helpers import create_lineplot


times = []
cpu_percs = []
cpus = []
pids = []

with open("results/cpu.log") as f:
    f.readline()  # drop first line

    for line in f.readlines():
        if not line.strip():  # ignore blanks
            continue

        if line.lstrip().startswith("#") or line.lstrip().startswith("Linux"):
            continue

        (
            time,
            am_pm,
            user,
            pid,
            usr_p,
            system_perc,
            guest_perc,
            wait,
            cpu_p,
            cpu,
            minflt,
            majflt,
            vsz,
            rss,
            mem_p,
            kb_rd,
            kb_wr,
            kb_cc,
            iodelay,
            command,
        ) = line.strip().split()

        now = dt.datetime.now()

        # TODO: Do it with date parse, maybe?
        hour, minute, second = time.split(":")
        times.append(
            dt.datetime(
                year=now.year,
                month=now.month,
                day=now.day,
                hour=int(hour) + (0 if am_pm == "AM" else 12),
                minute=int(minute),
                second=int(second),
            )
        )
        cpus.append(int(cpu))
        cpu_percs.append(float(cpu_p))


sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.set(rc={"figure.figsize": (11.7, 8.27)})

df = pd.DataFrame()

df["cpus"] = cpus
df["cpu_percs"] = cpu_percs
df["times"] = times

print(len(set(cpus)))

for cpu in set(cpus):
    create_lineplot(
        df.query(f"cpus=={cpu}").groupby("times", as_index=False).mean(),
        y="cpu_percs",
        output=f"cpu_graphs/cpu_graph_{cpu}.png",
        title="CPU usage",
        y_label="cpu",
        label=cpu,
        x="times",
    )

    plt.clf()
