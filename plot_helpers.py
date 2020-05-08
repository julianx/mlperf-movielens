import seaborn as sns
import matplotlib.pyplot as plt


# TODO: Remove the default x
def create_lineplot(df, y: str, output: str, title: str, y_label: str, label: str, x: str = "epochs"):
    sns.lineplot(
        x=x, y=y, data=df, estimator=None, lw=1, sort=True, dashes=False, label=label
    )

    # Set title
    plt.title(title)

    # Set x-axis label
    plt.xlabel("epochs")

    # Set y-axis label
    plt.ylabel(y_label)

    plt.savefig(output)
