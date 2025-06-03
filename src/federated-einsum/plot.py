import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_horizontal = {
    "FedAvg [TabNet] (5 cl.)": {
        "Cancer": {"Acc": (0.92, 0.03), "F1": (0.92, 0.03)},
        "Credit": {"Acc": (0.71, 0.11), "F1": (0.48, 0.04)},
        "Income": {"Acc": (0.68, 0.06), "F1": (0.51, 0.03)},
    },
    "FedAvg [TabNet] (10 cl.)": {
        "Cancer": {"Acc": (0.92, 0.04), "F1": (0.91, 0.05)},
        "Credit": {"Acc": (0.56, 0.12), "F1": (0.47, 0.06)},
        "Income": {"Acc": (0.64, 0.06), "F1": (0.52, 0.03)},
    },
    "FedTree (5 cl.)": {
        "Cancer": {"Acc": (0.93, 0.01), "F1": (0.92, 0.01)},
        "Credit": {"Acc": (0.91, 0.01), "F1": (0.63, 0.01)},
        "Income": {"Acc": (0.88, 0.01), "F1": (0.82, 0.02)},
    },
    "FedTree (10 cl.)": {
        "Cancer": {"Acc": (0.94, 0.01), "F1": (0.93, 0.01)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.69, 0.01)},
        "Income": {"Acc": (0.87, 0.01), "F1": (0.80, 0.01)},
    },
    "FC [PC] (5 cl.)": {
        "Cancer": {"Acc": (0.98, 0.01), "F1": (0.98, 0.01)},
        "Credit": {"Acc": (0.93, 0.02), "F1": (0.68, 0.02)},
        "Income": {"Acc": (0.87, 0.02), "F1": (0.80, 0.01)},
    },
    "FC [PC] (10 cl.)": {
        "Cancer": {"Acc": (0.95, 0.02), "F1": (0.95, 0.02)},
        "Credit": {"Acc": (0.93, 0.01), "F1": (0.66, 0.02)},
        "Income": {"Acc": (0.87, 0.01), "F1": (0.80, 0.02)},
    },
    "FC [DT] (5 cl.)": {
        "Cancer": {"Acc": (0.95, 0.03), "F1": (0.93, 0.02)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.67, 0.01)},
        "Income": {"Acc": (0.89, 0.01), "F1": (0.83, 0.01)},
    },
    "FC [DT] (10 cl.)": {
        "Cancer": {"Acc": (0.95, 0.02), "F1": (0.93, 0.03)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.97, 0.02)},
        "Income": {"Acc": (0.89, 0.01), "F1": (0.83, 0.02)},
    },
    "SplitNN [TabNet]": {
        "Cancer": {"Acc": (None, None), "F1": (None, None)},
        "Credit": {"Acc": (None, None), "F1": (None, None)},
        "Income": {"Acc": (None, None), "F1": (None, None)},
    },
}

data_vertical = {
    "SplitNN [TabNet] (2 cl.)": {
        "Cancer": {"Acc": (0.98, 0.01), "F1": (0.98, 0.01)},
        "Credit": {"Acc": (0.93, 0.01), "F1": (0.48, 0.01)},
        "Income": {"Acc": (0.56, 0.25), "F1": (0.42, 0.17)},
    },
    "SplitNN [TabNet] (3 cl.)": {
        "Cancer": {"Acc": (0.98, 0.01), "F1": (0.98, 0.01)},
        "Credit": {"Acc": (0.93, 0.01), "F1": (0.48, 0.01)},
        "Income": {"Acc": (0.62, 0.20), "F1": (0.56, 0.16)},
    },
    "FedTree (2 cl.)": {
        "Cancer": {"Acc": (0.94, 0.01), "F1": (0.93, 0.01)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.69, 0.02)},
        "Income": {"Acc": (0.87, 0.01), "F1": (0.80, 0.01)},
    },
    "FedTree (3 cl.)": {
        "Cancer": {"Acc": (0.93, 0.01), "F1": (0.92, 0.01)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.69, 0.01)},
        "Income": {"Acc": (0.87, 0.01), "F1": (0.80, 0.01)},
    },
    "FC [PC] (2 cl.)": {
        "Cancer": {"Acc": (0.96, 0.01), "F1": (0.96, 0.01)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.67, 0.01)},
        "Income": {"Acc": (0.84, 0.02), "F1": (0.74, 0.01)},
    },
    "FC [PC] (3 cl.)": {
        "Cancer": {"Acc": (0.95, 0.01), "F1": (0.95, 0.01)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.66, 0.02)},
        "Income": {"Acc": (0.84, 0.01), "F1": (0.74, 0.01)},
    },
    "FC [DT] (2 cl.)": {
        "Cancer": {"Acc": (0.96, 0.01), "F1": (0.96, 0.02)},
        "Credit": {"Acc": (0.93, 0.01), "F1": (0.60, 0.02)},
        "Income": {"Acc": (0.83, 0.02), "F1": (0.67, 0.02)},
    },
    "FC [DT] (3 cl.)": {
        "Cancer": {"Acc": (0.95, 0.01), "F1": (0.95, 0.03)},
        "Credit": {"Acc": (0.93, 0.01), "F1": (0.60, 0.02)},
        "Income": {"Acc": (0.82, 0.02), "F1": (0.67, 0.02)},
    },
    "FedAvg [TabNet]": {
        "Cancer": {"Acc": (None, None), "F1": (None, None)},
        "Credit": {"Acc": (None, None), "F1": (None, None)},
        "Income": {"Acc": (None, None), "F1": (None, None)},
    },
}

data_hybrid = {
    "FC [PC] (2 cl.)": {
        "Cancer": {"Acc": (0.94, 0.01), "F1": (0.94, 0.01)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.67, 0.01)},
        "Income": {"Acc": (0.82, 0.02), "F1": (0.71, 0.01)},
    },
    "FC [PC] (3 cl.)": {
        "Cancer": {"Acc": (0.94, 0.01), "F1": (0.94, 0.01)},
        "Credit": {"Acc": (0.92, 0.01), "F1": (0.67, 0.02)},
        "Income": {"Acc": (0.80, 0.01), "F1": (0.70, 0.01)},
    },
    "FC [DT] (2 cl.)": {
        "Cancer": {"Acc": (0.96, 0.01), "F1": (0.96, 0.02)},
        "Credit": {"Acc": (0.93, 0.01), "F1": (0.60, 0.02)},
        "Income": {"Acc": (0.82, 0.02), "F1": (0.66, 0.02)},
    },
    "FC [DT] (3 cl.)": {
        "Cancer": {"Acc": (0.96, 0.01), "F1": (0.96, 0.01)},
        "Credit": {"Acc": (0.93, 0.01), "F1": (0.54, 0.02)},
        "Income": {"Acc": (0.82, 0.02), "F1": (0.66, 0.02)},
    },
    "FedAvg [TabNet]": {
        "Cancer": {"Acc": (None, None), "F1": (None, None)},
        "Credit": {"Acc": (None, None), "F1": (None, None)},
        "Income": {"Acc": (None, None), "F1": (None, None)},
    },
    "SplitNN [TabNet]": {
        "Cancer": {"Acc": (None, None), "F1": (None, None)},
        "Credit": {"Acc": (None, None), "F1": (None, None)},
        "Income": {"Acc": (None, None), "F1": (None, None)},
    },
    "FedTree": {
        "Cancer": {"Acc": (None, None), "F1": (None, None)},
        "Credit": {"Acc": (None, None), "F1": (None, None)},
        "Income": {"Acc": (None, None), "F1": (None, None)},
    },
}

plt.rcParams.update({"font.size": 18, "font.family": "serif"})


def plot_classification_results():

    # Sample data (Accuracy or F1 values can be substituted here)
    categories = ["Cancer", "Credit", "Income"]

    # Number of methods and categories
    methods = 5
    datasets = 3
    metric = "F1"

    # Generate random sample data for each of the sections (horizontal, vertical, hybrid)
    horizontal_data = np.zeros((methods, datasets))
    horizontal_std = np.zeros((methods, datasets))
    horizontal_methods = [
        "FedAvg [TabNet] (10 cl.)",
        "SplitNN [TabNet]",
        "FedTree (10 cl.)",
        "FC [PC] (10 cl.)",
        "FC [DT] (10 cl.)",
    ]
    for i, method in enumerate(horizontal_methods):
        horizontal_data[i] = [
            data_horizontal[method][category][metric][0] for category in categories
        ]
        horizontal_std[i] = [
            data_horizontal[method][category][metric][1] for category in categories
        ]

    vertical_data = np.zeros((methods, datasets))
    vertical_std = np.zeros((methods, datasets))
    vertical_methods = [
        "FedAvg [TabNet]",
        "SplitNN [TabNet] (3 cl.)",
        "FedTree (3 cl.)",
        "FC [PC] (3 cl.)",
        "FC [DT] (3 cl.)",
    ]
    for i, method in enumerate(vertical_methods):
        vertical_data[i] = [
            data_vertical[method][category][metric][0] for category in categories
        ]
        vertical_std[i] = [
            data_vertical[method][category][metric][1] for category in categories
        ]

    hybrid_data = np.zeros((methods, datasets))
    hybrid_std = np.zeros((methods, datasets))
    hybrid_methods = [
        "FedAvg [TabNet]",
        "SplitNN [TabNet]",
        "FedTree",
        "FC [PC] (3 cl.)",
        "FC [DT] (3 cl.)",
    ]
    for i, method in enumerate(hybrid_methods):
        hybrid_data[i] = [
            data_hybrid[method][category][metric][0] for category in categories
        ]
        hybrid_std[i] = [
            data_hybrid[method][category][metric][1] for category in categories
        ]

    # Set up the figure and axes
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), sharey=True, width_ratios=[2, 2, 1]
    )

    # Titles for each section
    titles = ["Horizontal FL", "Vertical FL", "Hybrid FL"]

    # Data for each section
    data = [horizontal_data, vertical_data, hybrid_data]
    data_std = [horizontal_std, vertical_std, hybrid_std]

    method_names = [
        "Fed Avg [TabNet]",
        "SplitNN [TabNet]",
        "FedTree",
        "FC [PC]",
        "FC [DT]",
    ]

    colors = ["#ff97c4", "#ffb0c4", "#ffc8c4", "#b5e8a4", "#89d2a3"]
    colors = sns.color_palette("pastel")
    colors = [colors[1], colors[2], colors[3], colors[9], colors[0]]

    # remove data where all entries are NaN
    methods_per_data = [np.sum(np.isnan(data), axis=1) for data in data]
    nr_entries_per_method = [np.sum(method == 0) for method in methods_per_data]

    handles = {}

    # Plotting for each section
    for ax, title, data_section, std_section, nr_methods in zip(
        axes, titles, data, data_std, nr_entries_per_method
    ):
        x = np.arange(len(categories))  # the label locations
        width = 0.15  # the width of the bars
        offsets = np.linspace(-width * 2, width * 2, nr_methods)
        if nr_methods == 2:
            offsets = np.linspace(-width * 1.4, width * 1.4, nr_methods)

        # Plot each method as a group of bars
        index = 0
        for i, offset in enumerate(offsets):
            while any(np.isnan(data_section[index])):
                index += 1
            if nr_methods == 2:
                width = 0.3
            handles[index] = ax.bar(
                x + offset,
                np.nan_to_num(data_section[index], nan=0),
                width,
                label=method_names[index],
                yerr=np.nan_to_num(std_section[index], nan=0),
                capsize=5,
                color=colors[index],
                error_kw=dict(ecolor="gray", lw=1.5, capsize=3, capthick=1),
            )
            index += 1

        # Customizing the axes
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_ylim([0, 1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if nr_methods == 2:
            ax.set_xticklabels(["Can.", "Cre.", "Inc."])
        else:
            ax.set_xticklabels(categories)
        # ax.set_xlabel('Datasets')

    # Common labels and legend
    axes[0].set_ylabel("F1 Score")
    legend_handles = []
    for i in range(methods):
        legend_handles.append(handles[i])
    fig.legend(method_names, loc="upper center", ncol=methods, handles=legend_handles)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fl_settings_results_raw.pdf")


def plot_runtime_comparison_image_datasets():

    # Data
    categories = ["CelebA", "Imagenet32", "Imagenet"]
    x_labels = ["Einet", "PyJuice", "FC 2 cl.", "FC 4 cl.", "FC 8 cl.", "FC 16 cl."]
    num_groups = len(x_labels)
    num_bars = len(categories)

    # Generate random data between 0 and 1
    np.random.seed(42)
    data = np.random.rand(num_groups, num_bars)

    # Bar width and x-axis positions
    bar_width = 0.25
    x = np.arange(num_groups)

    # Colors for bars
    colors = ["royalblue", "tomato", "gold"]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(num_bars):
        ax.bar(
            x + i * bar_width,
            data[:, i],
            width=bar_width,
            label=categories[i],
            color=colors[i],
        )

    # X-axis and labels
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Value")
    ax.set_title("Comparison of Categories Across Models")
    ax.legend()

    # Show the plot
    plt.show()
