import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from matplotlib.ticker import FuncFormatter

from dataset_analysis.settings import PLOT_SUFFIXES, QUERY_LANGUAGES, RES_FOLDER, RESCALED_FOLDER


# ['4477AA', 'EE6677', '228833', 'CCBB44', '66CCEE', 'AA3377', 'BBBBBB'])
def get_colors_bright():
    return np.array([
        [0x44/255, 0x77/255, 0xAA/255],  # '4477AA' (blue, Cypher)
        [0x22/255, 0x88/255, 0x33/255],  # '228833' (green, SPARQL)
        [0xCC/255, 0xBB/255, 0x44/255],  # 'CCBB44' (yellow, SQL)
    ])


def get_colors():
    return np.array([
        [0.1, 0.1, 0.1],          # black
        [0.4, 0.4, 0.4],          # very dark gray
        [0.7, 0.7, 0.7],          # dark gray
        [0.9, 0.9, 0.9],          # light gray
        [0.984375, 0.7265625, 0], # dark yellow
        [1, 1, 0.9]               # light yellow
    ])


def color_bars(ax, colors):
    # Iterate over each subplot
    dark_color = colors[2]
    for p in ax.patches:
        p.set_edgecolor(dark_color)


def set_style(font_scale=1.0, style_rc=None):
    import seaborn as sns
    import matplotlib
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    # sns.set(font='serif')
    # sns.set(font_scale=font_scale)
    # if style_rc != None:
    sns.set(font='serif', font_scale=font_scale, rc=style_rc)
    # sns.set(font='serif', font_scale=font_scale)
    sns.set_style('whitegrid', {'axes.edgecolor': '.0', 'grid.linestyle': u'--',
                                'grid.color': '.8', 'xtick.color': '.15',
                                'xtick.minor.size': 3.0, 'xtick.major.size':
                                6.0, 'ytick.color': '.15', 'ytick.minor.size':
                                3.0, 'ytick.major.size': 6.0,
                                "font.family": "serif",
                                "font.serif": ["Times", "Palatino", "serif"]
                                # , "patch.linewidth":1.1
                                })
    matplot_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    custom_colors = get_colors_bright()
    sns.set_palette(custom_colors)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42



def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(round(100 * y))
    
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def plot_bars(df, y_list, split: str, order=None, grid_size=(2, 2), ymax=None, ylab="Number"):
    set_style(font_scale=1.2)
    
    f, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 8))
    i = 0
    hue_order = sorted(df["query_language"].unique())
    ncol = len(df["query_language"].unique())
    for y in y_list:
        row = i // 2
        col = i % 2
        if grid_size == (1, 1):
            axs_i = axs
        else:
            axs_i = axs[row, col]
        sns.barplot(x="query_language", hue="query_language", y=y, data=df, ax=axs_i,
            errorbar=None, order=order, hue_order=hue_order)
        axs_i.set_title(f"{y} ({split})")
        axs_i.set_xlabel('')  # Remove the x-axis label
        if ymax is not None:
            try: 
                ym = ymax[i]
            except TypeError:
                ym = ymax
            axs_i.set_ylim(0, ym)
        if row==0 and col==0:
            axs_i.set_ylabel(ylab)
        elif row ==1 and col==0:
            axs_i.set_ylabel(ylab)
        else:
            axs_i.set_ylabel('')

        i+=1
    f.tight_layout()
    f.savefig("query_stats.pdf", bbox_inches='tight')
    return f


def plot_histograms(df, split, dataset):
    sql_df = df[df['query_language'] == 'SQL']
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(sql_df['Complexity Score'], kde=True, color='blue')
    plt.title("Your Complexity Score")
    plt.subplot(1,2,2)
    sns.histplot(sql_df['SQomplexity Score'], kde=True, color='orange')
    plt.title("SQomplexity Score")
    plt.suptitle(f"{dataset} - {split}: Histogram of SQL Complexity Scores")
    for suffix in PLOT_SUFFIXES:
        plt.savefig(RES_FOLDER.joinpath(f"{dataset}-{split}_complexity_hist").with_suffix(suffix),
                    bbox_inches='tight')
    plt.close()


def plot_bucket_comparison(df, split, dataset, n_buckets=5):
    sql_df = df[df['query_language'] == 'SQL'].dropna(subset=['Complexity Score', 'SQomplexity Score'])
    # Try to assign quantile buckets, and handle the case where too few unique edges remain.
    try:
        # Use qcut to get number of bins actually possible
        buckets, bin_edges = pd.qcut(
            sql_df['Complexity Score'],
            n_buckets,
            labels=None,
            retbins=True,
            duplicates='drop'
        )
        # Now generate correct number of labels
        correct_n_buckets = len(bin_edges) - 1
        bucket_labels = [f"Q{i+1}" for i in range(correct_n_buckets)]
        sql_df['bucket'] = pd.qcut(
            sql_df['Complexity Score'],
            correct_n_buckets,
            labels=bucket_labels,
            duplicates='drop'
        )
    except ValueError as e:
        print(f"Could not assign quantile buckets for {dataset}-{split}: {e}")
        return  # or handle fallback case

    # Continue as before
    bucket_means = sql_df.groupby('bucket')[['Complexity Score', 'SQomplexity Score']].mean().reset_index()
    
    plt.figure(figsize=(8,6))
    width = 0.35
    x = range(correct_n_buckets)
    plt.bar(x, bucket_means['Complexity Score'], width, label="Your Score")
    plt.bar([i+width for i in x], bucket_means['SQomplexity Score'], width, label="SQomplexity Score")
    plt.xticks([i + width/2 for i in x], bucket_means['bucket'])
    plt.xlabel("Complexity Score Quantile")
    plt.ylabel("Mean Score")
    plt.title(f'{dataset} - {split}: Mean SQL Complexity Per Bucket')
    plt.legend()
    for suffix in PLOT_SUFFIXES:
        plt.savefig(RES_FOLDER.joinpath(f"{dataset}-{split}_bucket_comparison").with_suffix(suffix),
                    bbox_inches='tight')
    plt.close()



# RESCALED versions
def plot_histograms_rescaled(df, split, dataset, output_folder=RESCALED_FOLDER):
    sql_df = df[df['query_language'].str.lower() == 'sql']
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(sql_df['Complexity Score'], kde=True, color='blue')
    plt.title("Your Complexity Score")
    plt.subplot(1,2,2)
    sns.histplot(sql_df['SQomplexity Score (rescaled)'], kde=True, color='orange')
    plt.title("SQomplexity Score (rescaled)")
    plt.suptitle(f"{dataset} - {split}: Histogram of SQL Complexity Scores (Rescaled)")

    for suffix in PLOT_SUFFIXES:
        plt.savefig(output_folder.joinpath(f"{dataset}-{split}_complexity_hist_rescaled").with_suffix(suffix),
                    bbox_inches='tight')
    plt.close()


def plot_bucket_comparison_rescaled(df, split, dataset, output_folder=RESCALED_FOLDER, n_buckets=5):
    import matplotlib.pyplot as plt
    sql_df = df[df['query_language'].str.lower() == 'sql'].dropna(subset=['Complexity Score', 'SQomplexity Score (rescaled)'])
    # Robust qcut with correct label assignment (from our prior messages)
    try:
        buckets, bin_edges = pd.qcut(
            sql_df['Complexity Score'],
            n_buckets,
            labels=None,
            retbins=True,
            duplicates='drop'
        )
        correct_n_buckets = len(bin_edges) - 1
        bucket_labels = [f"Q{i+1}" for i in range(correct_n_buckets)]
        sql_df['bucket'] = pd.qcut(
            sql_df['Complexity Score'],
            correct_n_buckets,
            labels=bucket_labels,
            duplicates='drop'
        )
    except ValueError as e:
        print(f"Could not assign quantile buckets for {dataset}-{split} (rescaled): {e}")
        return
    bucket_means = sql_df.groupby('bucket')[['Complexity Score', 'SQomplexity Score (rescaled)']].mean().reset_index()
    width = 0.35
    x = range(correct_n_buckets)
    plt.figure(figsize=(8,6))
    plt.bar(x, bucket_means['Complexity Score'], width, label="Your Score")
    plt.bar([i+width for i in x], bucket_means['SQomplexity Score (rescaled)'], width, label="SQomplexity Score (rescaled)")
    plt.xticks([i + width/2 for i in x], bucket_means['bucket'])
    plt.xlabel("Complexity Score Quantile")
    plt.ylabel("Mean Score")
    plt.title(f'{dataset} - {split}: Mean SQL Complexity Per Bucket (Rescaled)')
    plt.legend()
    for suffix in PLOT_SUFFIXES:
        plt.savefig(output_folder.joinpath(f"{dataset}-{split}_bucket_comparison_rescaled").with_suffix(suffix),
                    bbox_inches='tight')
    plt.close()


def compare_complexity_metrics_rescaled(df, split, dataset, output_folder=RESCALED_FOLDER):
    import matplotlib.pyplot as plt
    sql_df = df[df['query_language'].str.lower() == 'sql'].dropna(subset=['Complexity Score', 'SQomplexity Score (rescaled)'])
    corr = sql_df['Complexity Score'].corr(sql_df['SQomplexity Score (rescaled)'])
    print(f"Rescaled Correlation ({dataset}-{split}): {corr:.3f}")

    plt.figure()
    plt.scatter(sql_df['Complexity Score'], sql_df['SQomplexity Score (rescaled)'])
    plt.xlabel('Your Complexity Score')
    plt.ylabel('SQomplexity Score (rescaled)')
    plt.title('SQL Complexity Comparison (Rescaled)')
    for suffix in PLOT_SUFFIXES:
        plt.savefig(output_folder.joinpath(f"{dataset}-{split}_sqomplexity_scatter_rescaled").with_suffix(suffix), bbox_inches='tight')
    plt.close()