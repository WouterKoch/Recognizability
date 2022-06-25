import os
import json
from turtle import color
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from MachineLearning.keras_utils import DataFrameIterator
import numpy as np
from Tools.prefetch_images import fetch as prefetch_images
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from tqdm import tqdm
import scipy
from Scripts.bias_plot import create_plots as create_bias_plots
from Scripts.bias_plot import get_rank_stats as get_rank_stats
import statsmodels.api as sm

sns.set_style("whitegrid")
pd.options.mode.chained_assignment = None


# def plot_taxon_performance_vs_trait(trait, path, output_path, extension, log=False, categorical=False):
#     df_species_stats = pd.read_csv(os.path.join(
#         path, "Stats per species.csv")).sort_values(by="Species")

#     df_traits = pd.read_csv(os.path.join(
#         path, "Species characteristics.csv"))

#     for taxon in ["Anseriformes", "Charadriiformes", "Passeriformes"]:
#         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#         fig.suptitle(f"Performance over {trait} of {taxon} species")
#         plt.subplots_adjust(wspace=.25, hspace=.4)

#         for i, metric in enumerate(["Average score (max)", "Recall (max)", "Precision (max)", "F1 (max)"]):
#             df_subset = df_species_stats[(df_species_stats["Metric"] == metric) & (
#                 df_species_stats["Taxon"] == taxon)]
#             df_subset["trait"] = df_subset["Species"].apply(
#                 lambda x: df_traits[df_traits["Scientific name"] == x][trait].values[0])

#             if log:
#                 df_subset["trait"] = np.log10(df_subset["trait"])

#             if categorical:
#                 ax = sns.violinplot(ax=axes[int(
#                     i / 2), int(i % 2)], x="trait", y="Value", data=df_subset, inner="stick")
#             else:
#                 ax = sns.scatterplot(
#                     ax=axes[int(i / 2), int(i % 2)], data=df_subset, y="Value", x="trait")

#                 add_linregress(ax, df_subset, x="trait", y="Value", position="bottom")

#                 if log:
#                     ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

#             ax.set(ylim=(0, 1.1))
#             ax.set_yticks(np.arange(0, 1.1, .2))
#             ax.set_xlabel(trait)
#             ax.set_ylabel(metric)
#             ax.legend([], [], frameon=False)

#         plt.savefig(os.path.join(
#             output_path, f"Species vs {trait} ({taxon}).{extension}"), dpi=300)
#         plt.close()



def collected_stats_slopes(path, filename, output_path, extension, x_parameter, y_parameter, x_log=False, taxa=["Anseriformes", "Charadriiformes", "Passeriformes"]):
    df = pd.read_csv(os.path.join(path, filename))
    df = df[df["Order"].isin(taxa)]
    df.sort_values(by=[x_parameter], inplace=True)
    taxa.sort()

    fig, ax = plt.subplots(1, 1, figsize=(9*.75, 6*.75))
    ax.set_facecolor('#f9f3eb')

    x_min, x_max = 0, 0

    for taxon in taxa:
        df_subset = df[df["Order"] == taxon]
        df_subset = df_subset[[x_parameter, y_parameter]]
        df_subset.dropna(inplace=True)

        if x_log:
            df_subset[x_parameter] = np.log10(df_subset[x_parameter])

        xdat = df_subset[x_parameter]
        xdat = sm.add_constant(xdat)
        ydat = df_subset[y_parameter]
        model = sm.OLS(ydat, xdat).fit()

        alpha = .05 # for a 95% confidence interval

        if model.pvalues[x_parameter] < .05:
            print(f"{taxon} s: {model.params[x_parameter]}, i: {model.params['const']}")

            stats_str = f"{taxon}\n("
            stats_str += "R² = " + "{:.2f}".format(np.round(model.rsquared, 2))
            stats_str += ", "
            stats_str += "p = " + "{:.4f}".format(model.pvalues[x_parameter]) + ")"

            conf = model.conf_int(.05)

            x_min = min(x_min, conf[0][1])
            x_max = max(x_max, conf[1][1] + .05)

            ax.plot([conf[0][1], conf[1][1]], [taxon, taxon])
            ax.scatter([model.params[x_parameter]], [taxon], label=stats_str)

    ax.invert_yaxis()    
    ax.set_yticklabels([])
    ax.set_ylabel("Order")
    ax.set_xlabel("Data availabiliy slope")
    ax.set_xlim(x_min, x_max)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_path, f"Collected - {y_parameter} vs {x_parameter} slopes {', '.join(taxa)}.{extension}"), dpi=300)
    plt.close()

def collected_stats_regressions(path, filename, output_path, extension, x_parameter, y_parameter, x_log=False, taxa=["Anseriformes", "Charadriiformes", "Passeriformes"]):
    df = pd.read_csv(os.path.join(path, filename))
    df = df[df["Order"].isin(taxa)]
    df.sort_values(by=[x_parameter], inplace=True)
    taxa.sort()
    
    x_max = df[x_parameter].max()
    x_min = df[x_parameter].min()
    y_max = df[y_parameter].max()
    y_min = df[y_parameter].min()

    if y_max <= 1 and y_min >= 0:
        y_max, y_min = 1, 0.5

    if x_log:
        x_min, x_max = np.log10(x_min), np.log10(x_max)
        df[x_parameter] = np.log10(df[x_parameter])

    fig, ax = plt.subplots(1, 1, figsize=(9*.75, 6*.75))
    if x_log or x_parameter[-6:] == " (log)":
        ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

    for taxon in taxa:
        df_subset = df[df["Order"] == taxon]
        df_subset = df_subset[[x_parameter, y_parameter]]
        df_subset.dropna(inplace=True)

        if x_log:
            df_subset[x_parameter] = np.log10(df_subset[x_parameter])

        xdat = df_subset[x_parameter]
        xdat = sm.add_constant(xdat)
        ydat = df_subset[y_parameter]
        model = sm.OLS(ydat, xdat).fit()

        alpha = .05 # for a 95% confidence interval
        predictions = model.get_prediction(xdat).summary_frame(alpha)

        if model.pvalues[x_parameter] < .05:
            stats_str = f"{taxon}\n("
            stats_str += "R² = " + "{:.2f}".format(np.round(model.rsquared, 2))
            stats_str += ", "
            stats_str += "p = " + "{:.4f}".format(model.pvalues[x_parameter]) + ")"


            ax.fill_between(df_subset[x_parameter], predictions['mean_ci_lower'],
                            predictions['mean_ci_upper'], alpha=.15, label=None)
            ax.plot(df_subset[x_parameter], predictions['mean'], label=stats_str)


    x_label = x_parameter if x_parameter[-6:] != " (log)" else x_parameter[:-6]
    x_label = "Data availability" if x_label == "Images" else x_label
    y_label = y_parameter if y_parameter[:3] != "F1 " else "$F_{1}$-score"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_path, f"Collected - {y_parameter} vs {x_parameter} regressions {', '.join(taxa)}.{extension}"), dpi=300)
    plt.close()


def collected_stats_scatter(path, filename, output_path, extension, x_parameter, y_parameter, x_log=False, taxa=["Anseriformes", "Charadriiformes", "Passeriformes"], highlight=[]):
    df = pd.read_csv(os.path.join(path, filename))
    df = df[df["Order"].isin(taxa)]

    x_max = df[x_parameter].max()
    x_min = df[x_parameter].min()
    y_max = df[y_parameter].max()
    y_min = df[y_parameter].min()

    if y_max <= 1 and y_min >= 0:
        y_max, y_min = 1.05, 0

    if x_log:
        x_min, x_max = np.log10(x_min), np.log10(x_max)

    if len(taxa) == 3:
        fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plt.subplots_adjust(wspace=.25, hspace=.4)

    for i, taxon in enumerate(taxa):
        df_subset = df[df["Order"] == taxon]

        df_subset = df_subset[[x_parameter, y_parameter, "Species"]]
        df_subset.dropna(inplace=True)

        if x_log:
            df_subset[x_parameter] = np.log10(df_subset[x_parameter])

        df_subset.sort_values(by=[x_parameter], inplace=True)

        xdat = df_subset[x_parameter]
        xdat = sm.add_constant(xdat)
        ydat = df_subset[y_parameter]
        model = sm.OLS(ydat, xdat).fit()

        alpha = .05
        predictions = model.get_prediction(xdat).summary_frame(alpha)

        if len(taxa) == 3:
            ax = axes[i]
        else:
            ax = axes[int(i / 2), int(i % 2)]

        ax.set_facecolor('#f9f3eb')

        if model.pvalues[x_parameter] < .05:
            ax.fill_between(df_subset[x_parameter], predictions['mean_ci_lower'],
                            predictions['mean_ci_upper'], alpha=.15, label='Confidence interval', color="darkslategray")
        for hl in highlight:
            if hl in df_subset["Species"].to_list():
                ax.scatter(x=df_subset[df_subset["Species"] == hl][x_parameter],
                           y=df_subset[df_subset["Species"] == hl][y_parameter], c="darkgoldenrod", s=100)
        
        ax.scatter(data=df_subset, y=y_parameter, x=x_parameter, color="darkslategray", s=10)
        if model.pvalues[x_parameter] < .05:
            ax.plot(df_subset[x_parameter], predictions['mean'],
                    label='Regression line', color="darkslategray")

        stats_str = "R² = " + "{:.2f}".format(np.round(model.rsquared, 2))
        stats_str += "\n"
        stats_str += "p = " + "{:.2e}".format(model.pvalues[x_parameter])


        if "const" in model.params:
            print(f"{taxon} s: {model.params[x_parameter]}, i: {model.params['const']}")

        ax.text(0.7, 0.05, stats_str, horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

     

        if x_log or x_parameter[-6:] == " (log)":
            ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

        x_label = x_parameter if x_parameter[-6:] != " (log)" else x_parameter[:-6]
        x_label = "Data availability" if x_label == "Images" else x_label
        y_label = y_parameter if y_parameter[:3] != "F1 " else "$F_{1}$-score"

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(x_min*.99, x_max*1.01)
        ax.set_ylim(y_min, y_max*1.01)
        ax.legend([], [], frameon=False)
        ax.set_title(taxon)

    plt.savefig(os.path.join(
        output_path, f"Collected - {y_parameter} vs {x_parameter} {', '.join(taxa)}.{extension}"), dpi=300)
    plt.close()


def get_order(df, species, column):
    if species not in df["Species"].to_list():
        return None

    return df[df["Species"] == species][column].to_list()[0]


# def annotation_vs_collected_stats(path, stats_filename, annotations_filename, output_path, extension, x_parameter, y_parameter, x_log=False, y_log=False):
#     df = pd.read_csv(os.path.join(path, annotations_filename))
#     df_stats = pd.read_csv(os.path.join(path, stats_filename))

#     df["Order"] = df.apply(lambda x: get_order(
#         df_stats, x["species"], column="Order"), axis=1)
#     df["Images (log)"] = df.apply(lambda x: get_order(
#         df_stats, x["species"], column="Images (log)"), axis=1)

#     df = df.dropna()

#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(f"{y_parameter} over {x_parameter} per species")
#     plt.subplots_adjust(wspace=.25, hspace=.4)

#     for i, taxon in enumerate(["Anseriformes", "Charadriiformes", "Passeriformes"]):
#         df_subset = df[df["Order"] == taxon]

#         if x_log:
#             df_subset[x_parameter] = np.log10(df_subset[x_parameter])

#         medians = df_subset.groupby('species').median()

#         medians.to_csv(os.path.join(path, f"{taxon} annotation stats.csv"))

#         if y_log:
#             df_subset[y_parameter] = np.log10(df_subset[y_parameter])
#             medians[y_parameter] = np.log10(medians[y_parameter])

#         ax = sns.scatterplot(
#             ax=axes[int(i / 2), int(i % 2)], data=df_subset, y=y_parameter, x=x_parameter, alpha=.15)

#         medians = df_subset.groupby('species').median()

#         ax.scatter(data=medians, x=x_parameter, y=y_parameter)

#         add_linregress(ax, df_subset, x=x_parameter, y=y_parameter, position="bottom")

#         if x_log:
#             ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))
#         if y_log:
#             ax.yaxis.set_major_formatter(lambda x, y: int(10 ** y))

#         ax.set_xlabel(x_parameter)
#         ax.set_ylabel(y_parameter)
#         ax.legend([], [], frameon=False)
#         ax.set_title(taxon)

#     plt.savefig(os.path.join(
#         output_path, f"Collected - {y_parameter} vs {x_parameter} per species.{extension}"), dpi=300)
#     plt.close()


# def plot_taxon_performance_vs_annotation(annotation, path, output_path, extension, log=False, categorical=False, stats_file="Annotation stats"):
#     df_species_stats = pd.read_csv(os.path.join(
#         path, "Stats per species.csv")).sort_values(by="Species")

#     df_annotated = pd.read_csv(os.path.join(
#         path, f"{stats_file}.csv"))

#     for taxon in ["Anseriformes", "Charadriiformes", "Passeriformes"]:
#         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#         fig.suptitle(f"Performance over {annotation} of {taxon} species")
#         plt.subplots_adjust(wspace=.25, hspace=.4)

#         for i, metric in enumerate(["Average score (max)", "Recall (max)", "Precision (max)", "F1 (max)"]):
#             df_subset = df_species_stats[(df_species_stats["Metric"] == metric) & (
#                 df_species_stats["Taxon"] == taxon)]
#             df_subset["trait"] = df_subset["Species"].apply(
#                 lambda x: -999 if x not in df_annotated["species"].to_list() else df_annotated[df_annotated["species"] == x][annotation].values[0])

#             df_subset = df_subset[df_subset["trait"] > -999]

#             if log:
#                 df_subset["trait"] = np.log10(df_subset["trait"])

#             if categorical:
#                 ax = sns.violinplot(ax=axes[int(
#                     i / 2), int(i % 2)], x="trait", y="Value", data=df_subset, inner="stick")
#             else:
#                 ax = sns.scatterplot(
#                     ax=axes[int(i / 2), int(i % 2)], data=df_subset, y="Value", x="trait")

#                 add_linregress(ax, df_subset, "trait", "Value", position="bottom")

#                 if log:
#                     ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

#             ax.set(ylim=(0, 1.1))
#             ax.set_yticks(np.arange(0, 1.1, .2))
#             ax.set_xlabel(annotation)
#             ax.set_ylabel(metric)
#             ax.legend([], [], frameon=False)

#         plt.savefig(os.path.join(
#             output_path, f"Species vs {annotation} ({taxon}).{extension}"), dpi=300)
#         plt.close()


# def plot_trait_vs_trait(trait_x, trait_y, path, output_path, extension):
#     df_traits = pd.read_csv(os.path.join(
#         path, "Species characteristics.csv"))

#     df_species_stats = pd.read_csv(os.path.join(
#         path, "Stats per species.csv"))

#     df_traits["Taxon"] = df_traits["Scientific name"].apply(
#         lambda name: df_species_stats[df_species_stats["Species"] == name].reset_index().at[0, "Taxon"])

#     for taxon in list(df_traits["Taxon"].unique()):
#         df_subset = df_traits[(df_traits["Taxon"] == taxon)]

#         fig, axes = plt.subplots(1, 1, figsize=(12, 8))
#         fig.suptitle(f"{trait_y} over {trait_x} of {taxon} species")
#         ax = sns.scatterplot(data=df_subset, y=trait_y, x=trait_x)

#         add_linregress(ax, df_subset, x=trait_x, y=trait_y, position="bottom")

#         ax.set_xlabel(trait_x)
#         ax.set_ylabel(trait_y)
#         ax.legend([], [], frameon=False)

#         plt.savefig(os.path.join(
#             output_path, f"{trait_y} vs {trait_x} ({taxon}).{extension}"), dpi=300)
#         plt.close()


# def plot_taxon_performance_vs_images(path, output_path, extension):
#     df_species_stats = pd.read_csv(os.path.join(
#         path, "Stats per species.csv")).sort_values(by="Species")

#     for taxon in ["Anseriformes", "Charadriiformes", "Passeriformes"]:
#         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#         fig.suptitle("Performance over number of images of " + taxon)
#         plt.subplots_adjust(wspace=.25, hspace=.4)

#         x = np.log10(
#             df_species_stats[(df_species_stats["Metric"] == "Images (testing)") & (df_species_stats["Taxon"] == taxon)][
#                 "Value"].tolist())

#         for i, metric in enumerate(["Average score (max)", "Recall (max)", "Precision (max)", "F1 (max)"]):
#             y = \
#                 df_species_stats[(df_species_stats["Metric"] == metric) & (df_species_stats["Taxon"] == taxon)][
#                     "Value"].tolist()

#             ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y)
#             ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

#             add_linregress(ax, x, y, position="bottom")

#             ax.set(ylim=(0, 1.1))
#             ax.set_yticks(np.arange(0, 1.1, .2))
#             ax.set_xlabel("Number of images in test set")
#             ax.set_ylabel(metric)
#             ax.legend([], [], frameon=False)

#         plt.savefig(os.path.join(
#             output_path, f"Species vs number of images ({taxon}).{extension}"), dpi=300)
#         plt.close()


# def add_linregress(ax, df, x, y, position="top"):
#     df = pd.DataFrame(df)
#     df.sort_values(by=[x], inplace=True)

#     x_vals = df[x].to_list()
#     y_vals = df[y].to_list()

#     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
#         x_vals, y_vals)

#     # if p_value < .05:
#     #     ax.plot(x_vals, slope * np.asarray(x_vals) + intercept)

#     stats_str = "R² = " + "{:.2f}".format(np.round(r_value * r_value, 2))
#     stats_str += "\n"
#     if p_value < 0.0001:
#         stats_str += "p = " + "{:.2e}".format(p_value)
#     else:
#         stats_str += "p = " + "{:.5f}".format(np.round(p_value, 5))
#     ax.text(0.7, 0.05 if position == "bottom" else 0.95, stats_str, horizontalalignment="left",
#             verticalalignment=position,
#             transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

#     # df_ols=df
#     # df_ols.rename(columns={x:'x'}, inplace=True)
#     # df_ols.rename(columns={y:'y'}, inplace=True)
#     # df_ols['y']=y
#     # print(df_ols.head())

#     # results_formula = sm.OLS(formula='y ~ x', data=df_ols).fit()
#     # print(results_formula.params)

#     xdat = df[x]
#     xdat = sm.add_constant(xdat)
#     ydat = df[y]
#     model = sm.OLS(ydat, xdat).fit()

#     print(model.params.const)
#     print(model.params[x])

#     alpha = .05
#     predictions = model.get_prediction(xdat).summary_frame(alpha)

#     ax.fill_between(df[x], predictions['mean_ci_lower'],
#                     predictions['mean_ci_upper'], alpha=.5, label='Confidence interval')
#     ax.plot(df[x], predictions['mean'], label='Regression line')

#     # result = sm.OLS(y_vals, x_vals).fit()

#     print(
#         f"lingress, slope={slope} intercept={intercept}, p={p_value}, r={r_value}")
#     # print(result.summary())
#     # print(result.params)

#     # alpha = .05

#     return slope, intercept


# def plot_stats_between_experiments(subsetted_csv, superset_csv, output_path, extension):
#     df_subset = pd.read_csv(subsetted_csv)
#     df_superset = pd.read_csv(superset_csv)

#     for taxon in ["Anseriformes", "Charadriiformes", "Passeriformes"]:

#         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#         fig.suptitle("Models with 17 vs all species of " + taxon)
#         plt.subplots_adjust(wspace=.25, hspace=.4)

#         for i, metric in enumerate(["Average score (mean)", "Recall (mean)", "Precision (mean)", "F1 (mean)"]):
#             df = df_superset[(df_superset["Taxon"] == taxon) & (
#                 df_superset["Metric"] == metric)][["Species", "Value"]]
#             df_sub = df_subset[(df_subset["Taxon"] == taxon) & (
#                 df_subset["Metric"] == metric)][["Species", "Value"]]
#             df = df.merge(df_sub, left_on='Species',
#                           right_on='Species', suffixes=('_super', '_sub'))

#             ax = sns.scatterplot(
#                 ax=axes[int(i / 2), int(i % 2)], data=df, x="Value_sub", y="Value_super")

#             ax.plot([0, 1], [0, 1])

#             ax.set(ylim=(0, 1))
#             ax.set(xlim=(0, 1))
#             ax.set_xlabel("Performance among 17 species")
#             ax.set_ylabel(metric)
#             ax.legend([], [], frameon=False)

#         plt.savefig(os.path.join(
#             output_path, f"All vs some species, mean ({taxon}).{extension}"), dpi=300)
#         plt.close()


if __name__ == "__main__":
    print("USAGE")
    print("evaluate(): based on folders in the directory defined by the JOBS_DIR environment variable")
