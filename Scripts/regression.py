import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sys import exit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

import seaborn as sns
import scipy

import zipfile


def collect_data(folder):
    stats = pd.read_csv(os.path.join(os.environ["JOBS_DIR"], "Stats per species.csv"))

    stats.sort_values(by=["Taxon", "Species"], inplace=True)
    df = stats[stats["Metric"] == "F1 (mean)"].reset_index(drop=True)
    df.drop("Metric", axis=1, inplace=True)
    df.columns = ["Order", "Species", "F1 (mean)"]


    df["Images (log)"] = stats[stats["Metric"] ==
                               "Images (testing)"]["Value"].to_list()
    df["Images (log)"] = df["Images (log)"].apply(lambda x: np.log10(x + 200))

    stats = pd.read_csv(os.path.join(os.environ["STORAGE_DIR"], "Species characteristics.csv"))
    stats = stats[["Scientific name", "HWI",
                   "Body mass (log)", "Habitat", "Proportion AO vs TOVe", "Proportion AO+img vs AO", "Urban percentage"]]
    df = df.merge(stats, left_on="Species",
                  right_on="Scientific name", how="outer")

    df["spec"] = df["Species"].apply(lambda x: x[:x.rfind(" ")])
    for metric in ["HWI", "Body mass (log)", "Habitat"]:
        df[metric] = df.apply(lambda x: x[metric] if x[metric] > 0 else (None if len(stats[stats["Scientific name"] == x["spec"]]
                              [metric].to_list()) < 1 else stats[stats["Scientific name"] == x["spec"]][metric].to_list()[0]), axis=1)

    stats = pd.read_csv(os.path.join(os.environ["STORAGE_DIR"], "Annotation stats (median).csv"))
    stats["target pixels (log)"] = stats["max_target_cropped_pixels"].apply(
        lambda x: np.log10(x))
    stats["info pixels (log)"] = stats["max_target_info_cropped_pixels"].apply(
        lambda x: np.log10(x))
    stats = stats[["species", "target pixels (log)", "info pixels (log)"]]
    df = df.merge(stats, left_on="Species", right_on="species", how="left")

    stats = pd.read_csv(os.path.join(os.environ["STORAGE_DIR"], "img_per_obs.csv"), usecols=[
                        "scientificName", "img per obs"])
    df = df.merge(stats, left_on="Species",
                  right_on="scientificName", how="left")

    df.drop(["species", "spec", "Scientific name",
            "scientificName"], axis=1, inplace=True)

    df.to_csv(os.path.join(os.environ["STORAGE_DIR"], "out.csv"), index=False)


def add_linregress(ax, x, y, position="top", color=None, showstats=True):
    x = np.asarray(x)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    if p_value < .05:
        ax.plot(x, slope * x + intercept, color=color)
        stats_str = "R² = " + "{:.2f}".format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + "{:.2e}".format(p_value)
        else:
            stats_str += "p = " + "{:.5f}".format(np.round(p_value, 5))
        if showstats:
            ax.text(0.7, 0.05 if position == "bottom" else 0.95, stats_str, horizontalalignment="left",
                    verticalalignment=position,
                    transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))


def sanity(path, annotations_filename, stats_filename, characteristics_filename):
    df = pd.read_csv(
        "/home/wouter/Projects/PhD/voi_tested/Stats per species.csv")
    df = df.sort_values(by=["Species"])

    taxa = list(df["Taxon"].unique())

    for taxon in taxa:
        df_subset = df[df["Taxon"] == taxon]

        x_list = df_subset[df_subset["Metric"] == "Images (testing)"]["Value"].apply(
            lambda x: np.log10(x)).to_list()
        y_list = df_subset[df_subset["Metric"]
                           == "F1 (max)"]["Value"].to_list()

        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f"{taxon}")
        ax = sns.scatterplot(y=y_list, x=x_list)

        add_linregress(ax, x_list, y_list, position="bottom")

        ax.set_xlabel("Images testing log")
        ax.set_ylabel("F1")
        ax.legend([], [], frameon=False)

        plt.savefig(
            f"/home/wouter/Projects/PhD/Recognizability/Sanity {taxon}.png", dpi=300)
        plt.close()


def sanity_together():
    colormap = sns.color_palette("tab20", n_colors=12)

    df = pd.read_csv(
        "/home/wouter/Projects/PhD/voi_tested/Stats per species.csv")
    df = df.sort_values(by=["Species"])

    taxa = list(df["Taxon"].unique())
    taxa.sort()

    taxon_list = df[df["Metric"]
                    == "F1 (max)"]["Taxon"].to_list()

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    ax = sns.scatterplot()
    ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

    for t, taxon in enumerate(taxa):
        if taxon in ["Anseriformes", "Charadriiformes", "Passeriformes"]:
            continue

        color = colormap[t]
        df_subset = df[df["Taxon"] == taxon]
        x_list = df_subset[df_subset["Metric"] == "Images (testing)"]["Value"].apply(
            lambda x: np.log10(x+200)).to_list()
        y_list = df_subset[df_subset["Metric"]
                           == "F1 (max)"]["Value"].to_list()
        ax.scatter(y=y_list, x=x_list, color=color)

        add_linregress(ax, x_list, y_list, position="bottom",
                       color=color, showstats=False)

    ax.set_xlabel("Images")
    ax.set_ylabel("F1")
    # ax.legend([], [], frameon=False)

    plt.savefig(
        f"/home/wouter/Projects/PhD/Recognizability/Sanity combined.png", dpi=300)
    plt.close()


def pearson(path, x, y, order=None):
    df = pd.read_csv(os.path.join(path, "out.csv"))

    # if x == "img per obs":
    #     df = df[df[x] <= 2]

    if order is not None:
        df = df[df["Order"] == order]

    my_rho = np.corrcoef(df[x], df[y])
    print(f"{order}: {x} vs {y}: rho = {my_rho[0][1]}")


def img_per_obs(zip, path):
    df_media = pd.read_csv(zipfile.ZipFile(zip).open('multimedia.txt'), sep='\t',
                           error_bad_lines=False, usecols=["gbifID"])

    df_occ = pd.read_csv(zipfile.ZipFile(zip).open('verbatim.txt'), sep='\t',
                         error_bad_lines=False, usecols=["scientificName", "gbifID", "order"])

    df_media = df_media.merge(df_occ, how='left', on="gbifID")
    df_media = df_media[df_media["order"].isin(
        ["Anseriformes", "Charadriiformes", "Passeriformes"])]
    df_occ = df_occ[df_occ["order"].isin(
        ["Anseriformes", "Charadriiformes", "Passeriformes"])]

    img = pd.DataFrame(df_media.value_counts(
        subset=["scientificName"])).reset_index()
    img.columns = ["scientificName", "images"]

    obs = pd.DataFrame(df_occ.value_counts(
        subset=["scientificName"])).reset_index()
    obs.columns = ["scientificName", "observations"]

    df_stats = obs.merge(img, how='left', on="scientificName")
    df_stats["img per obs"] = df_stats["images"] / df_stats["observations"]

    print(df_stats.head())

    df_stats.to_csv(os.path.join(path, "img_per_obs.csv"))

    # df_media = pd.DataFrame(df_media.value_counts(subset=["gbifID"])).reset_index()
    # df_media.columns = ["gbifID", "count"]


def lasso(path, y, x, factor="Order", show_coef=False):
    df = pd.read_csv(os.path.join(path, "out.csv"))
    df.dropna(inplace=True)

    df = df[['Order', 'F1 (mean)', 'Images (log)', 'Habitat', 'HWI', 'Body mass (log)', 'Proportion AO vs TOVe', 'Proportion AO+img vs AO', 'Urban percentage', "info pixels (log)", "img per obs"]]

    dummies = pd.get_dummies(df[[factor]])
    df = pd.concat([df, dummies], axis=1)
    factors = dummies.columns[1:].to_list()

    y = df[y]
    x = df[ factors + x]

    columns = x.columns

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    model = LassoCV()
    model.fit(x, y)
    LassoCV()

    # print("Aplha:", model.alpha_)

    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(x, y)

    Lasso(alpha=model.alpha_)

    if show_coef:
        print(lasso_best)

        coef = list(zip(np.abs(lasso_best.coef_), columns))
        coef.sort()
        coef.reverse()

        for value, factor in coef:
            if value > 0:
                print(f"{factor}: {value}")

    print('R squared training set', round(
        lasso_best.score(x, y), 2))
    # mean_squared_error(y, lasso_best.predict(x))

    # Training data
    # pred_train = lasso_best.predict(x)
    # mse_train = mean_squared_error(y, pred_train)
    # print('MSE training set', round(mse_train, 2))

    # plt.semilogx(model.alphas_, model.mse_path_, ":")
    # plt.plot(
    #     model.alphas_,
    #     model.mse_path_.mean(axis=-1),
    #     "k",
    #     label="Average across the folds",
    #     linewidth=2,
    # )
    # plt.axvline(
    #     model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
    # )

    # plt.legend()
    # plt.xlabel("alphas")
    # plt.ylabel("Mean square error")
    # plt.title("Mean square error on each fold")
    # plt.axis("tight")
    # plt.savefig(os.path.join(path, f"Alpha.png"))
    # plt.close()


def regression_teveel(path, annotations_filename, stats_filename, characteristics_filename):
    df = pd.read_csv(os.path.join(path, stats_filename))
    df_char = pd.read_csv(os.path.join(path, characteristics_filename))

    df = pd.merge(df, df_char, left_on="Species", right_on="Scientific name")
    df.drop(['Scientific name', 'GBIF ID'], axis=1, inplace=True)
    df.dropna(inplace=True)

    df["Habitat - Semi-naturlig mark"] = df["Habitat - Semi-naturlig mark"].apply(
        lambda x: int(x))
    df["Habitat - Sterkt endret mark"] = df["Habitat - Sterkt endret mark"].apply(
        lambda x: int(x))
    df["Territoriality"] = df["Territoriality"].apply(
        lambda x: 0 if x == "none" else (1 if x == "strong" else .5))

    df['Images'] = df['Images'].apply(lambda x: np.log10(x))
    # df['Observations in TOVe'] = df['Observations in TOVe'].apply(lambda x: np.log10(x))
    # df['Observations in Artsobservasjoner'] = df['Observations in Artsobservasjoner'].apply(lambda x: np.log10(x))
    # df['Provinces'] = df['Provinces'].apply(lambda x: np.log10(x))
    df['Observations in Artsobservasjoner with images'] = df['Observations in Artsobservasjoner with images'].apply(
        lambda x: np.log10(x))

    for i, taxon in enumerate(["Anseriformes", "Charadriiformes", "Passeriformes"]):
        df_subset = df[df["Order"] == taxon]
        dummies = pd.get_dummies(df_subset[['Diet']])

        print("-----------", taxon, "------------")

        y = df_subset['F1 (max)']
        X_numerical = df_subset.drop(
            ['Diet', 'Order', 'Species', 'F1 (max)'], axis=1).astype('float64')
        list_numerical = X_numerical.columns
        X = pd.concat([X_numerical, dummies], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)

        scaler = StandardScaler().fit(X_train[list_numerical])
        X_train[list_numerical] = scaler.transform(X_train[list_numerical])
        X_test[list_numerical] = scaler.transform(X_test[list_numerical])

        # reg = Lasso(alpha=1)
        # reg.fit(X_train, y_train)

        # print('R squared training set', round(
        #     reg.score(X_train, y_train)*100, 2))
        # print('R squared test set', round(reg.score(X_test, y_test)*100, 2))
        # # Training data
        # pred_train = reg.predict(X_train)
        # mse_train = mean_squared_error(y_train, pred_train)
        # print('MSE training set', round(mse_train, 2))

        # # Test data
        # pred = reg.predict(X_test)
        # mse_test = mean_squared_error(y_test, pred)
        # print('MSE test set', round(mse_test, 2))

        alphas = np.linspace(0.01, 500, 100)
        lasso = Lasso(max_iter=1000000)
        coefs = []

        for a in alphas:
            lasso.set_params(alpha=a)
            lasso.fit(X_train, y_train)
            coefs.append(lasso.coef_)

        ax = plt.gca()

        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.axis('tight')
        plt.xlabel('alpha')
        plt.ylabel('Standardized Coefficients')
        plt.title('Lasso coefficients as a function of alpha')

        plt.savefig(os.path.join(path, f"Lasso {taxon}.png"))
        plt.close()

        # Lasso with 5 fold cross-validation
        model = LassoCV(cv=5, random_state=0, max_iter=1000000)

        # Fit model
        model.fit(X_train, y_train)

        LassoCV(cv=5, max_iter=1000000, random_state=0)

        print("Aplha:", model.alpha_)

        lasso_best = Lasso(alpha=model.alpha_)
        lasso_best.fit(X_train, y_train)

        Lasso(alpha=model.alpha_)
        print(list(zip(lasso_best.coef_, X)))

        print('R squared training set', round(
            lasso_best.score(X_train, y_train)*100, 2))
        print('R squared test set', round(
            lasso_best.score(X_test, y_test)*100, 2))
        mean_squared_error(y_test, lasso_best.predict(X_test))

        # Training data
        pred_train = lasso_best.predict(X_train)
        mse_train = mean_squared_error(y_train, pred_train)
        print('MSE training set', round(mse_train, 2))

        # Test data
        pred = lasso_best.predict(X_test)
        mse_test = mean_squared_error(y_test, pred)
        print('MSE test set', round(mse_test, 2))

        plt.semilogx(model.alphas_, model.mse_path_, ":")
        plt.plot(
            model.alphas_,
            model.mse_path_.mean(axis=-1),
            "k",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(
            model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
        )

        plt.legend()
        plt.xlabel("alphas")
        plt.ylabel("Mean square error")
        plt.title("Mean square error on each fold")
        plt.axis("tight")
        plt.savefig(os.path.join(path, f"Alpha {taxon}.png"))
        plt.close()


def regression_example(path, annotations_filename, stats_filename, characteristics_filename):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/kirenz/datasets/master/Hitters.csv")
    df = df.dropna()
    dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
    y = df['Salary']
    X_numerical = df.drop(
        ['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

    list_numerical = X_numerical.columns

    X = pd.concat(
        [X_numerical, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=10)

    scaler = StandardScaler().fit(X_train[list_numerical])
    X_train[list_numerical] = scaler.transform(X_train[list_numerical])
    X_test[list_numerical] = scaler.transform(X_test[list_numerical])

    reg = Lasso(alpha=1)
    reg.fit(X_train, y_train)

    print('R squared training set', round(reg.score(X_train, y_train)*100, 2))
    print('R squared test set', round(reg.score(X_test, y_test)*100, 2))

    # Training data
    pred_train = reg.predict(X_train)
    mse_train = mean_squared_error(y_train, pred_train)
    print('MSE training set', round(mse_train, 2))

    # Test data
    pred = reg.predict(X_test)
    mse_test = mean_squared_error(y_test, pred)
    print('MSE test set', round(mse_test, 2))

    alphas = np.linspace(0.01, 500, 100)
    lasso = Lasso(max_iter=1000000)
    coefs = []

    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_train, y_train)
        coefs.append(lasso.coef_)

    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('Standardized Coefficients')
    plt.title('Lasso coefficients as a function of alpha')

    plt.savefig(os.path.join(path, "Lasso.png"))

    # Lasso with 5 fold cross-validation
    model = LassoCV(cv=5, random_state=0, max_iter=1000000)

    # Fit model
    model.fit(X_train, y_train)

    LassoCV(cv=5, max_iter=10000, random_state=0)

    print(model.alpha_)

    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(X_train, y_train)

    Lasso(alpha=2.3441244939374593)
    print(list(zip(lasso_best.coef_, X)))

    print('R squared training set', round(
        lasso_best.score(X_train, y_train)*100, 2))
    print('R squared test set', round(lasso_best.score(X_test, y_test)*100, 2))
    mean_squared_error(y_test, lasso_best.predict(X_test))

    plt.semilogx(model.alphas_, model.mse_path_, ":")
    plt.plot(
        model.alphas_,
        model.mse_path_.mean(axis=-1),
        "k",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(
        model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
    )

    plt.legend()
    plt.xlabel("alphas")
    plt.ylabel("Mean square error")
    plt.title("Mean square error on each fold")
    plt.axis("tight")

    ymin, ymax = 50000, 250000
    plt.ylim(ymin, ymax)


def regression_old(path, annotations_filename, stats_filename, characteristics_filename):
    df = pd.read_csv(os.path.join(path, stats_filename))
    # df_ann = pd.read_csv(os.path.join(path, annotations_filename))
    df_char = pd.read_csv(os.path.join(path, characteristics_filename))

    #df = pd.merge(df, df_ann, left_on="Species", right_on="species")
    df = pd.merge(df, df_char, left_on="Species", right_on="Scientific name")
    df.drop(['Scientific name', 'GBIF ID'], axis=1, inplace=True)
    df.dropna(inplace=True)

    df["Habitat - Semi-naturlig mark"] = df["Habitat - Semi-naturlig mark"].apply(
        lambda x: int(x))
    df["Habitat - Sterkt endret mark"] = df["Habitat - Sterkt endret mark"].apply(
        lambda x: int(x))
    df["Territoriality"] = df["Territoriality"].apply(
        lambda x: 0 if x == "none" else (1 if x == "strong" else .5))

    diets = pd.get_dummies(df['Diet'], drop_first=True)
    df = pd.concat([df, diets], axis=1)
    df.drop(['Diet'], axis=1, inplace=True)

    toscale = ['a', 'k', 'F1 (max)', 'Images', 'Observations in TOVe', 'Observations in Artsobservasjoner', 'Habitat - Sterkt endret mark', 'Habitat - Semi-naturlig mark', 'Provinces', 'HWI', 'Body mass (log)', 'Island', 'Territoriality', 'Habitat',
               'Proportion of Artsobservasjoner observations', 'Observations in Artsobservasjoner with images', 'Proportion AO vs TOVe', 'Proportion AO+img vs AO', 'Proportion AO+img vs TOVe', 'Urban percentage', 'invertebrates', 'omnivore', 'plants', 'seeds', 'vertebrates']

    scaler = MinMaxScaler()
    df[toscale] = scaler.fit_transform(df[toscale])

    df.rename(columns={"Habitat": "Habitat openness"}, inplace=True)

    df.to_csv(os.path.join(path, "scaled.csv"))

    for i, taxon in enumerate(["Anseriformes", "Charadriiformes", "Passeriformes"]):
        df_order = df[df["Order"] == taxon]
        df_order.drop(['Order', "Species"], axis=1, inplace=True)

        np.random.seed(0)
        df_train, df_test = train_test_split(
            df_order, train_size=0.7, test_size=0.3, random_state=100)

        y_train = df_train.pop('F1 (max)')
        X_train = df_train

        X_train_lm = sm.add_constant(X_train)

        lr_1 = sm.OLS(y_train, X_train_lm).fit()

        lr_1.summary()

        exit(0)

        lm = LinearRegression()
        lm.fit(X_train, y_train)

        rfe = RFE(lm, n_features_to_select=4)
        rfe = rfe.fit(X_train, y_train)

        print(list(zip(X_train.columns, rfe.support_, rfe.ranking_)))

        # Creating X_test dataframe with RFE selected variables
        X_train_rfe = X_train[col]

        # Adding a constant variable
        import statsmodels.api as sm
        X_train_rfe = sm.add_constant(X_train_rfe)

        lm = sm.OLS(y_train, X_train_rfe).fit()   # Running the linear model

        print(lm.summary())

        exit(0)
