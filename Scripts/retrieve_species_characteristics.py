import zipfile
import pandas as pd
import os
import requests
import json
from bs4 import BeautifulSoup
import numpy as np
import Scripts.esacci as esa


def count_species(csv_file, dwca, order):
    df = pd.read_csv(csv_file, encoding='cp1252', sep=';')
    df = df[df["Orden"].isin(order)]
    df = df[df["FinnesINorge"] == "Ja"]

    print(len(df["Art"].unique()), "species in Norway")

    columns = ['scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    df_verbatim = pd.read_csv(zipfile.ZipFile(dwca).open('verbatim.txt'), sep='\t', usecols=columns,
                              error_bad_lines=False)

    df_verbatim = df_verbatim[df_verbatim["order"].isin(order)]

    df_counts = pd.DataFrame(df_verbatim.value_counts(subset=columns)).reset_index()
    df_counts.columns = columns + ['count']
    df_counts = df_counts[df_counts['count'] >= 220]

    print(len(df_counts), "species with at least 220 observations with images\n\n")

    print(len(df_counts) / len(df["Art"].unique()) * 100, "%")




def get_urbanness(gbif_id, samplesize, esafile, height, width):
    url = requests.get(
        f"https://www.gbif.org/api/occurrence/search?dataset_key=b124e1e0-4755-430f-9eab-894f25a9b59c&media_type=StillImage&taxon_key={gbif_id}&limit={samplesize}")

    count = 0
    for result in json.loads(url.text)["results"]:
        if esa.get_from_file(result["decimalLatitude"], result["decimalLongitude"], esafile, height, width) == 190:
            count += 1
    return count / samplesize


def retrieve_landcover(df, samplesize=100, force=False):

    esafile, width, height = esa.load_file(
        "/home/wouter/Projects/PhD/datasets/ESACCI/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc")

    df[f"Urban percentage"] = df.apply(lambda row: row[f"Urban percentage"] if f"Urban percentage" in df.columns and not pd.isna(row[f"Urban percentage"])
                                       and not force else get_urbanness(row["GBIF ID"], samplesize, esafile, height, width), axis=1)

    return df


def retrieve_trait(df, species, trait):
    # Some bird species' scientific names differ. This maps from GBIF to Global_DWI
    # Only species we needed, so check the output and/or implement a different lookup method

    mapping = {
        # Apparently regarded as a subspecies by Global_DWI
        "Acanthis cabaret": "Carduelis flammea",
        "Acanthis flammea": "Carduelis flammea",
        "Acanthis hornemanni": "Carduelis hornemanni",
        "Chloris chloris": "Carduelis chloris",
        "Curruca communis": "Sylvia communis",
    }

    if species in df["Species name"].to_list():
        return df[df["Species name"] == species][trait].values[0]
    elif species in df["IUCN name"].to_list():
        return df[df["IUCN name"] == species][trait].values[0]
    elif species in mapping.keys():
        return df[df["Species name"] == mapping[species]][trait].values[0]

    return np.nan


def get_traits(df, source_file):
    traits = pd.read_csv(source_file)

    for trait in ["HWI", "Body mass (log)", "Island", "Territoriality", "Diet", "Habitat"]:
        df[trait] = df["Scientific name"].apply(
            lambda x: retrieve_trait(traits, x, trait))

    return df


def get_habitats_and_provinces(name, habitats):
    url = requests.get(
        f"https://artsdatabanken.no/api/Taxon/ScientificName?scientificName={name}")

    active_habitats = []
    accepted_name = False
    provinces = np.nan

    for taxon in json.loads(url.text):
        if taxon["taxonomicStatus"] != "accepted":
            if taxon["acceptedNameUsage"]:
                accepted_name = taxon["acceptedNameUsage"]["scientificName"]
            continue
        for result in taxon["dynamicProperties"]:
            if result["Properties"][0]["Value"] == "Rødliste 2021" and result["Properties"][3]["Value"] == "Norge":
                page = requests.get(result["Properties"][-1]["Value"])
                soup = BeautifulSoup(page.content, 'html.parser').select(
                    ".habitat_container .active b")
                active_habitats = [h.get_text() for h in soup]
                soup = BeautifulSoup(page.content, 'html.parser').select(
                    ".regionlist .indicator.known, .regionlist .indicator.presumed")
                provinces = len(soup)
                break

    if len(active_habitats) == 0:
        if accepted_name:
            return get_habitats_and_provinces(accepted_name, habitats)
        results = dict(zip(habitats, [np.nan for h in habitats]))
    else:
        bools = [h in active_habitats for h in habitats]
        results = dict(zip(habitats, bools))

    results["provinces"] = provinces
    return results


def add_habitats_and_provinces(df, habitats, force):
    df[f"Habitats"] = df.apply(lambda row: row[f"Habitat - {habitats[0]}"] if f"Habitat - {habitats[0]}" in df.columns and not pd.isna(row[f"Habitat - {habitats[0]}"])
                               and not force else get_habitats_and_provinces(row["Scientific name"], habitats), axis=1)

    for habitat in habitats:
        df[f"Habitat - {habitat}"] = df.apply(
            lambda row: row["Habitats"][habitat], axis=1)

    df[f"Provinces"] = df.apply(
        lambda row: row["Habitats"]["provinces"], axis=1)

    return df.drop(columns=["Habitats"])


def get_count_sum(row, datasets):
    sum = 0
    for dataset in datasets:
        sum += int(row[f"Observations in {dataset}"])
    
    if sum == 0:
        print(row)
    
    return sum


def add_proportion(df, dataset_name, datasets, column_name):
    df[column_name] = df.apply(
        lambda row: int(row[f"Observations in {dataset_name}"]) / get_count_sum(row, datasets), axis=1)
    return df


def add_counts(df, dataset_id, dataset_name, force):
    df[f"Observations in {dataset_name}"] = df.apply(lambda row: row[f"Observations in {dataset_name}"] if f"Observations in {dataset_name}" in df.columns and not pd.isna(row[f"Observations in {dataset_name}"])
                                                     and not force else requests.get(f"https://api.gbif.org/v1/occurrence/count?datasetKey={dataset_id}&taxonKey={row['GBIF ID']}").text, axis=1)
    return df


def add_counts_with_img(df, dataset_id, dataset_name, force):
    df[f"Observations in {dataset_name} with images"] = df.apply(lambda row: row[f"Observations in {dataset_name} with images"] if f"Observations in {dataset_name} with images" in df.columns and not pd.isna(row[f"Observations in {dataset_name} with images"])
                                                                 and not force else requests.get(f"https://api.gbif.org/v1/occurrence/search?datasetKey={dataset_id}&taxonKey={row['GBIF ID']}&mediaType=StillImage&limit=0").json()["count"], axis=1)
    return df


def retrieve_gbif_id(name):
    url = requests.get(f"https://api.gbif.org/v1/species/suggest?q={name}")

    for result in json.loads(url.text):
        if "canonicalName" in result and "speciesKey" in result:
                return result["speciesKey"]
    print(f"GBIF key for {name} not found!")
    exit(1)


def add_gbif_ids(df, force):
    df["GBIF ID"] = df.apply(lambda row: row["GBIF ID"] if "GBIF ID" in df.columns and not pd.isna(row["GBIF ID"])
                             and not force else retrieve_gbif_id(row["Scientific name"]), axis=1)
    return df


def add_names(df, scientific_names):
    for name in scientific_names:
        if name not in df["Scientific name"].to_list():
            df = df.append({"Scientific name": name}, ignore_index=True)
    return df


def retrieve(scientific_names, force=False):
    if not os.path.isfile(os.path.join(os.environ["STORAGE_DIR"], "Species characteristics.csv")):
        df = pd.DataFrame(columns=["Scientific name"])
    else:
        df = pd.read_csv(os.path.join(
            os.environ["STORAGE_DIR"], "Species characteristics.csv"))

    df = add_names(df, scientific_names)

    df = add_gbif_ids(df, force)
    df = add_counts(df, "4a00502d-6342-4294-aad1-9727e5c24041", "TOVe", force)
    df = add_counts(df, "b124e1e0-4755-430f-9eab-894f25a9b59c",
                    "Artsobservasjoner", force)
    df = add_counts_with_img(df, "b124e1e0-4755-430f-9eab-894f25a9b59c",
                             "Artsobservasjoner", force)

    df = add_proportion(df, "Artsobservasjoner", [
                        "Artsobservasjoner", "TOVe"], column_name="Proportion AO vs TOVe")
    df = add_proportion(df, "Artsobservasjoner with images", [
                        "Artsobservasjoner"], column_name="Proportion AO+img vs AO")
    df = add_proportion(df, "Artsobservasjoner with images", [
                        "Artsobservasjoner with images", "TOVe"], column_name="Proportion AO+img vs TOVe")

    df.to_csv(os.path.join(
        os.environ["STORAGE_DIR"], "Species characteristics.csv"), index=False)

    df = retrieve_landcover(df, samplesize=100)
    df.to_csv(os.path.join(
        os.environ["STORAGE_DIR"], "Species characteristics.csv"), index=False)

    # df = add_habitats_and_provinces(
    #     df, ["Sterkt endret mark", "Semi-naturlig mark"], force)

    # df.to_csv(os.path.join(
    #     os.environ["STORAGE_DIR"], "Species characteristics.csv"), index=False)

    df = get_traits(df, os.path.join(
        os.environ["STORAGE_DIR"], "Global-HWI.csv"))

    df.to_csv(os.path.join(
        os.environ["STORAGE_DIR"], "Species characteristics.csv"), index=False)


if __name__ == "__main__":
    print("USAGE")
    print("retrieve(scientific_names, force=False): Add missing stats for each species in the provided list to the 'species_characteristics.csv' file in STORAGE_DIR. Retrieves all if force=True")
