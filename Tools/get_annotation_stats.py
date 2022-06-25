import socket
import os
import json
import requests
import pandas as pd
import sys


timeout = 60
socket.setdefaulttimeout(timeout)


def get_annotation_stats(folder, api, token, pixels=299):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {token}',
        'charset': 'utf-8'
    }

    pics = []

    response = requests.get(f'{api}/api/projects',
                            headers=headers, params={"page_size": 999})

    projects = map(lambda x: {"id": x["id"], "species": x["description"], "annotations": x["num_tasks_with_annotations"]},
                   filter(lambda x: (x["num_tasks_with_annotations"] > 0) and (x["id"] > 1),
                          json.loads(response.content)["results"]))

    for project in projects:
        species = project["species"]

        response = requests.get(
            f"{api}/api/projects/{project['id']}/tasks/", headers=headers)
        tasks = json.loads(response.content)

        for task in tasks:
            if len(task["annotations"]) > 1:
                print("Task has more than one annotation!")
                print(f"{api}/projects/{project['id']}/data?task={task['id']}")

            tags = []

            if len(task["annotations"]) == 0:
                print(f"{api}/projects/{project['id']}/data?task={task['id']}", "has no annotation")
                continue

            cropped_pixels = 0

            annotation = task["annotations"][0]

            for ind in annotation["result"]:

                tags += [{
                    "Label": ind["value"]["rectanglelabels"][0],
                    "Target": ind["value"]["rectanglelabels"][0] != "Annen fugl",
                    "Percentage": ind["value"]["width"] * ind["value"]["height"] * .01,
                    "Pixels": (ind["value"]["width"] * .01 * ind["value"]["height"] * .01) * (int(ind["original_width"] * ind["original_height"])),
                    "Cropped pixels": (ind["value"]["width"] * .01 * ind["value"]["height"] * .01) * (int(ind["original_width"] * ind["original_height"])),
                }]
                cropped_pixels = min(
                    pixels, int(ind["original_width"])) * (min(pixels, int(ind["original_height"])))

            targets = list(filter(lambda x: x["Target"], tags))
            non_targets = list(filter(lambda x: not x["Target"], tags))

            pics += [{
                "species": species,
                "task": f"{api}/projects/{project['id']}/data?task={task['id']}",
                "tags": tags,
                "num_targets": len(targets),
                "num_nontargets": len(non_targets),
                "total_target_percentage": sum(map(lambda x: x["Percentage"], targets)),
                "avg_target_percentage": sum(map(lambda x: x["Percentage"], targets)) / len(targets) if len(targets) > 0 else 0,
                "max_target_percentage": max(map(lambda x: x["Percentage"], targets)) if len(targets) > 0 else 0,
                "total_target_pixels": sum(map(lambda x: x["Pixels"], targets)),
                "avg_target_pixels": sum(map(lambda x: x["Pixels"], targets)) / len(targets) if len(targets) > 0 else 0,
                "max_target_pixels": max(map(lambda x: x["Pixels"], targets)) if len(targets) > 0 else 0,
                "total_nontarget_percentage": sum(map(lambda x: x["Percentage"], non_targets)),
                "avg_nontarget_percentage": sum(map(lambda x: x["Percentage"], non_targets)) / len(non_targets) if len(non_targets) > 0 else 0,
                "max_nontarget_percentage": max(map(lambda x: x["Percentage"], non_targets)) if len(non_targets) > 0 else 0,
                "total_nontarget_pixels": sum(map(lambda x: x["Pixels"], non_targets)),
                "cropped_pixels": sum(map(lambda x: x["Pixels"], non_targets)),
                "avg_nontarget_pixels": sum(map(lambda x: x["Pixels"], non_targets)) / len(non_targets) if len(non_targets) > 0 else 0,
                "max_nontarget_pixels": max(map(lambda x: x["Pixels"], non_targets)) if len(non_targets) > 0 else 0,
            }]

    df = pd.DataFrame(pics)

    df["informative_percentage"] = df.apply(
        lambda row: row["total_target_percentage"] - row["total_nontarget_percentage"], axis=1)
    df["target_individuals_percent"] = df.apply(lambda row: 100 * (row["num_targets"] / (
        row["num_targets"] + row["num_nontargets"]) if (row["num_targets"] + row["num_nontargets"]) > 0 else 0), axis=1)
    df["target_area_percent"] = df.apply(lambda row: 100 * (row["total_target_pixels"] / (row["total_target_pixels"] +
                                         row["total_nontarget_pixels"]) if (row["total_target_pixels"] + row["total_nontarget_pixels"]) > 0 else 0), axis=1)
    df["max_target_info_percentage"] = df.apply(
        lambda row: row["max_target_percentage"] - row["total_nontarget_percentage"], axis=1)
    df["max_target_info_pixels"] = df.apply(
        lambda row: row["max_target_pixels"] - row["total_nontarget_pixels"], axis=1)
    df["max_target_info_cropped_pixels"] = df.apply(lambda row: (
        row["max_target_percentage"] - row["total_nontarget_percentage"]) * cropped_pixels, axis=1)
    df["max_target_cropped_pixels"] = df.apply(lambda row: (
        row["max_target_percentage"]) * cropped_pixels, axis=1)
    df["max_nontarget_cropped_pixels"] = df.apply(lambda row: (
        row["max_nontarget_percentage"]) * cropped_pixels, axis=1)
    df["total_target_cropped_pixels"] = df.apply(lambda row: (
        row["total_target_percentage"]) * cropped_pixels, axis=1)
    df["total_nontarget_cropped_pixels"] = df.apply(lambda row: (
        row["total_nontarget_percentage"]) * cropped_pixels, axis=1)

    df["cropped_target_big_enough"] = df.apply(
        lambda row: 1 if row["max_target_cropped_pixels"] > 400 else 0, axis=1)
    df["cropped_informative_enough"] = df.apply(
        lambda row: 1 if row["max_target_info_cropped_pixels"] > 400 else 0, axis=1)



    df.to_csv(os.path.join(folder, "Annotations.csv"), index=False)
    df.groupby('species').mean().to_csv(
        os.path.join(folder, "Annotation stats (mean).csv"))
    df.groupby('species').median().to_csv(
        os.path.join(folder, "Annotation stats (median).csv"))


if __name__ == "__main__":
    print("USAGE")
    print("get_annotation_stats(folder, api, token)")
