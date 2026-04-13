import json
import os
import random
import time
import requests
from dotenv import load_dotenv
import re
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def accuracy(path: str):
    with open("dataset/Group.json") as f:
        benchmark = json.load(f)
    with open(path) as f:
        evals = json.load(f)
    
    correct_evals = {}
    for doc in benchmark:
        correct_evals[doc["sim_id"]] = doc["evaluation"]

    n_correct = 0
    n_total = 0
    for eval in evals:
        correct = correct_evals[eval["sim_id"]]
        for section, v in eval["evaluation"].items():
            for feature, grade in v["features"].items():
                if grade == correct[section]["features"][feature]:
                    n_correct += 1
                n_total += 1
    
    print(f"\nACCURACY: {n_correct}/{n_total} --> {n_correct/n_total}")


def accuracy_by_bin(path: str):
    N_BINS = 3

    with open("dataset/Group.json") as f:
        benchmark = json.load(f)
    with open(path) as f:
        evals = json.load(f)

    correct_evals = {}
    for doc in benchmark:
        correct_evals[doc["sim_id"]] = doc["evaluation"]

    metric = path.split("_")[-1].split(".")[0]
    bin_start = 0.5
    if metric == "msp":
        bin_start = 0.0
    bins = np.linspace(bin_start, 1.0, N_BINS+1)

    n_correct = {n+1: 0 for n in range(N_BINS)}
    n_total = {n+1: 0 for n in range(N_BINS)}
    bin_confs = {n+1: [] for n in range(N_BINS)}
    for eval in evals:
        correct = correct_evals[eval["sim_id"]]
        for section, v in eval["evaluation"].items():
            for feature, grade in v["features"].items():
                bin = np.digitize([v["confidence"][feature]], bins)[0]
                bin = np.clip(bin, 1, len(bins) - 1) # clamp to fix zzzz
                bin_confs[bin].append(v["confidence"][feature])
                if grade == correct[section]["features"][feature]:
                    n_correct[bin] += 1
                n_total[bin] += 1

    print(f"\nAccuracy by bin:")
    bar_labels, bar_heights, bar_annotations = [], [], []
    for b in range(1, N_BINS+1):
        if n_total[b] == 0:
            print(f"Bin {b} ({bins[b-1]:.2f} to {bins[b]:.2f}): {n_correct[b]}/{n_total[b]} --> N/A")
            continue
        acc = n_correct[b] / n_total[b]
        print(f"Bin {b} ({bins[b-1]:.2f} to {bins[b]:.2f}): {n_correct[b]}/{n_total[b]} --> {acc}")
        bar_labels.append(f"{bins[b-1]:.2f}–{bins[b]:.2f}")
        bar_heights.append(acc)
        bar_annotations.append(f"{n_correct[b]}/{n_total[b]}")

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(bar_labels, bar_heights)
    for bar, annotation in zip(bars, bar_annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            annotation,
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_xlabel("Confidence Interval (bin)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric.upper()} Accuracy vs Confidence", fontsize=14)
    plt.tight_layout()
    plt.show()


def ECE(path: str):
    N_BINS = 3

    with open("dataset/Group.json") as f:
        benchmark = json.load(f)
    with open(path) as f:
        evals = json.load(f)

    correct_evals = {}
    for doc in benchmark:
        correct_evals[doc["sim_id"]] = doc["evaluation"]

    metric = path.split("_")[-1].split(".")[0]
    bin_start = 0.5
    if metric == "msp":
        bin_start = 0.0
    bins = np.linspace(bin_start, 1.0, N_BINS+1)

    n_correct = {n+1: 0 for n in range(N_BINS)}
    n_total = {n+1: 0 for n in range(N_BINS)}
    bin_confs = {n+1: [] for n in range(N_BINS)}
    for eval in evals:
        correct = correct_evals[eval["sim_id"]]
        for section, v in eval["evaluation"].items():
            for feature, grade in v["features"].items():
                bin = np.digitize([v["confidence"][feature]], bins)[0]
                bin = np.clip(bin, 1, len(bins) - 1) # clamp to fix zzzz
                bin_confs[bin].append(v["confidence"][feature])
                if grade == correct[section]["features"][feature]:
                    n_correct[bin] += 1
                n_total[bin] += 1

    print(f"\nAccuracy by bin:")
    ece = 0
    for b in range(1, N_BINS+1):
        if n_total[b] == 0:
            continue
        print(f"Bin {b} ({bins[b-1]:.2f} to {bins[b]:.2f}): {n_correct[b]}/{n_total[b]} --> {n_correct[b]/n_total[b]}")
        ece += (n_total[b]/720) * abs((n_correct[b]/n_total[b]) - np.mean(bin_confs[b]))
    print(f"\nECE: {ece}")


def AUROC(path: str):
    with open("dataset/Group.json") as f:
        benchmark = json.load(f)
    with open(path) as f:
        evals = json.load(f)

    correct_evals = {}
    for doc in benchmark:
        correct_evals[doc["sim_id"]] = doc["evaluation"]

    labels, confs = [], []
    for eval in evals:
        correct = correct_evals[eval["sim_id"]]
        for section, v in eval["evaluation"].items():
            for feature, grade in v["features"].items():
                confs.append(v["confidence"][feature])
                labels.append(int(grade == correct[section]["features"][feature]))

    auroc = roc_auc_score(labels, confs)
    print(f"\nAUROC: {auroc}")


if __name__ == "__main__":
    paths = [
        "results/fireworks_gpt-oss-120b_vce.json",
        "results/fireworks_gpt-oss-120b_msp.json",
        "results/fireworks_gpt-oss-120b_sc.json"
        ]
    for path in paths:
        print(f"\n\n---------- {path} ----------")
        accuracy_by_bin(path)