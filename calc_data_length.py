import os
import json
import pandas as pd
from statistics import mean
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


TOKENIZER_PATH = "/mnt/data1/TC/TextClassDemo/llama3.1-8b"


JSON_FILES = [
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/Biomedical_Test.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/Biomedical_Train.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/CR_Test.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/CR_Train.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/dblp_Test.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/dblp_Train.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/ohsumed_Test.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/ohsumed_Train.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/R52_Test.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/R52_Train.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/TREC_Test.json",
    "/mnt/data1/TC/TextClassDemo/data/data_original/data/TREC_Train.json"
]


OUTPUT_DIR = "/mnt/data1/TC/TextClassDemo/data/data_original/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "stat_token_lengths_llama3.csv")


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


stats = []
all_lengths = []

for file_path in JSON_FILES:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lengths = []
    dataset_name = os.path.basename(file_path).replace(".json", "")
    dataset = dataset_name.replace("_Train", "").replace("_Test", "")
    split = "train" if "Train" in dataset_name else "test"

    for item in data:
        text = item.get("text")
        if text:
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            l = len(token_ids)
            lengths.append(l)
            all_lengths.append({
                "dataset": dataset,
                "split": split,
                "length": l
            })

    if lengths:
        stats.append({
            "dataset": dataset,
            "split": split,
            "max_len": max(lengths),
            "min_len": min(lengths),
            "avg_len": round(mean(lengths), 2)
        })


df_stats = pd.DataFrame(stats)
df_stats.to_csv(CSV_PATH, index=False)


sns.set(style="whitegrid")


agg_df = df_stats.groupby("dataset").agg({"max_len": "max", "min_len": "min"}).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x="dataset", y="max_len", data=agg_df, color='skyblue', label="Max")
sns.barplot(x="dataset", y="min_len", data=agg_df, color='lightgreen', label="Min")
plt.title("Max/Min Token Length per Dataset (Train+Test Combined)")
plt.ylabel("Token Length")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_combined_max_min.png"))
plt.close()


plt.figure(figsize=(10, 6))
sns.barplot(x="dataset", y="max_len", hue="split", data=df_stats)
plt.title("Max Token Length per Dataset and Split")
plt.ylabel("Token Length")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_split_max.png"))
plt.close()


plt.figure(figsize=(10, 6))
sns.barplot(x="dataset", y="min_len", hue="split", data=df_stats)
plt.title("Min Token Length per Dataset and Split")
plt.ylabel("Token Length")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_split_min.png"))
plt.close()


plt.figure(figsize=(10, 6))
sns.barplot(x="dataset", y="avg_len", hue="split", data=df_stats)
plt.title("Average Token Length per Dataset and Split")
plt.ylabel("Token Length")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_split_avg.png"))
plt.close()


df_lengths = pd.DataFrame(all_lengths)
plt.figure(figsize=(12, 6))
sns.boxplot(x="dataset", y="length", hue="split", data=df_lengths)
plt.title("Token Length Distribution (Boxplot)")
plt.ylabel("Token Length")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_boxplot.png"))
plt.close()

print("✅ Done. CSV and all 5 plots saved to:", OUTPUT_DIR)
