import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import nltk
from nltk.metrics.agreement import AnnotationTask

df = pd.read_csv("data/annotations_and_llmasajudge.csv")
df_human = df[df["annotator_group"].isin(["finnish", "international"])].copy()
df_llm = df[df["annotator_group"] == "-"].copy()

# Distinguish Topic words vs Random words
topic_words = [
    "company",
    "euros",
    "age",
    "health",
    "rydman",
    "border",
    "win",
    "medal",
    "court",
    "president",
    "government",
    "tax",
    "venezuela",
    "agency",
    "year",
    "parliamentary",
    "temperatures",
    "january",
    "kivimki",
    "retirement",
    "people",
    "food",
    "school",
    "areas",
    "police",
    "olympics",
    "language",
    "chance",
    "purra",
    "posti",
    "positions",
    "social",
    "students",
    "finlands",
    "country",
    "finland",
    "prices",
    "umk",
    "employee",
    "entry",
    "minister",
    "wolt",
    "employees",
    "test",
    "finns",
    "finnish",
    "juuso",
    "contest",
    "party",
    "song",
]
df_human["word_type"] = df_human["word"].apply(
    lambda x: "Topic (News)" if x in topic_words else "Random"
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})


# Plot1：RQ1-Are the produced generations actually funny and political?
# Here we take average scores for funny and political across 6 annotators,
# plot barplots with the distribution of both funny and political and conclude whether they are or not.


# Caculate and print average scores for funny and political across 6 annotators
avg_funny = df_human["funny"].mean()
avg_political = df_human["political"].mean()
print(f"\n--- RQ1 Data ---")
print(f"Average Funny Score: {avg_funny:.2f}")
print(f"Average Political Score: {avg_political:.2f}")

plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
sns.countplot(
    data=df_human, x="funny", hue="funny", palette="Blues", ax=ax1, legend=False
)
plt.title("Distribution of Humor Scores")
for container in ax1.containers:
    ax1.bar_label(container, fmt="%d", padding=3)

ax2 = plt.subplot(1, 2, 2)
sns.countplot(
    data=df_human, x="political", hue="political", palette="Reds", ax=ax2, legend=False
)
plt.title("Distribution of Political Scores")
for container in ax2.containers:
    ax2.bar_label(container, fmt="%d", padding=3)

plt.tight_layout()
plt.savefig("data/Plot_1_Distributions_Labeled.pdf", dpi=300)
plt.close()


# Plot 2: RQ2 - Is satire cultural?
# Here we compute inter annotator agreement for all of us vs. finnish vs. international,
# and we can also report averages separatly.


# Caculate and print Inter-Annotator Agreement
def calc_alpha(data, col):
    task_data = list(zip(data["annotator_id"], data["item_id"], data[col]))
    try:
        return AnnotationTask(data=task_data).alpha()
    except:
        return float("nan")


print(f"\n--- RQ2 Data: Inter-Annotator Agreement (Krippendorff's Alpha) ---")
print(f"All of us (Funny): {calc_alpha(df_human, 'funny'):.3f}")
print(
    f"Finnish (Funny): {calc_alpha(df_human[df_human['annotator_group'] == 'finnish'], 'funny'):.3f}"
)
print(
    f"International (Funny): {calc_alpha(df_human[df_human['annotator_group'] == 'international'], 'funny'):.3f}"
)

plt.figure(figsize=(7, 5))
df_melt_cult = df_human.melt(
    id_vars=["annotator_group"],
    value_vars=["funny", "political"],
    var_name="Metric",
    value_name="Score",
)

ax3 = sns.barplot(
    data=df_melt_cult,
    x="annotator_group",
    y="Score",
    hue="Metric",
    errorbar=None,
    palette="Set2",
)
plt.title("Scores by Cultural Group (Finnish vs International)")
plt.ylim(1, 4)
for container in ax3.containers:
    ax3.bar_label(container, fmt="%.2f", padding=3)
plt.savefig("data/Plot_2_Cultural_Differences_Labeled.pdf", dpi=300)
plt.close()


# Plot 3: RQ4 - Is word candidate selection relevant?
# Same as in 3 but with random words vs. topic modeled words

word_stats = df_human.groupby("word_type")[["funny", "political"]].mean()
print(f"\n--- RQ4: Word Candidate Selection (Averages) ---")
print(word_stats.round(2).to_string())

plt.figure(figsize=(7, 5))
df_melt_word = df_human.melt(
    id_vars=["word_type"],
    value_vars=["funny", "political"],
    var_name="Metric",
    value_name="Score",
)
ax4 = sns.barplot(
    data=df_melt_word,
    x="word_type",
    y="Score",
    hue="Metric",
    errorbar=None,
    palette="Set3",
)
plt.title("Word Candidate Selection: Topic vs Random")
plt.ylim(1, 4)
for container in ax4.containers:
    ax4.bar_label(container, fmt="%.2f", padding=3)
plt.savefig("data/Plot_3_Word_Selection_Labeled.pdf", dpi=300)
plt.close()


# Plot5: RQ3 - Is retrieval improving the satire?
# Again, we take averaged scores for all annotators and compute averages for rag vs non rag

rag_stats = df_human.groupby("rag")[["funny", "political"]].mean()
print(f"\n--- RQ3: RAG vs Non-RAG (Averages) ---")
print(rag_stats.round(2).to_string())

plt.figure(figsize=(7, 5))
df_melt_rag = df_human.melt(
    id_vars=["rag"],
    value_vars=["funny", "political"],
    var_name="Metric",
    value_name="Score",
)
ax5 = sns.barplot(
    data=df_melt_rag, x="rag", y="Score", hue="Metric", errorbar=None, palette="Pastel1"
)
plt.title("RAG vs Non-RAG Average Scores")
plt.xticks([0, 1], ["Non-RAG (0)", "RAG (1)"])
plt.ylim(1, 4)
for container in ax5.containers:
    ax5.bar_label(container, fmt="%.2f", padding=3)
plt.savefig("data/Plot_5_RAG_vs_NonRAG_Labeled.pdf", dpi=300)
plt.close()


# Plot 4: RQ5 How good are LLMs are at evaluating satire?
# We provide correlations with humans


human_means = df_human.groupby("item_id")[["funny", "political"]].mean().reset_index()
llm_means = df_llm.groupby("item_id")[["funny", "political"]].mean().reset_index()
merged = pd.merge(human_means, llm_means, on="item_id", suffixes=("_human", "_llm"))

# Caculate Spearman
corr_funny, p_fun = spearmanr(merged["funny_human"], merged["funny_llm"])
corr_pol, p_pol = spearmanr(merged["political_human"], merged["political_llm"])

print(f"\n--- RQ5: LLM vs Human Correlation (Spearman) ---")
print(f"Humor Correlation: r = {corr_funny:.3f} (p-value: {p_fun:.4f})")
print(f"Political Correlation: r = {corr_pol:.3f} (p-value: {p_pol:.4f})")

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

plt.figure(figsize=(11, 5))

plt.subplot(1, 2, 1)
sns.regplot(
    data=merged,
    x="funny_human",
    y="funny_llm",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
)
plt.title(f"Humor: Human vs LLM\n(Spearman r={corr_funny:.2f})")
plt.xlabel("Human Average Score")
plt.ylabel("LLM Judge Score")
plt.xlim(1, 5)
plt.ylim(1, 5)

plt.subplot(1, 2, 2)
sns.regplot(
    data=merged,
    x="political_human",
    y="political_llm",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
)
plt.title(f"Political: Human vs LLM\n(Spearman r={corr_pol:.2f})")
plt.xlabel("Human Average Score")
plt.ylabel("LLM Judge Score")
plt.xlim(1, 5)
plt.ylim(1, 5)

plt.tight_layout()
plt.savefig("data/Plot_4_LLM_Correlation_Labeled.pdf", dpi=300)
plt.close()

print("\nAll plots and calculations are done!")
