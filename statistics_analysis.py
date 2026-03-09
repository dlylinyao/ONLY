import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon
import nltk
from nltk.metrics.agreement import AnnotationTask

# Set random seed for reproducible bootstrapping
np.random.seed(42)

df = pd.read_csv("data/annotations_and_llmasajudge.csv")
df_human = df[df["annotator_group"].isin(["finnish", "international"])].copy()
df_llm = df[df["annotator_group"] == "-"].copy() # Assuming LLM model names are in 'annotator_id'

# Distinguish Topic words vs Random words
topic_words = [
    "company", "euros", "age", "health", "rydman", "border", "win", "medal",
    "court", "president", "government", "tax", "venezuela", "agency", "year",
    "parliamentary", "temperatures", "january", "kivimki", "retirement",
    "people", "food", "school", "areas", "police", "olympics", "language",
    "chance", "purra", "posti", "positions", "social", "students", "finlands",
    "country", "finland", "prices", "umk", "employee", "entry", "minister",
    "wolt", "employees", "test", "finns", "finnish", "juuso", "contest",
    "party", "song",
]
df_human["word_type"] = df_human["word"].apply(
    lambda x: "Topic (News)" if x in topic_words else "Random"
)

# Suggestion 3: Normalize annotator scores (Z-score normalization)
def z_score(x):
    std = x.std()
    if pd.isna(std) or std == 0:
        return x - x.mean()
    return (x - x.mean()) / std

df_human["funny_z"] = df_human.groupby("annotator_id")["funny"].transform(z_score)
df_human["political_z"] = df_human.groupby("annotator_id")["political"].transform(z_score)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

# Plot 1: RQ1 - Absolute Quality (Are they funny and political?)

print(f"\n--- RQ1 Data: Absolute Quality ---")
print(f"Average Funny Score: {df_human['funny'].mean():.2f} ± {df_human['funny'].std():.2f}")
print(f"Average Political Score: {df_human['political'].mean():.2f} ± {df_human['political'].std():.2f}")

plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
sns.countplot(data=df_human, x="funny", hue="funny", palette="Blues", ax=ax1, legend=False)
plt.title("Distribution of Humor Scores")

ax2 = plt.subplot(1, 2, 2)
sns.countplot(data=df_human, x="political", hue="political", palette="Reds", ax=ax2, legend=False)
plt.title("Distribution of Political Scores")

plt.tight_layout()
plt.savefig("data/Plot_1_Distributions.pdf", dpi=300)
plt.close()

# Plot 2: RQ2 - Is satire cultural?
# Added IAA Calculation and Political p-value
print(f"\n--- RQ2 Data: Cultural Differences & IAA ---")

def calculate_iaa(df_subset, metric):
    # AnnotationTask expects a list of (coder, item, label)
    task_data = list(zip(df_subset['annotator_id'], df_subset['item_id'], df_subset[metric]))
    task = AnnotationTask(data=task_data)
    try:
        return task.alpha() # Krippendorff's alpha is good for interval/ordinal data
    except:
        return float('nan')

print(f"IAA (Krippendorff's alpha) - Overall Humor: {calculate_iaa(df_human, 'funny'):.3f}, Political: {calculate_iaa(df_human, 'political'):.3f}")
print(f"IAA - Finnish Humor: {calculate_iaa(df_human[df_human['annotator_group'] == 'finnish'], 'funny'):.3f}, Political: {calculate_iaa(df_human[df_human['annotator_group'] == 'finnish'], 'political'):.3f}")
print(f"IAA - International Humor: {calculate_iaa(df_human[df_human['annotator_group'] == 'international'], 'funny'):.3f}, Political: {calculate_iaa(df_human[df_human['annotator_group'] == 'international'], 'political'):.3f}")

# Humor significance
fin_humor = df_human[df_human["annotator_group"] == "finnish"]["funny_z"].dropna()
int_humor = df_human[df_human["annotator_group"] == "international"]["funny_z"].dropna()
stat_h, p_val_cult_h = mannwhitneyu(fin_humor, int_humor, alternative='two-sided')
print(f"Mann-Whitney U (Finnish vs Int - Humor Z-score): p-value = {p_val_cult_h:.4f}")

# Political significance
fin_pol = df_human[df_human["annotator_group"] == "finnish"]["political_z"].dropna()
int_pol = df_human[df_human["annotator_group"] == "international"]["political_z"].dropna()
stat_p, p_val_cult_p = mannwhitneyu(fin_pol, int_pol, alternative='two-sided')
print(f"Mann-Whitney U (Finnish vs Int - Political Z-score): p-value = {p_val_cult_p:.4f}")

plt.figure(figsize=(7, 5))
df_melt_cult = df_human.melt(
    id_vars=["annotator_group"],
    value_vars=["funny", "political"],
    var_name="Metric", value_name="Score"
)
ax3 = sns.barplot(data=df_melt_cult, x="annotator_group", y="Score", hue="Metric", errorbar="ci", capsize=0.1, palette="Set2")
plt.title(f"Scores by Cultural Group\n(p-vals: Humor {p_val_cult_h:.3f}, Pol {p_val_cult_p:.3f})")
plt.ylim(1, 4)
plt.savefig("data/Plot_2_Cultural_Differences.pdf", dpi=300)
plt.close()

# Plot 3: RQ4 - Word candidate selection
# Added Political p-value

print(f"\n--- RQ4 Data: Word Candidate Selection ---")
# Humor significance
topic_humor = df_human[df_human["word_type"] == "Topic (News)"]["funny_z"].dropna()
random_humor = df_human[df_human["word_type"] == "Random"]["funny_z"].dropna()
stat_h, p_val_word_h = mannwhitneyu(topic_humor, random_humor, alternative='two-sided')
print(f"Mann-Whitney U (Topic vs Random - Humor Z-score): p-value = {p_val_word_h:.4f}")

# Political significance
topic_pol = df_human[df_human["word_type"] == "Topic (News)"]["political_z"].dropna()
random_pol = df_human[df_human["word_type"] == "Random"]["political_z"].dropna()
stat_p, p_val_word_p = mannwhitneyu(topic_pol, random_pol, alternative='two-sided')
print(f"Mann-Whitney U (Topic vs Random - Political Z-score): p-value = {p_val_word_p:.4f}")

plt.figure(figsize=(7, 5))
df_melt_word = df_human.melt(
    id_vars=["word_type"], value_vars=["funny", "political"],
    var_name="Metric", value_name="Score"
)
ax4 = sns.barplot(data=df_melt_word, x="word_type", y="Score", hue="Metric", errorbar="ci", capsize=0.1, palette="Set3")
plt.title(f"Word Selection: Topic vs Random\n(p-vals: Humor {p_val_word_h:.3f}, Pol {p_val_word_p:.3f})")
plt.ylim(1, 4)
plt.savefig("data/Plot_3_Word_Selection.pdf", dpi=300)
plt.close()

# Plot 5: RQ3 - Is retrieval improving the satire?
# Added Political p-value

print(f"\n--- RQ3 Data: RAG vs Non-RAG ---")
# Humor significance
rag_paired_h = df_human.groupby(["word", "rag"])["funny_z"].mean().unstack().dropna()
if not rag_paired_h.empty and len(rag_paired_h.columns) == 2:
    stat, p_val_rag_h = wilcoxon(rag_paired_h[1], rag_paired_h[0])
    print(f"Wilcoxon Signed-Rank (RAG vs Non-RAG - Humor Z-score): p-value = {p_val_rag_h:.4f}")
else:
    p_val_rag_h = float('nan')

# Political significance
rag_paired_p = df_human.groupby(["word", "rag"])["political_z"].mean().unstack().dropna()
if not rag_paired_p.empty and len(rag_paired_p.columns) == 2:
    stat, p_val_rag_p = wilcoxon(rag_paired_p[1], rag_paired_p[0])
    print(f"Wilcoxon Signed-Rank (RAG vs Non-RAG - Political Z-score): p-value = {p_val_rag_p:.4f}")
else:
    p_val_rag_p = float('nan')

plt.figure(figsize=(7, 5))
df_melt_rag = df_human.melt(
    id_vars=["rag"], value_vars=["funny", "political"],
    var_name="Metric", value_name="Score"
)
ax5 = sns.barplot(data=df_melt_rag, x="rag", y="Score", hue="Metric", errorbar="ci", capsize=0.1, palette="Pastel1")
plt.title(f"RAG vs Non-RAG Average Scores\n(p-vals: Humor {p_val_rag_h:.3f}, Pol {p_val_rag_p:.3f})")
plt.xticks([0, 1], ["Non-RAG (0)", "RAG (1)"])
plt.ylim(1, 4)
plt.savefig("data/Plot_5_RAG_vs_NonRAG.pdf", dpi=300)
plt.close()

# Plot 4: RQ5 - LLMs evaluating satire

print(f"\n--- RQ5 Data: LLM vs Human Correlation ---")

human_means = df_human.groupby("item_id")[["funny_z", "political_z"]].mean().reset_index()
human_means = human_means.rename(columns={"funny_z": "funny_human", "political_z": "political_human"})

def bootstrap_spearman(x, y, n_resamples=1000):
    indices = np.arange(len(x))
    corrs = []
    for _ in range(n_resamples):
        sample_idx = np.random.choice(indices, size=len(indices), replace=True)
        corr, _ = spearmanr(x.iloc[sample_idx], y.iloc[sample_idx])
        corrs.append(corr)
    return np.percentile(corrs, 2.5), np.percentile(corrs, 97.5)

llm_models = df_llm["annotator_id"].unique()

plt.figure(figsize=(12, 5 * len(llm_models)))

for idx, model in enumerate(llm_models):
    df_model = df_llm[df_llm["annotator_id"] == model]
    model_means = df_model.groupby("item_id")[["funny", "political"]].mean().reset_index()
    model_means = model_means.rename(columns={"funny": "funny_llm", "political": "political_llm"})
    
    merged = pd.merge(human_means, model_means, on="item_id")
    
    corr_fun, p_fun = spearmanr(merged["funny_human"], merged["funny_llm"])
    ci_fun_low, ci_fun_high = bootstrap_spearman(merged["funny_human"], merged["funny_llm"])
    
    corr_pol, p_pol = spearmanr(merged["political_human"], merged["political_llm"])
    ci_pol_low, ci_pol_high = bootstrap_spearman(merged["political_human"], merged["political_llm"])
    
    print(f"\nModel: {model}")
    print(f"  Humor Correlation: r = {corr_fun:.3f} (95% CI: [{ci_fun_low:.3f}, {ci_fun_high:.3f}]), p = {p_fun:.4f}")
    print(f"  Political Correlation: r = {corr_pol:.3f} (95% CI: [{ci_pol_low:.3f}, {ci_pol_high:.3f}]), p = {p_pol:.4f}")

    plt.subplot(len(llm_models), 2, idx * 2 + 1)
    sns.regplot(data=merged, x="funny_human", y="funny_llm", scatter_kws={"alpha": 0.5})
    plt.title(f"{model} - Humor Correlation\nr={corr_fun:.2f} CI:[{ci_fun_low:.2f}, {ci_fun_high:.2f}]")
    plt.xlabel("Human Avg (Z-score)")
    plt.ylabel(f"{model} Score")

    plt.subplot(len(llm_models), 2, idx * 2 + 2)
    sns.regplot(data=merged, x="political_human", y="political_llm", scatter_kws={"alpha": 0.5})
    plt.title(f"{model} - Political Correlation\nr={corr_pol:.2f} CI:[{ci_pol_low:.2f}, {ci_pol_high:.2f}]")
    plt.xlabel("Human Avg (Z-score)")
    plt.ylabel(f"{model} Score")

plt.tight_layout()
plt.savefig("data/Plot_4_LLM_Correlation_Separated.pdf", dpi=300)
plt.close()

print("\nAll analyses complete! Charts have been saved with CIs and significance testing.")