from pathlib import Path
import pandas as pd
from glob import glob
import json

from dropexp.utils import mean_confidence_interval, ks

def analyze_kns(dropout, dropfree):
    dropout_cluster_counts = []

    dropout_intervene_match_augment = []
    dropout_intervene_match_suppress = []
    dropout_intervene_no_match_augment = []
    dropout_intervene_no_match_suppress = []

    dropout_baseline_match_augment = []
    dropout_baseline_match_suppress = []
    dropout_baseline_no_match_augment = []
    dropout_baseline_no_match_suppress = []

    def compute_kn_means(type):
        augment = df.groupby("match").augment_success.mean()
        suppress = df.groupby("match").suppress_success.mean()

        return augment.loc[True], suppress.loc[True], augment.loc[False], suppress.loc[False]


    concepts = glob(str(dropout / "kns" / "*_intervene.csv"))
    for i in concepts:
        df = pd.read_csv(i)
        a,b,c,d = compute_kn_means(df)
        dropout_intervene_match_augment.append(a)
        dropout_intervene_match_suppress.append(b)
        dropout_intervene_no_match_augment.append(c)
        dropout_intervene_no_match_suppress.append(d)
        dropout_cluster_counts.append(len(df.knowledge_cluster.value_counts()))

    concepts = glob(str(dropout / "kns" / "*_baseline.csv"))
    for i in concepts:
        df = pd.read_csv(i)
        a,b,c,d = compute_kn_means(df)
        dropout_baseline_match_augment.append(a)
        dropout_baseline_match_suppress.append(b)
        dropout_baseline_no_match_augment.append(c)
        dropout_baseline_no_match_suppress.append(d)

    dropfree_cluster_counts = []
    dropfree_intervene_match_augment = []
    dropfree_intervene_match_suppress = []
    dropfree_intervene_no_match_augment = []
    dropfree_intervene_no_match_suppress = []

    dropfree_baseline_match_augment = []
    dropfree_baseline_match_suppress = []
    dropfree_baseline_no_match_augment = []
    dropfree_baseline_no_match_suppress = []

    def compute_kn_means(type):
        augment = df.groupby("match").augment_success.mean()
        suppress = df.groupby("match").suppress_success.mean()

        return augment.loc[True], suppress.loc[True], augment.loc[False], suppress.loc[False]

    concepts = glob(str(dropfree / "kns" / "*_intervene.csv"))
    for i in concepts:
        df = pd.read_csv(i)
        a,b,c,d = compute_kn_means(df)
        dropfree_intervene_match_augment.append(a)
        dropfree_intervene_match_suppress.append(b)
        dropfree_intervene_no_match_augment.append(c)
        dropfree_intervene_no_match_suppress.append(d)
        dropfree_cluster_counts.append(len(df.knowledge_cluster.value_counts()))

    concepts = glob(str(dropfree / "kns" / "*_baseline.csv"))
    for i in concepts:
        df = pd.read_csv(i)
        a,b,c,d = compute_kn_means(df)
        dropfree_baseline_match_augment.append(a)
        dropfree_baseline_match_suppress.append(b)
        dropfree_baseline_no_match_augment.append(c)
        dropfree_baseline_no_match_suppress.append(d)

    do_clusters = mean_confidence_interval(dropout_cluster_counts)
    df_clusters = mean_confidence_interval(dropfree_cluster_counts)

    do_mnm_augment = ks(dropout_intervene_match_augment, dropout_intervene_no_match_augment)
    ndo_mnm_augment = ks(dropfree_intervene_match_augment, dropfree_intervene_no_match_augment)

    do_baseline_augment = ks(dropout_baseline_match_augment, dropout_baseline_no_match_augment)
    ndo_baseline_augment = ks(dropfree_baseline_match_augment, dropfree_baseline_no_match_augment)

    do_concepts = str(dropout / "kns" / "kns.json")
    do_knowledge = mean_confidence_interval(pd.read_json(do_concepts, orient="split").knowledge.apply(len))

    df_concepts = str(dropfree / "kns" / "kns.json")
    df_knowledge = mean_confidence_interval(pd.read_json(df_concepts, orient="split").knowledge.apply(len))
    
    return  {
        "neurons": {
            "neuron_count_p95": {
                "dropout": do_knowledge,
                "no_dropout": df_knowledge,
            }
        },
        "clustering": {
            "clusters_p95": {
                "dropout": do_clusters,
                "no_dropout": df_clusters,
            }
        },
        "clustering_effect": {
            "augment_success_matching_ks_pval": {
                "dropout": do_mnm_augment.pvalue,
                "no_dropout": ndo_mnm_augment.pvalue,
            },
            "augment_success_baseline_ks_pval": {
                "dropout": do_baseline_augment.pvalue,
                "no_dropout": ndo_baseline_augment.pvalue,
            }
        }
    }

