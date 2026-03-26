import os
import evaluate
import json
from tqdm import tqdm
import numpy as np # NOTE: you don't have to use it but you are allowed to
import pandas as pd
from itertools import combinations
from tqdm import tqdm

def load_json(filename):
    """Helper to load JSON files."""
    with open(filename, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data

def save_json(mydictlist, filename):
    """Helper to save JSON files."""
    f = open(filename, 'w', encoding='utf-8')
    json.dump(mydictlist, f, ensure_ascii=False, indent=4) 
    f.close()

def create_entryid2score(entry_ids, scores):
    """Zips entry IDs and scores and creates a dictionary out of this mapping.

    Args:
        entry_ids (str list): list of data entry IDs
        scores (float list): list of scores

    Returns:
        dict: given a list of aligned entry IDs and scores creates a dictionary 
                that maps from an entry ID to the corresponding score

    """
    score_dict = {}
    for entry_id, res in zip(entry_ids, scores):
        score_dict[str(entry_id)] = res
    return score_dict

def calculate_metrics():
    ############################################################################
    # 1) Load data
    ############################################################################
    wmt_da_df = pd.read_csv(os.path.join("data", "wmt-da-human-evaluation_filtered.csv"))
    
    all_entry_ids = wmt_da_df["entry_id"].values.tolist()
    src_sent = [sent.strip().replace("\n","").replace("\t","")  for sent in wmt_da_df["src"].values]
    pred_sent = [sent.strip().replace("\n","").replace("\t","")  for sent in wmt_da_df["mt"].values]
    ref_sent = [sent.strip().replace("\n","").replace("\t","")  for sent in wmt_da_df["ref"].values]
    ref_sent_bleu = [[sent] for sent in ref_sent]
    # NOTE: in case one groups the references by src-lp pair, which should also be valid / ~ same ballpark
    # ref_by_group_dict = wmt_da_df.groupby(["src", "lp"])["ref"].apply(list).to_dict()
    # ref_sent_bleu = [ref_by_group_dict[(row.src, row.lp) ] for row in wmt_da_df.itertuples()]
    # print(len(ref_sent_bleu))
    print(len(pred_sent))
    assert len(pred_sent) == len(ref_sent)
    
    ############################################################################
    # 2) Load HF metrics
    ############################################################################
    # load the BLEU, BERTScore and COMET metrics from the evaluate package
    ############################################################################
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load('comet')    
    
    # NOTE: of the form {metric_name: {entry_id: score, ...}, ...}
    metric_dict = {}
        
    ############################################################################
    # 3.1) Calculate BLEU    
    ############################################################################
    # calculate the following bleu scores for each hypothesis
    #       - BLEU
    #       - BLEU-1
    #       - BLEU-4
    # Make sure to populate the metric_dict dictionary for each of these scores:
    #   For example, for BLEU-1 the metric_dict entry should look like:
    #   metric_dict["bleu-1"] = { "23423": 0.5 } where "23423" is an entry_id 
    #   and 0.5 is the BLEU-1 score it got. 
    # 
    # Feel free to use the `create_entryid2score` helper function.
    ############################################################################
    print("-" * 50)
    print("Calculating BLEU...")
    metric_dict["bleu"] = {}
    metric_dict["bleu-1"] = {}
    metric_dict["bleu-4"] = {}
    bleu_dictlist = [bleu.compute(predictions=[pred],  references=[ref]) for pred, ref in zip(pred_sent, ref_sent_bleu)]
    metric_dict["bleu"] = create_entryid2score(
        entry_ids=all_entry_ids,
        scores=[bleu_dict["bleu"] for bleu_dict in bleu_dictlist]
    )
    metric_dict["bleu-1"] = create_entryid2score(
        entry_ids=all_entry_ids,
        scores=[bleu_dict["precisions"][0] for bleu_dict in bleu_dictlist]
    )
    metric_dict["bleu-4"] = create_entryid2score(
        entry_ids=all_entry_ids,
        scores=[bleu_dict["precisions"][3] for bleu_dict in bleu_dictlist]
    )
    print("metric name instance: ", list(metric_dict.keys())[0])
    print("id,score pair instance: ", list(metric_dict["bleu"].items())[0])
    print("Done.")
        
    ############################################################################
    # 3.2) Calculate BERTScore
    ############################################################################
    # calculate the following BERTScore-s for each hypothesis
    #       - Precision
    #       - Recall
    #       - F-1
    # Make sure to populate the metric_dict dictionary for each of these scores.
    # Feel free to use the `create_entryid2score` helper function.
    #
    # For BERTScore, you will require to pass a `lang` parameter. Please read 
    # the documentation to figure out what that might mean. 
    # (Hint: For `lang`, you may want to use the `groupby` function of pandas dataframes!)
    ############################################################################
    print("-" * 50)
    print("Calculating BERTScore...")
    metric_dict["bertscore-precision"] = {}
    metric_dict["bertscore-recall"] = {}
    metric_dict["bertscore-f1"] = {}
    for group_name, group_df in tqdm(wmt_da_df.groupby("lp")):
        group_entry_ids = group_df["entry_id"].values
        group_preds = group_df["mt"].values
        group_refs = group_df["ref"].values
        tgt_lang = group_name.split("-")[1]
    
        bertscore_dict = bertscore.compute(predictions=group_preds, references=group_refs, lang=tgt_lang, verbose=True)
        metric_dict["bertscore-precision"].update(create_entryid2score(
            entry_ids=group_entry_ids,
            scores=bertscore_dict["precision"]
        ))
        metric_dict["bertscore-recall"].update(create_entryid2score(
            entry_ids=group_entry_ids,
            scores=bertscore_dict["recall"]
        ))
        metric_dict["bertscore-f1"].update(create_entryid2score(
            entry_ids=group_entry_ids,
            scores=bertscore_dict["f1"]
        ))
    print("metric name instance: ", list(metric_dict.keys())[3])
    print("id,score pair instance: ", list(metric_dict["bertscore-precision"].items())[0])
    print("Done.")
    
    ############################################################################
    # 3.3) Calculate COMET
    ############################################################################
    # calculate the COMET score for each hypothesis
    # Make sure to populate the metric_dict dictionary for COMET.
    # Feel free to use the `create_entryid2score` helper function.
    ############################################################################
    print("-" * 50)
    print("Calculating COMET...")
    comet_dict = comet.compute(predictions=pred_sent, references=ref_sent, sources=src_sent, progress_bar=True)
    metric_dict["comet"] = create_entryid2score(
        entry_ids=all_entry_ids,
        scores=comet_dict["scores"]
    )
    print("metric name instance: ", list(metric_dict.keys())[6])
    print("id,score pair instance: ", list(metric_dict["comet"].items())[0])
    print("Done.")
    
    ############################################################################
    # 4) Save the output in a JSON file
    ############################################################################
    save_json(metric_dict, "part2_metrics.json")
    return metric_dict
    

def evaluate_metrics():
    ############################################################################
    # 1) Load data
    ############################################################################
    wmt_da_df = pd.read_csv(os.path.join("data", "wmt-da-human-evaluation_filtered.csv"))
    print(wmt_da_df.head())
    print(len(wmt_da_df))
    
    ############################################################################
    # 2) Create ranked data for Kendall's Tau
    ############################################################################
    # For each (source, lp) group, rank the entry_id s by the "score".
    #       And then create rank_pairs_list: a list of ranking pairs which are
    #       (worse hypothesis id, better_hypothesis id)
    #       Hint: use combinations from itertools!
    ############################################################################
    def generate_rank_pairs(group):
        sorted_group = group.sort_values(by="score", ascending=True)
        ids = sorted_group["entry_id"].tolist()
        return list(combinations(ids, 2))
    rank_pairs_by_group = wmt_da_df.groupby(["src", "lp"]).apply(generate_rank_pairs)
    rank_pairs_list = [
        pair for _, pair_list in rank_pairs_by_group.items() for pair in pair_list
    ]
    # NOTE: The following should be ~3351
    print("Size of rank combinations: ", len(rank_pairs_list))
    
    ############################################################################
    # 2) Create a class to calculate Kendalls Tau for each metric
    ############################################################################
    # Complete the class such that each call to the class can update the 
    #       count of concordant and discordant values
    ############################################################################
    class KendallsTau:
        """
        A class to accumulate concordant and discordant instances and to
        compute Kendall's Tau correlation coefficient. 
        Helps when iteratively doing the computation.
        Feel free to implement it otherwise if you don't want to do it iteratively.
        """
        def __init__(self):
            self.concordant = 0.0
            self.discordant = 0.0
            self.total = 0.0

        def update(self, worse_hyp_score, better_hyp_score):
            """Updates the concordant and discordant values.

            Args:
                worse_hyp_score (float): the score for the worse hypothesis 
                        according to human ranking
                better_hyp_score (float): the score for the better hypothesis 
                        according to human ranking
            """
            self.concordant += int(better_hyp_score > worse_hyp_score)
            self.discordant += int(better_hyp_score < worse_hyp_score)
            self.total += 1.0

        def compute(self):
            """
            Calculates the Kendall's Tau correlation coefficient. 
            Call when all ranked pairs have been evaluated.
            """
            return (self.concordant - self.discordant) / self.total
    
    ############################################################################
    # 3) Calculate Kendall's Tau correlation for each metric
    ############################################################################
    # Populate the metrics2kendalls dictionary s.t. we get 
    #       metric2kendalls = {metric_name: correlation, ...} for all metrics
    ############################################################################
    metric_dict = load_json("part2_metrics.json")
    metric2kendalls = {}
    for metric_name, entryid2score in metric_dict.items():
        print("-" * 80)
        print("Kendall's Tau for: ", metric_name)
        kendallstau = KendallsTau()
        for worse_hyp_id, better_hype_id in rank_pairs_list:
            worse_hyp_score = entryid2score[str(worse_hyp_id)]
            better_hype_score = entryid2score[str(better_hype_id)]
            kendallstau.update(
                worse_hyp_score=worse_hyp_score,
                better_hyp_score=better_hype_score
            )
        print(kendallstau.compute())
        metric2kendalls[metric_name] = kendallstau.compute()
    
    ############################################################################
    # 4) Save the output in a JSON file
    ############################################################################
    save_json(metric2kendalls, "part2_corr.json")
        

if __name__ == '__main__':
    already_predicted_scores = False
    if not already_predicted_scores:
        calculate_metrics()
    evaluate_metrics()