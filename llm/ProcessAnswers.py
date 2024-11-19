from typing import List, Optional
import re
import math
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def process_fewshot_answers(
        answers_unclean: dict,
        out_dir: str,
        filename: str,
        col_names: List[str],
        gen_answers: Optional[dict] = None
        ) -> List:
    """
    Saves llm answers to file.
    Params:
        answers_unclean: answers generated from llm
        out_dir: where to save the processed answers
        filename: name for processed answers file
        col_names: categories answered
        gen_answers: if adding to previous set of answers
    Returns:
        Any unprocessed uids
    """
    if gen_answers:
        few_shot_processed_dict = gen_answers
    else:
        few_shot_processed_dict = {}
    unprocessed = []
    for k, v in answers_unclean.items():
        tmp_dict = {}
        ans = v.split("Here is the output schema:")[-1].split("AI:")[-1]
        for n in col_names:
            if ans is not None:
                try:
                    s = re.search(f"\"*\s*{n}\"*:\s+\d", ans).group()
                    d = re.search("\d", s).group()
                    if d is None:
                        if n not in ("InjuryLocationType", "WeaponType1"):
                            d = 0
                        else:
                            d = 1 # this will bias scores
                    tmp_dict[n] = int(d)
                except Exception as e:
                    unprocessed.append(k)
            else:
                tmp_dict[n] = -1
        if len(tmp_dict.keys()) != 0:
            few_shot_processed_dict[k] = tmp_dict
    
    # save the answers
    with open(out_dir + "/" + filename + ".json", "w") as f:
        json.dump(few_shot_processed_dict, f)

    return unprocessed


def calc_score(
        gen_answers_dict: dict, 
        actual_answers_df: pd.DataFrame, 
        return_separate: bool = False
        ):
    """
    Calculates F1 score in aggregate for all labels, binary and multi-category.
    Params:
        gen_answers_dict: processed answers from llm
        actual_answers_df: true answers in df form w/ uids
        return_separate: to return label and individual score
    Returns:
        final score
    """
    fk = next(iter(gen_answers_dict))
    columns = list(gen_answers_dict[fk])
    num_cols = len(columns)
    d = {c: {'actual': [], 'pred': []} for c in columns}
    for k, v in gen_answers_dict.items():
        actual = actual_answers_df[actual_answers_df['uid'] == k]
        for sk, sv in v.items():
            act_ans = actual[sk].values[0]
            d[sk]['actual'].append(act_ans)
            if sk not in ('WeaponType1', 'InjuryLocationType'):
                if math.isnan(sv) | sv not in (0,1):
                    d[sk]['pred'].append(int(not bool(act_ans)))
                else:
                    d[sk]['pred'].append(sv)
            elif sk == 'WeaponType1':
                if math.isnan(sv) | sv < 1 | sv > 12:
                    d[sk]['pred'].append(sv+1)
                else:
                    d[sk]['pred'].append(sv)
            else:
                if math.isnan(sv) | sv < 1 | sv > 6:
                    d[sk]['pred'].append(sv+1)
                else:
                    d[sk]['pred'].append(sv)

    score_dict = {}
    scores = []
    for k, v in d.items():
        a = np.array(v['actual'])
        p = np.array(v['pred'])
        if len(np.unique(a)) > 2:
            s = f1_score(a, p, average='micro')
        else:
            s = f1_score(a, p, average='binary')
        scores.append(s)
        score_dict[k] = s

    overall_score = sum(scores) / len(scores)
    if return_separate:
        return overall_score, score_dict
    return overall_score
    

if __name__ == "__main__":

    print('Script that cleans llm answers.')