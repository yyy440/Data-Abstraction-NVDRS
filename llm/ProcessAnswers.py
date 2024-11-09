from typing import List, Optional
import os
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


def multi_f1_binary(preds: np.ndarray, actuals: np.ndarray) -> float:

    per_task_f1 = []
    for a, p in zip(actuals, preds):
        per_task_f1.append(f1_score(a, p, average="micro"))
    
    return np.sum(per_task_f1) / len(per_task_f1)

def calc_binary_f1(gen_answers_dict: dict, 
                   actual_answers_df: pd.DataFrame, 
                   return_separate: bool=False):
    
    # could just do a merge / join on uid for the two dataframes
    # then fix the gen answer cols using apply
    fk = next(iter(gen_df))
    columns = list(gen_df[fk])
    num_cols = len(columns)
    
    gen_df = pd.DataFrame(gen_answers_dict).T
    actual_answers_df = actual_answers_df.set_index("uid")
    subset_answers = actual_answers_df.loc[:, columns]

    answer_matrix = [tuple() for _ in range(num_cols*2)]

    for row in gen_df.itertuples():
        uid = row.Index
        answer = subset_answers[subset_answers.index == uid]
        for c in range(len(columns)):
            col = columns[c]
            ans = answer[col].values[0]
            answer_matrix[c+1] = answer_matrix[c+1] + (ans,)
            gen_ans = row[c+1]
            if col not in ("InjuryLocationType", "WeaponType1"):
                if (gen_ans > 1) | (math.isnan(gen_ans)):
                    answer_matrix[c] = answer_matrix[c] + (int(not bool(ans)),)
                else:
                    answer_matrix[c] = answer_matrix[c] + (gen_ans,)
            elif col == "InjuryLocationType":
                if (gen_ans > 6) | (gen_ans < 1) | (math.isnan(gen_ans)):
                    answer_matrix[c] = answer_matrix[c] + (ans+1,)
                else:
                    answer_matrix[c] = answer_matrix[c] + (gen_ans,)
            elif col == "WeaponType1":
                if (gen_ans > 12) | (gen_ans < 1) | (math.isnan(gen_ans)):
                    answer_matrix[c] = answer_matrix[c] + (ans+1,)
                else:
                    answer_matrix[c] = answer_matrix[c] + (gen_ans,)
    
    scores = []
    for i in range(num_cols):
        actuals = np.array(answer_matrix[i])
        gens = np.array(answer_matrix[i+1])
        if len(np.unique(actuals)) > 2:
            actuals = acutals - 1
            gens = gens - 1
            scores.append(f1_score(actuals, preds, average='micro'))
        else:
            scores.append(f1_score(actuals, gens), average='binary')
    mv_f1 = sum(scores) / len(scores)
    assert (mv_f1 <= 1) & (mv_f1 >= 0), f"Multi F1 incorrect {mv_f1}"

    if return_separate:
        return mv_f1, answer_matrix, columns

    return mv_f1