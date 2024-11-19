import os
import logging
import pandas as pd
import argparse
import json

from huggingface_hub import login
import AnswerGeneration as agen
import SchemaClasses as sch
import PromptTemplates as ptemps
import ProcessAnswers as proc


def load_data():
    """
    Checks if there is already generated answers. 
    Load features, labels, similarity df, and generated answers.
    Params:
        None
    Returns:
        pd.dataframes and a answer dict
    """
    DIR = os.get_cwd()
    FEATURE_FILE = os.path.join(DIR, args.data_dir, args.feature_file)
    features_df = pd.read_csv(FEATURE_FILE)
    LABEL_FILE = os.path.join(DIR, args.data_dir, args.label_file)
    labels_df = pd.read_csv(LABEL_FILE)
    SIM_FILE = os.path.join(DIR, args.data_dir, args.sim_filename)
    sim_df = pd.read_csv(SIM_FILE)
    examples_df = features_df

    ANSWER_PATH = os.path.join(DIR, args.answer_dir, f"{args.answer_filename}.json")
    if os.path.exists(ANSWER_PATH):
        with open(ANSWER_PATH, "r") as f:
            gen_answers_dict = json.load(f)
            assert isinstance(gen_answers_dict, dict), f"gen_answers_dict not dict {type(gen_answers_dict)}"
        processed_ids_list = list(gen_answers_dict.keys())
        features_df = features_df[~features_df['uid'].isin(processed_ids_list)]
        logging.warning(f"Num rows after removing answered uids: {len(features_df)}")
    else:
        gen_answers_dict = {}
    features_df = features_df.iloc[:args.n_data, :]
    logging.warning(f"Num rows after subsetting by n_data: {len(features_df)}")
    return features_df, labels_df, sim_df, examples_df, gen_answers_dict

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    # hf login api key
    parser.add_argument("hf_key", help="Text with HF API login key", type=str)
    # data paths
    parser.add_argument("data_dir", help="Dir with data", type=str)
    parser.add_argument("feature_file", help="Features filename", type=str)
    parser.add_argument("label_file", help="Labels filename", type=str)
    parser.add_argument("sim_filename", help="Similarity matrix filename", type=str)
    parser.add_argument("n_data", help="How many data points to process", type=int)
    parser.add_argument("answer_dir", help="Dir with gen answers", type=str)
    parser.add_argument("answer_filename", help="Gen answers filename", type=str)
    # parser.add_argument("feature_cols", help="What columns to gen answers for", type=list)
    
    # model
    parser.add_argument("model_id", help="HF model ID", type=str)
    parser.add_argument("q8bit", help="8 bit quant", type=bool)

    # answer gen
    parser.add_argument("top_k", help="How many examples", type=int)
    parser.add_argument("schema_num", help="Which schema", type=int, choices=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("batch_size", help="Batch size in mutliples of 2", type=int)

    args = parser.parse_args()

    return args


def main(args):
    """
    Generate answers using a LLM and save to specified out_dir
    Params:
        args: settings for script
    Returns: None
    """

    ANSWER_DIR = args.answer_dir
    if not os.path.exists(ANSWER_DIR):
        os.makedirs(ANSWER_DIR)

    # login to hf
    login(args.hf_key)

    # load data
    features_df, labels_df, sim_df, examples_df, gen_answers_dict = load_data()

    logging.warning(f"Dict has: {len(gen_answers_dict.keys())} keys after loading")

    # generate answers
    if args.schema_num == 1:
        schema = sch.FirstSchema 
    elif args.schema_num == 2:
        schema = sch.SecondSchema
    elif args.schema_num == 3:
        schema = sch.ThirdSchema
    elif args.schema_num == 4:
        schema = sch.FourthSchemaSchema
    elif args.schema_num == 5:
        schema = sch.FifthSchema
    elif args.schema_num==6:
        schema = sch.SixthSchema
    else:
        schema = sch.AllSchema
    
    feature_col_names = list(schema.__fields__.keys())
    # gen answers
    unprocessed_ans = agen.langchain_fewshot_answers(top_k=args.top_k,
                        schema=schema,
                        prefix=ptemps.all_prefix,
                        similarity_df=sim_df, 
                        features_df=features_df, 
                        labels_df=labels_df,
                        model_id=args.model_id,
                        pipeline_kwargs=agen.pipeline_kwargs,
                        batch_size=args.batch_size)

    # process and save answers
    unprocessed_uids = proc.process_fewshot_answers(answers_unclean=unprocessed_ans,
                            out_dir=args.answer_dir,
                            filename=args.answer_filename,
                            col_names=feature_col_names,
                            gen_answers=gen_answers_dict)

    

if __name__ == "__main__":

    args = parse_arguments()
    main(args)

    