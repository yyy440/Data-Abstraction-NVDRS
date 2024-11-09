import copy
import pandas as pd
from tqdm import tqdm
import gc

import torch
from accelerate import Accelerator
from transformers import (AutoModel, AutoTokenizer,AutoModelForCausalLM, pipeline, 
                          GenerationConfig, BitsAndBytesConfig)

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

import TorchDSClasses as ds
import PromptTemplates as ptemps

pipeline_kwargs = {"max_new_tokens":200,
                   "device":"cuda",
                   "temperature": 0.1,
                  "top_k": 1}

def get_examples(uid: str, 
                 column_names: list[str], 
                 top_k: int, 
                 similarity_df: pd.DataFrame,
                 features_df: pd.DataFrame, 
                 labels_df: pd.DataFrame) -> list[dict]:

    '''
    Get top k examples formatted as list of dicts for FewShotTemplate
    Params:
        uid: str to pull examples for
    Return: list of dicts for prompt
    '''
    num_labels = len(column_names)
    sim_row = similarity_df[similarity_df["uid"] == uid].drop("uid", axis=1)
    top_uids = sim_row.iloc[:, :top_k].values[0]
    top_passages = features_df[features_df["uid"].isin(top_uids)]
    ex_passages = top_passages[["uid", "NarrativeCME"]]
    
    col_names_w_uid = copy.deepcopy(column_names)
    col_names_w_uid.insert(0, "uid")
    ex_answers = labels_df[labels_df["uid"].isin(top_uids)][col_names_w_uid]
    temp_input_vars = []
    json_form = """{{"""
    for i in range(num_labels):
        temp_input_vars.append(f"col{i+1}")
        temp_input_vars.append(f"ans{i+1}")
        if i != (num_labels-1):
            json_form += f"{{col{i+1}}}: {{ans{i+1}}}, "
        else:
            json_form += f"{{col{i+1}}}: {{ans{i+1}}}"
    json_form += """}}"""
    
    json_temp = PromptTemplate(input_variables=temp_input_vars, template=json_form)
    formatted_examples = []
    for row in ex_passages.itertuples():
        d = {}
        try:
            ex = ex_answers[ex_answers["uid"] == row.uid]
        except Exception as e:
            print(f"Error when subsetting ex_answers {e}")
            print(ex_answers)
            print(row.uid)
        for i in range(num_labels):
            d[f"col{i+1}"] = column_names[i]
            d[f"ans{i+1}"] = ex[column_names[i]].values[0]
        
        formatted_prompt = json_temp.format(**d)
        l_bracket = "{"
        formatted_prompt += "}"
        final_formatted_prompt = l_bracket + formatted_prompt
        example = {"Text": f"{row.NarrativeCME}", 
                    "Answer": f"{final_formatted_prompt}"}
        formatted_examples.append(example)
    
    return formatted_examples

def gen_prompt_kwargs(schema, prefix, examples):
    
    parser = PydanticOutputParser(pydantic_object=schema)
    prefix_temp = PromptTemplate(template=prefix
                                 #,partial_variables={"format_instructions": parser.get_format_instructions()}
                                          )
    prefix_temp = prefix_temp.format()
    prompt_kwargs = {"examples": examples, #dynamic
                    "example_prompt":ptemps.example_prompt, # static
                    "prefix": prefix_temp, #dynamic 
                    "suffix": ptemps.suffix, #static
                    "input_variables":["input"],
                     "partial_variables": {"format_instructions": parser.get_format_instructions()},
                     "output_parser": parser
                    } # static
    
    return prompt_kwargs, parser

# REFACTOR FOR CLI
def langchain_fewshot_answers(top_k: int,
                        schema,
                        prefix: str,
                        similarity_df: pd.DataFrame, 
                        features_df: pd.DataFrame, 
                        labels_df: pd.DataFrame,
                        model_id: str,
                        pipeline_kwargs: dict,
                        batch_size: int = 4) -> dict:
    
    column_names = list(schema.__fields__.keys())
    # init model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                              torch_dtype=torch.float16,
                                              device_map="auto",
                                              cache_dir="/home/hice1/yyao386/scratch/huggingface/hub")
    tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir="/home/hice1/yyao386/scratch/huggingface/hub")
    hf_pipe = pipeline("text-generation", model=model, max_new_tokens=300, 
                       tokenizer=tokenizer, device_map="auto")
    langchain_pipe = HuggingFacePipeline(pipeline=hf_pipe, batch_size=batch_size, 
                                        pipeline_kwargs=pipeline_kwargs)
    gen_answers = {}
    for b in range(len(features_df)):
        sub = features_df.iloc[b, :]
        uid = sub["uid"]
        question = sub["NarrativeCME"]
        examples = get_examples(uid, column_names, top_k, similarity_df, features_df, labels_df)
        prompt_kwargs, parser = gen_prompt_kwargs(schema, prefix, examples)
        prompt = FewShotPromptTemplate(**prompt_kwargs)
        chain = prompt | langchain_pipe# | parser
        answer = chain.invoke({"input": question})
        gen_answers[uid] = answer
    
    return gen_answers


def hfpipe_fewshot_answers(model_id,
                           top_k: int,
                           schema,
                           prompt_template,
                           prefix: str,
                           similarity_df: pd.DataFrame, 
                           features_df: pd.DataFrame,
                           examples_df: pd.DataFrame,
                           labels_df: pd.DataFrame,
                           summary: str = "NarrativeCME",
                           batch_size: int = 4,
                           quant = None) -> dict:
    
    if quant:
        q_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=q_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    dataset = ds.TextDataset(features_df=features_df, 
                            labels_df=labels_df,
                            sim_df=similarity_df,
                            schema=schema,
                            prompt_template=prompt_template,
                            summary=summary,
                            is_fewshot=True,
                            prefix=prefix,
                            examples_df=examples_df,
                            top_k=top_k)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=False, pin_memory=True)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(
        model, dataloader
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_pipe = pipeline("text-generation", model=model, max_new_tokens=100, 
                       tokenizer=tokenizer, device_map="auto")
    
    gen_answers = {}
    for idx, (uids, prompts) in enumerate(tqdm(dataloader)):
        answers = hf_pipe(prompts)
        for a in range(len(answers)):
            gen_answers[uids[a]] = answers[a][0]['generated_text']

    del hf_pipe, model, tokenizer, dataloader, dataset
    gc.collect()

    return gen_answers
        
        
    
    