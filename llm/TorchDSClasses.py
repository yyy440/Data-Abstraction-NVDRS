from typing import Optional
import pandas as pd
import torch
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from AnswerGeneration import get_examples, gen_prompt_kwargs

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 features_df: pd.DataFrame,
                 labels_df:pd.DataFrame,
                 sim_df: pd.DataFrame,
                 schema,
                 prompt_template,
                 summary: str = "NarrativeCME",
                 is_fewshot: bool = True,
                 prefix: Optional[str] = None,
                 examples_df: Optional[pd.DataFrame] = None,
                 top_k: int = 3):
        super().__init__()
        
        self.features_df = features_df
        self.labels_df = labels_df
        self.examples_df = examples_df
        self.sim_df = sim_df

        self.summary = summary
        self.parser = PydanticOutputParser(pydantic_object=schema)
        self.schema = schema
        self.col_names = list(schema.__fields__.keys())
        self.prompt_template = prompt_template
        self.is_fewshot = is_fewshot
        self.top_k = top_k
        self.prefix = prefix
    
    def __len__(self):

        return len(self.features_df)
    
    def __getitem__(self, idx):
        
        uid = self.features_df.iloc[idx]["uid"]
        if self.is_fewshot:
            examples = get_examples(uid=uid, column_names=self.col_names, top_k=self.top_k, similarity_df=self.sim_df, features_df=self.examples_df, labels_df=self.labels_df)
            prompt_kwargs, parser = gen_prompt_kwargs(schema=self.schema, prefix=self.prefix, 
                                                      examples=examples)
            prompt = FewShotPromptTemplate(**prompt_kwargs)
        else:
            prompt = PromptTemplate(template=self.prompt_template, input_variables=["text"],
                           partial_variables={"format_instructions": self.parser.get_format_instructions()})
        passage = self.features_df.iloc[idx][self.summary]
        assert isinstance(passage, str), print(f"passage wrong type {type(passage)},\n{passage}")
        if self.is_fewshot:
            prompt = prompt.invoke({"input": passage}).to_string()
        else:
            prompt = prompt.format(text=passage)
        
        return uid, prompt

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 features_df, 
                 labels_df, 
                 tokenizer,
                 schema,
                 prompt_template,
                 max_ctx_len,
                 summary="NarrativeCME",
                 examples_df=None,
                 is_test=False,
                 few_shot=False,
                 top_k=3,
                 sim_df=None,
                 prefix=None
                 ):
        super().__init__()
        
        self.features = features_df
        self.examples_df = examples_df
        self.labels = labels_df
        self.parser = PydanticOutputParser(pydantic_object=schema)
        self.prompt_template = prompt_template
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.few_shot = few_shot
        self.is_test = is_test
        self.sim_df = sim_df
        self.schema = schema
        self.top_k = top_k
        self.col_names = list(schema.__fields__.keys())
        self.prefix = prefix
        self.max_ctx_len = max_ctx_len
        self.summary = summary

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        
        uid = self.features.iloc[idx]["uid"]
        if self.few_shot:
            examples = get_examples(uid=uid, column_names=self.col_names, top_k=self.top_k, 
                                    similarity_df=self.sim_df, features_df=self.examples_df,
                                    labels_df=self.labels)
            prompt_kwargs, parser = gen_prompt_kwargs(schema=self.schema, prefix=self.prefix, 
                                                      examples=examples)
            prompt = FewShotPromptTemplate(**prompt_kwargs)
        else:
            prompt = PromptTemplate(template=self.prompt_template, input_variables=["text"],
                           partial_variables={"format_instructions": self.parser.get_format_instructions()})
        
        passage = self.features.iloc[idx][self.summary]
        assert isinstance(passage, str), print(f"passage wrong type {type(passage)},\n{passage}")
        if self.few_shot:
            prompt = prompt.invoke({"input": passage}).to_string()
        else:
            prompt = prompt.format(text=passage)
        
        text_tokenized = self.tokenizer(
            prompt, #str(self.titles[index]),
            max_length=self.max_ctx_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            #return_attention_mask=True,
            return_tensors="pt"
        )
        labels = self.labels.iloc[idx][self.col_names].values
        
        if not self.is_test:
            return uid, text_tokenized, labels

        return uid, text_tokenized