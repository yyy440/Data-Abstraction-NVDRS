# Data Abstraction from Unstructured Text
---
This repo hosts the code used for experiments on extracting information from unstructured text summaries as part of the NVDRS Youth Suicide competition hosted on Driven Data. The best results
were using a combination of Mistral 7b to extract a majority of the information and then SVM for a few of the categories that Mistral did poorly on.
---
Tour
---
There are 3 directories. The LLM folder contains code to extract info from text in a chat manner using any desired LLM. Fine-tune has a script that was used to train BERT models 
for direct multi-label classification. Notebooks contain Jupyter notebooks of experiments testing XGBoost, SVM, Random Forest, and MLP to extract each label. In addition, in fine-tune, there is a
folder labeled llm2vec that contain code for fine-tuning a llm as a encoder.
