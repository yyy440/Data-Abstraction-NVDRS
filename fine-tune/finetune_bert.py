# from this colab notebook and huggingface examples https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

import os
import pandas as pd 
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch

from datasets import Dataset, load_dataset

from transformers import (AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, TrainingArguments, Trainer)

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

def my_gen():
    for i in range(len(train_data)):
        yield {label_col_names[j]: train_data.iloc[i, j] for j in range(len(label_col_names))}

def main():

    DIR = os.getcwd()
    DATA_DIR = os.path.join(DIR, "data")
    TRAIN_TEXT_FILE = os.path.join(DATA_DIR, "train_features.csv")
    TRAIN_LABEL_FILE = os.path.join(DATA_DIR,"train_labels.csv")
    model_id = "microsoft/deberta-v3-base"
    train_text = pd.read_csv(TRAIN_TEXT_FILE)
    train_labels = pd.read_csv(TRAIN_LABEL_FILE)
    # set uid as index
    train_text.set_index('uid', inplace=True)
    train_labels.set_index('uid', inplace=True)

    # drop cols 
    drop_labels = ['InjuryLocationType', "WeaponType1"]
    train_text.drop(["NarrativeLE"], axis=1, inplace=True)
    train_labels.drop(drop_labels, axis=1, inplace=True)

    # merge 
    train_data = train_text.join(train_labels, how='inner')
    train_data.rename(columns={"NarrativeCME": "text"}, inplace=True)

    label_col_names = train_data.columns.to_list()

    # create dataset
    dataset = Dataset.from_generator(my_gen)
    dataset = dataset.train_test_split(test_size=0.15)

    labels = label_col_names[1:]
    id2label = {idx:label for idx, label in enumerate(labels)} 
    label2id = {label:idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess_data(examples):
        # take a batch of texts
        text = examples["text"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=1300)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        
        return encoding

    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                           torch_dtype=torch.bfloat16,
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

    batch_size = 4
    metric_name = "f1"

    args = TrainingArguments(
        f"deberta-finetuned-nvdrs",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        #device='cuda' if torch.cuda.is_available() else 'cpu',
        #learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=30,
        #weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        #push_to_hub=True,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # train
    trainer.train()

    trainer.evaluate()

if __name__ == "__main__":

    main()
