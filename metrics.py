# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


def compute_text_acc(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def compute_equation_acc(preds, labels):
    preds = [eval_equation(pred) for pred in preds]
    labels = [eval_equation(label) for label in labels]

    return np.mean(np.array(preds) == np.array(labels))


def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan

    return answer


def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics


def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics



def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics


def compute_metrics_equation_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics