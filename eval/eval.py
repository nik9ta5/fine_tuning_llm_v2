# =======================================
# Функции для оценки модели
# =======================================

# Для оценки
import re
import string
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm 
from collections import Counter


# Вспомогательные функции для EM/F1 (как выше)
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact_match(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(prediction_tokens) #Кол-во верных из всех предказанных токенов
    recall = num_same / len(ground_truth_tokens) #Кол-во верных из всех верных токенов
    f1 = (2 * precision * recall) / (precision + recall) #Ср. гармн.
    return f1 

def build_compute_metrics_fn(tokenizer, answer_pattern: str, mylogger=None):
    """
    Создает функцию compute_metrics, которая будет использоваться в Trainer.
    Принимает tokenizer и шаблон ответа для корректной обработки predictions и labels.
    """
    
    def custom_compute_metrics(eval_pred) -> dict:
        mylogger.log("\n begin func 'custom_compute_metrics' \n")
        mylogger.log("EVAL PRED")
        mylogger.log(eval_pred)

        #Распаковываем
        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
        
        # 1. Преобразование логитов в ID токенов
        if predictions.ndim == 3: # Если predictions - это логиты (batch, seq_len, vocab_size)
            predictions = predictions.argmax(axis=-1) # (batch, seq_len)
        
        if mylogger: #creatle log
            mylogger.log("\n PREDICTIONS - EVAL PRED\n")
            mylogger.log(predictions.shape) #5 seq on 512 tokens (vocab_size tokens - loggits)
            mylogger.log("\n LABEL_IDS - EVAL PRED\n")
            mylogger.log(label_ids.shape) #5 seq on 512 tokens (true tokens)
            mylogger.log("\n Greedy select\n")
            mylogger.log(predictions.shape)
            mylogger.log(f"Answer Pattern: '{answer_pattern}'")
        
        predicted_answers = []
        golden_answers = []

        # Для отладки, ограничим количество примеров для полного вывода
        num_debug_samples = 3 

        for i in range(predictions.shape[0]): # Итерируемся по каждой последовательности в батче
            pred_ids = predictions[i]
            label_ids_for_sample = label_ids[i] # Истинные метки для текущего примера

            #decode generated tokens
            full_pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            if mylogger and i < num_debug_samples:
                mylogger.log(f"Full Predicted Text:\n{full_pred_text}")

            #filtered true tokens
            filtered_label_ids = [token_id for token_id in label_ids_for_sample if token_id != -100]
            #decode true tokens
            full_golden_text = tokenizer.decode(filtered_label_ids, skip_special_tokens=True)
            
            if mylogger and i < num_debug_samples:
                mylogger.log(f"Full Golden Text (without -100):\n{full_golden_text}")

            
            # Для предсказаний
            if answer_pattern in full_pred_text:
                pred_ans = full_pred_text.split(answer_pattern)[-1].strip() #Часть после паттерная ответа
            else:
                pred_ans = "" 
                if mylogger and i < num_debug_samples:
                    mylogger.log("Warning: Answer pattern not found in predicted text.")
            predicted_answers.append(pred_ans)

            if answer_pattern in full_golden_text:
                golden_ans = full_golden_text.split(answer_pattern)[-1].strip()
            else:
                golden_ans = full_golden_text.strip() # Если шаблона нет, предполагаем, что это уже чистый ответ
            golden_answers.append(golden_ans)


            if mylogger and i < num_debug_samples:
                mylogger.log(f"Extracted Predicted Answer: '{pred_ans}'")
                mylogger.log(f"Extracted Golden Answer:  '{golden_ans}'")
                mylogger.log(f"EM (this sample): {compute_exact_match(pred_ans, golden_ans)}")
                mylogger.log(f"F1 (this sample): {compute_f1(pred_ans, golden_ans)}")

        # 5. Вычисление средних метрик по батчу
        
        mylogger.log("before EM and F1 calculate")
        mylogger.log(predicted_answers[0])
        mylogger.log(golden_answers[0])
        ems = [compute_exact_match(p, g) for p, g in zip(predicted_answers, golden_answers)]
        f1s = [compute_f1(p, g) for p, g in zip(predicted_answers, golden_answers)]

        avg_em = np.mean(ems)
        avg_f1 = np.mean(f1s)
        
        if mylogger:
            mylogger.log(f"\n--- Batch Metrics ---")
            mylogger.log(f"Average EM: {avg_em:.4f}")
            mylogger.log(f"Average F1: {avg_f1:.4f}")
            mylogger.log(f"--- End compute_metrics_for_qa ---")

        return {"em": avg_em, "f1": avg_f1}
    
    return custom_compute_metrics