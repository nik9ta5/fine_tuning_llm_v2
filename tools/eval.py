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



def normalize_text(text):
    """
    Нормализует текст: нижний регистр, удаление пробелов и пунктуации
    """
    text = text.lower() # Приводим к нижнему регистру
    text = text.translate(str.maketrans("", "", string.punctuation)) # Удаляем пунктуацию
    text = re.sub(r'\s+', ' ', text).strip() # Удаляем лишние пробелы
    return text

def compute_exact_match(prediction, ground_truth):
    """
    Вычисляет Exact Match для одного примера
    Принимает 2 строки для сравнения
    """
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def compute_f1_score(prediction, ground_truth):
    """
    Вычисляет F1 Score для одного примера
    Принимает 2 строки для сравнения
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    if len(pred_tokens) == 0 and len(truth_tokens) == 0: # Если оба ответа пустые, F1 = 1
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0: # Если один из ответов пустой, F1 = 0
        return 0.0
    
    # Находим общие токены
    common_tokens = set(pred_tokens) & set(truth_tokens)
    tp = len(common_tokens)
    
    precision = tp / len(pred_tokens)  # Precision = TP / (TP + FP), где FP = предсказанные токены, не входящие в правильные
    recall = tp / len(truth_tokens) # Recall = TP / (TP + FN), где FN = правильные токены, не входящие в предсказанные
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_evaluate_metrics_EM_F1(em_all, f1_all):
    ''' 
    Функция для вычисления среднего значения EM и F1
    '''
    em_all = np.array(em_all)
    f1_all = np.array(f1_all)
    print(f"EM: {em_all.mean()}, F1: {f1_all.mean()}")
    return em_all.mean(), f1_all.mean()


def evaluate_model_for_metrics(model, tokenizer, dataloader, device, max_new_tokens=None, temp=None, logger=None, pfraze_no_answ=None):
    """Функция для оценки модели по метриками EM, F1 """
    em_all = []
    f1_all = []

    if logger:
        logger.log("Test model. Calculate metrics")

    for batch, item in enumerate(tqdm(dataloader, desc="Test model")):
        with autocast(dtype=torch.bfloat16): #Использовать смешанную точность (Работает)
            #Генерируем батч
            pred = model.generate(
                input_ids=item['input_ids'].to(device),
                attention_mask=item['attention_mask'].to(device),
                max_new_tokens=max_new_tokens
            )

            tensor_shape_1 = item['input_ids'].shape[1]
            answers_model_text = tokenizer.batch_decode(pred[:, tensor_shape_1:], skip_special_tokens=True)
            
            answers_model_text = [item.split("###")[0] for item in answers_model_text] #Модель генерирует часть инструкции
            ### instructions:
            # item['answer']

            #Лишний цикл, но если внести в циклы которые ниже, то ничего не вычисляет
            #только с помощью этого стало нормально определять, если 
            # answers_model_text = ["" if normalize_text(item) == "no answer" else item for item in answers_model_text]

            # =====================
            # Модель учится отвечать, если не может ответить: "No Answer", следовательно, если ответ модели No Answer и в эталонном ответе нет ответа = "", то в функцию для вычисления метрики передаем ответ модели (No Answer) и как эталонный - No Answer
            # При формировании промпта идет четкое указание No Answer, если ответа нет
            # =====================

            
            tmp_em = [compute_exact_match(predict, answer) #Когда расчитываем руками - ответ модели расцениваем как "" и эталон ""
                      # if normalize_text(predict) == "no answer" else compute_exact_match(predict, answer)
                      for predict, answer in zip(answers_model_text, item['answer'])
            ]
            tmp_f1 = [compute_f1_score(predict, answer)
                      # if normalize_text(predict) == "no answer" else compute_f1_score(predict, answer)
                      for predict, answer in zip(answers_model_text, item['answer'])
             ]
            
            if logger:
                for predict, answer, emtmp, f1tmp in zip(answers_model_text, item['answer'], tmp_em, tmp_f1):
                    logger.log(f"\nPRED:{predict}\nANSW:{answer}\nEM:{emtmp}\nF1:{f1tmp}\n")
        
        em_all += tmp_em
        f1_all += tmp_f1
    
    em, f1 = calculate_evaluate_metrics_EM_F1(em_all, f1_all)
    if logger:
        logger.log(f"em_all_len: {len(em_all)} f1_all_len: {len(f1_all)}")
        logger.log(f"EM: {em} F1: {f1}")
    return em_all, f1_all, em, f1