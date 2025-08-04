# Общие
import os
import yaml
from datetime import datetime
from datasets import load_dataset
from llama_cpp import Llama 
from tqdm import tqdm
import re
import string
import numpy as np
import logging # Для логера

# =======================================
# Отключим варнинги (если появятся специфичные для llama.cpp, добавим)
# =======================================
import warnings
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization", category=UserWarning)
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.",category=UserWarning)

# Поскольку llama.cpp сам управляет GPU, напрямую torch.cuda не используем для DEVICE
# RANDOM_STATE не так критичен для llama.cpp, но можно использовать для seed в Llama инициализации
RANDOM_STATE = 42

# =======================================
# Загрузка конфига
# =======================================
with open(f'./config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

# =======================================
# Загрузка SQuAD 2.0
# =======================================

SQuAD_2_path = '../ft_v1/cdatasets/'
datasetSQUAD2 = load_dataset("rajpurkar/squad_v2", cache_dir=SQuAD_2_path)

train_dataset = datasetSQUAD2['train']
val_dataset = datasetSQUAD2['validation']

print(f"Size train part: {len(train_dataset)}")
print(f"Size validation part: {len(val_dataset)}")

# =======================================
# Загрузка Domein датасета (закомментировано, как и у вас)
# =======================================
# train_dataset = load_dataset('json', data_files='../model_ft/cdatasets/custom_gen_dataset_train.json')['train']
# val_dataset = load_dataset('json', data_files='../model_ft/cdatasets/custom_gen_dataset_val.json')['train']
# print(f"Size train part: {len(train_dataset)}")
# print(f"Size validation part: {len(val_dataset)}")


# =======================================
# Загрузка модели Llama.cpp
# =======================================

MODEL_GGUF_PATH = "./merge_models/base_base_Llama-3.1-8B-Instruct_15-06-2025_06-26-46_SQuAD_Adapters_18-06-2025_18-13-07_domen_Adapters/model.gguf"

if not os.path.exists(MODEL_GGUF_PATH):
    raise FileNotFoundError(f"Файл GGUF модели не найден по пути: {MODEL_GGUF_PATH}")

llm = Llama(
    model_path=MODEL_GGUF_PATH,
    n_gpu_layers=-1, # Загружаем все слои на GPU. Если не хватает VRAM, уменьшите.
    n_ctx=CONFIG['model']['max_length'], # Используем n_ctx из вашего конфига
    verbose=True, # Включаем подробный вывод для отладки
    seed=RANDOM_STATE # Для воспроизводимости генерации
)

print(f"Модель Llama.cpp загружена из: {MODEL_GGUF_PATH}")
print(f"Контекст модели (n_ctx): {llm.n_ctx()}") # Проверяем, какой n_ctx был установлен


# Новый промпт для Llama 3.1 Instruct
# Для create_completion, вы должны сами формировать полный промпт,
# включая токены начала/конца и роли.
# INSTRUCTION из вашего кода
INSTRUCTION = 'Answer the question using context only. The answer must be an exact quote from the context and not include any additional information. If the question cannot be answered using context only, answer "No Answer"'

def prompt_template_EVALUATE(context, question):
    # Полный шаблон Llama 3.1 Instruct для create_completion
    # <|begin_of_text|>
    # <|start_header_id|>system<|end_header_id|>
    #
    # {INSTRUCTION}
    # <|eot_id|>
    # <|start_header_id|>user<|end_header_id|>
    #
    # context: {context}
    # question: {question}
    # <|eot_id|>
    # <|start_header_id|>assistant<|end_header_id|>
    #
    # answer:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{INSTRUCTION}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"context:\n{context}\n\nquestion:\n{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"answer:" # Здесь модель должна начать генерировать ответ
    )

# Функция, если юзать SFTrainer (переработана для Llama.cpp)
def prompt_answer_ft(batch):
    answers = [
        item['text'][0] if item['text'] else "No Answer"
        for item in batch['answers']
    ]
    prompts = [
        prompt_template_EVALUATE(context, question)
        for context, question in zip(batch['context'], batch['question'])
    ]
    return {
        "prompt": prompts,
        "answers": answers
    }

# В llama.cpp нет необходимости в токенизации датасета на этапе подготовки,
# так как модель сама будет токенизировать входные строки на лету.
# Поэтому `tokenized_text` и `dataset_preprocess` будут сильно упрощены.
# Нам нужны только сырые промпты и ответы.

def dataset_preprocess(dataset):
    # Просто формируем промпты и ответы, никаких токенов или тензоров
    dataset = dataset.map(
        prompt_answer_ft,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
    )
    return dataset

# =======================================
# Формирование датасета для оценки
# =======================================

sub_val_dataset = val_dataset
sub_val_dataset = dataset_preprocess(sub_val_dataset)
print(f"len val: {len(sub_val_dataset)}")

# =======================================
# Формирование "даталоадера" (теперь просто итератор по батчам)
# =======================================
sub_val_dataset_list = list(sub_val_dataset) # Преобразование Dataset в список словарей


class SimpleDataLoader:
    def __init__(self, dataset_list, batch_size): # Принимаем список словарей
        self.dataset_list = dataset_list # Используем dataset_list
        self.batch_size = batch_size

    def __iter__(self):
        # Итерируемся по индексу
        for i in range(0, len(self.dataset_list), self.batch_size):
            # Извлекаем батч как срез из списка словарей
            batch = self.dataset_list[i:i + self.batch_size]
            
            prompts = [item['prompt'] for item in batch]
            answers = [item['answers'] for item in batch]
            yield {"prompt": prompts, "answer": answers}

    def __len__(self):
        return (len(self.dataset_list) + self.batch_size - 1) // self.batch_size

val_dataloader = SimpleDataLoader(
    sub_val_dataset_list, # Передаем список словарей
    batch_size=CONFIG['train']['val_batch']
)

print(f"size val dataloader: {len(val_dataloader)}")

# =======================================
# Кастомный логгер (остается почти без изменений)
# =======================================

class CustomLogger:
    def __init__(self, log_dir):
        self._timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self._log_filename = f"{log_dir}/log_run_{self._timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self._log_filename),
            ],
            encoding='UTF-8'
        )
        self._my_logger = logging.getLogger("my_custom_logger")
        self._my_logger.setLevel(logging.INFO)

    def log(self, text_for_log : str):
        self._my_logger.info(text_for_log)

# =======================================
# Функции для вычисления EM и F1 (без изменений)
# =======================================

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_exact_match(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def compute_f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common_tokens = set(pred_tokens) & set(truth_tokens)
    tp = len(common_tokens)

    precision = tp / len(pred_tokens)
    recall = tp / len(truth_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_evaluate_metrics_EM_F1(em_all, f1_all):
    em_all = np.array(em_all)
    f1_all = np.array(f1_all)
    print(f"EM: {em_all.mean()}, F1: {f1_all.mean()}")
    return em_all.mean(), f1_all.mean()


# =======================================
# Функция оценки модели для llama.cpp
# =======================================

def evaluate_model_for_metrics_llama_cpp(llm_model, dataloader, max_new_tokens=None, temp=None, logger=None, pfraze_no_answ=None):
    em_all = []
    f1_all = []

    if logger:
        logger.log("Test model. Calculate metrics using llama.cpp")

    # Для llama.cpp мы можем обрабатывать батчи последовательно,
    # или генерировать их параллельно, если llm.create_completion поддерживает батчинг.
    # В llama-cpp-python create_completion() не поддерживает батчинг напрямую для разных промптов,
    # поэтому мы будем итерироваться по промптам в батче и генерировать по одному.
    # Это важно для корректного замера времени.
    # `autocast` не нужен, т.к. llama.cpp сам управляет типами данных.

    for batch_data in tqdm(dataloader, desc="Test model"):
        prompts = batch_data['prompt']
        ground_truth_answers = batch_data['answer']
        
        answers_model_text = []

        for prompt in prompts:
            # Генерация ответа для каждого промпта
            # Здесь можно добавить параметры temperature, top_p, stop_words
            # Stop words особенно важны для Llama 3.1
            try:
                output = llm_model.create_completion(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temp,
                    # Можно добавить top_p=CONFIG['inference']['top_p']
                    stop=["<|eot_id|>"], # Важно для Llama 3.1
                    # verbose=False # Отключите verbose для каждой генерации, чтобы не засорять лог
                )
                # Извлекаем сгенерированный текст.
                # Llama.cpp по умолчанию выдает текст после промпта
                # Использование .strip() для удаления возможных лишних пробелов/переносов
                generated_text = output["choices"][0]["text"].strip()

                # Ваша логика для отрезания части инструкции "###"
                # (Если модель случайно сгенерирует часть следующей инструкции)
                generated_text = generated_text.split("###")[0].strip()

                answers_model_text.append(generated_text)
            except Exception as e:
                # Если произошла ошибка при генерации (например, промпт слишком длинный)
                logger.log(f"Ошибка при генерации для промпта: {prompt[:100]}... Ошибка: {e}")
                answers_model_text.append("") # Добавляем пустую строку, чтобы не ломать zip

        # Обработка ответов для метрик
        tmp_em = []
        tmp_f1 = []
        for predict, answer in zip(answers_model_text, ground_truth_answers):
            # Ваша логика с "No Answer"
            # if normalize_text(predict) == "no answer" and normalize_text(answer) == "no answer":
            #     em_score = 1.0
            #     f1_score = 1.0
            # elif normalize_text(predict) == "no answer": # Модель ответила "No Answer", а эталон - нет
            #     em_score = 0.0
            #     f1_score = 0.0
            # else:
            #     em_score = compute_exact_match(predict, answer)
            #     f1_score = compute_f1_score(predict, answer)

            em_score = compute_exact_match(predict, answer)
            f1_score = compute_f1_score(predict, answer)

            tmp_em.append(em_score)
            tmp_f1.append(f1_score)


        if logger:
            for predict, answer, emtmp, f1tmp in zip(answers_model_text, ground_truth_answers, tmp_em, tmp_f1):
                logger.log(f"\nPRED:{predict}\nANSW:{answer}\nEM:{emtmp}\nF1:{f1tmp}\n")

        em_all.extend(tmp_em)
        f1_all.extend(tmp_f1)

    em, f1 = calculate_evaluate_metrics_EM_F1(em_all, f1_all)
    if logger:
        logger.log(f"em_all_len: {len(em_all)} f1_all_len: {len(f1_all)}")
        logger.log(f"EM: {em} F1: {f1}")
    return em_all, f1_all, em, f1


# =======================================
# Создание логгера
# =======================================

time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
PREFIX_PATH_FOR_MODEL_DIRS = f'{CONFIG['logs']['dir']}/eval__{time_stamp}__{CONFIG['model']['model_name_log']}_llama_cpp'

os.makedirs(PREFIX_PATH_FOR_MODEL_DIRS, exist_ok=True)
mylogger = CustomLogger(log_dir=PREFIX_PATH_FOR_MODEL_DIRS)

mylogger.log(f"""Start logger\n------------ CONFIGURATE ------------ \n{CONFIG}\n------------ ------------""")
mylogger.log(f"MODEL GGUF PATH: {MODEL_GGUF_PATH}") # Информация о GGUF модели

# Вывод информации о модели llama.cpp (не так детализировано, как для HF Transformers)
# mylogger.log(f"\nLlama.cpp Model Info: n_ctx={llm.n_ctx()}, n_gpu_layers={llm.n_gpu_layers()}\n")


# =======================================
# Запуск оценки
# =======================================

mylogger.log(f"Start eval for model {CONFIG['model']['model_name_log']} (llama.cpp)")
evaluate_model_for_metrics_llama_cpp(
    llm, # Передаем объект Llama
    val_dataloader,
    max_new_tokens=CONFIG['model']['max_new_tokens'],
    temp=CONFIG['inference']['temp'],
    logger=mylogger,
    pfraze_no_answ="No Answer" # Этот параметр у вас не используется внутри функции
)
mylogger.log(f"Finished eval for model {CONFIG['model']['model_name_log']} (llama.cpp)")