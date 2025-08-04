# Общие
import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# Для оценки
import re
import string
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm 
# Для логера
import logging
from datetime import datetime
from transformers import TrainerCallback
from collections import Counter
from torch.utils.data import DataLoader


# =======================================
# Отключим варнинги от 
# =======================================
import warnings
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization", category=UserWarning)
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.",category=UserWarning)




os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.seed = RANDOM_STATE
torch.set_float32_matmul_precision('high')



# =======================================
# Загрузка конфига
# =======================================
with open(f'./config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

# =======================================
# Загрузка SQuAD 2.0
# =======================================

# #Пути до датасетов
SQuAD_2_path = '../ft_v1/cdatasets/' # Путь до датасета SQuAD 2.0

datasetSQUAD2 = load_dataset("rajpurkar/squad_v2", cache_dir=SQuAD_2_path)

train_dataset = datasetSQUAD2['train']
val_dataset = datasetSQUAD2['validation']


print(f"Size train part: {len(train_dataset)}")
print(f"Size validation part: {len(val_dataset)}")

# =======================================
# Загрузка Domein датасета
# =======================================

# train_dataset = load_dataset('json', data_files='../model_ft/cdatasets/custom_gen_dataset_train.json')['train']
# val_dataset = load_dataset('json', data_files='../model_ft/cdatasets/custom_gen_dataset_val.json')['train']

# print(f"Size train part: {len(train_dataset)}")
# print(f"Size validation part: {len(val_dataset)}")

# =======================================
# Конфиг квантования
# =======================================

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,                     # Загрузить модель в 8-битном формате
    bnb_8bit_quant_type="int8",             # Тип 8-битного квантования (может быть nf8 или int8)
    bnb_8bit_compute_dtype=torch.bfloat16, # Тип данных для вычислений в 8-битном режиме
    bnb_8bit_use_double_quant=False        # Использовать ли двойное квантование
)

# =======================================
# Загрузка токенайзера и модели
# =======================================

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG['model']['cache_dir']
    # CONFIG['model']['name'], 
    # cache_dir=CONFIG['model']['cache_dir']
)
tokenizer.pad_token_id = 128001 #tokenizer.eos_token_id # Токен отступа (Особенность: свой для каждой языковой модели)
tokenizer.padding_side = CONFIG['model']['tokenizer']['padding_size'] #Добавлять до максимальной длинны справа


model = AutoModelForCausalLM.from_pretrained(
    CONFIG['model']['cache_dir'],
    # CONFIG['model']['name'], 
    # cache_dir=CONFIG['model']['cache_dir'],
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    attn_implementation="eager"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# =======================================
# Загрузка LoRA конфига
# =======================================

lora_config = LoraConfig(
    r=CONFIG['lora']['r'],                            # ранг матриц адаптеров
    lora_alpha=CONFIG['lora']['lora_alpha'],          # коэффициент масштабирования LoRA
    target_modules=CONFIG['lora']['target_modules'],  # Модули для применения LoRA
    lora_dropout=CONFIG['lora']['lora_dropout'],      # Dropout для адаптеров LoRA
    bias="none",                                      # Тип применения смещения (bias)
    task_type="CAUSAL_LM",                            # Тип задачи
)

# =======================================
# Загрузка сохраненных LoRA адаптеров (в процессе обучения)
# =======================================

lora_adapter_path_base = "./saved_models/train__15-06-2025_14-38-14__base_Llama-3.1-8B-Instruct_15-06-2025_06-26-46_SQuAD_Adapters"
checkpoint_full_parh = f"{lora_adapter_path_base}/{CONFIG['evaluate_model']['checkpoint']}"

model = PeftModel.from_pretrained(model, checkpoint_full_parh)


#Новый промпт
def prompt_template(context, question, answer):
    #Общая инструкция для промпта (Новая инструкция)
    INSTRUCTION = 'Answer the question using context only. The answer must be an exact quote from the context and not include any additional information. If the question cannot be answered using context only, answer "No Answer"'
    return f"""### instructions:\n{INSTRUCTION}\n\n### context:\n{context}\n\n### question:\n{question}\n\n### answer:\n{answer}"""

def prompt_template_EVALUATE(context, question):
    #Общая инструкция для промпта (Новая инструкция)
    #Она нужна для оценки
    INSTRUCTION = 'Answer the question using context only. The answer must be an exact quote from the context and not include any additional information. If the question cannot be answered using context only, answer "No Answer"'
    return f"""### instructions:\n{INSTRUCTION}\n\n### context:\n{context}\n\n### question:\n{question}\n\n### answer:\n"""


#Функция, если юзать SFTrainer
def prompt_answer_ft(batch):
    """ Функция для формирования промпротов для train
    
    (ОСОБЕННОСТЬ: SQuAD формат)
    return (prompt, answer) - кортеж
    """
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

# def prompt_answer_ft(batch):
#     """ Функция для формирования промпротов для train (На доменном датасетае)
#     (ОСОБЕННОСТЬ: SQuAD формат)
#     """
#     answers = [
#         item if item else "No Answer"
#         for item in batch['answers']
#     ]
#     prompts = [
#         prompt_template_EVALUATE(context, question) 
#         for context, question in zip(batch['context'], batch['question'])
#     ]
#     return {
#         "prompt": prompts,
#         "answers":answers
#     }
    

def tokenized_text(batch, tokenizer, answer_pattern = None, max_len_seq = None):
    """ Функция для токенизации промптов

    return: input_ids, attention_mask, labels, answers
    """
    prompts = batch['prompt'] #Промпты на естественном языке
    answers = batch['answers'] #Чисто ответы на естественном языке
    
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=max_len_seq,
        truncation=True,
        padding="max_length"
    ) #input_ids, attention_mask
    
    #Для лейблов - клонируем тензор токенизированного промпта
    labels = encodings['input_ids'].clone()
    for i in range(len(prompts)):
        #Находим индекс слова в промпте, которое считается началом ответа для языковой модели
        answer_start = prompts[i].find(answer_pattern)
        
        #Определяем начало слова ответа в токенизированном формате (длинна тензора отличается от длинны промпта как строки)
        answer_start_id = len(prompts[i])
        if answer_start != -1:
            answer_start_id = tokenizer(prompts[i][:answer_start], return_tensors="pt")['input_ids'].size(1)

        #Сколько токенов паддинга
        padding_tokens = max_len_seq - encodings['attention_mask'][i].sum().item() 

        #Что не является ответом - устанавливаем -100 (defualt в PyTorch and HF)
        labels[i, :padding_tokens + answer_start_id] = -100

    encodings['labels'] = labels #Лейблы - с ответами
    encodings['answers'] = answers #Ответы на естественном языке
    return encodings # input_ids, attention_mask, labels, answers
    

def dataset_preprocess(dataset, tokenizer, answer_pattern = None, max_len_seq = None):
    dataset = dataset.map(
        prompt_answer_ft, #Функция, которая применяется ко всем строкам
        batched=True,        #Использовать батчинг
        num_proc=1,         #Количество процессов
        remove_columns=dataset.column_names,  #Удаляем исходные колонки
    ) #формируем пропты для оценки (prompt, answers)
    
    dataset = dataset.map(
        lambda x: tokenized_text(x, tokenizer, answer_pattern = answer_pattern, max_len_seq = max_len_seq),
        batched=True,        #Использовать батчинг
        num_proc=1,         #Количество процессов
        remove_columns=dataset.column_names,  #Удаляем исходные колонки
    ) #токенизируем (input_ids, attention_mask, labels, answers)
    return dataset



# =======================================
# Формирование датасета для оценки
# =======================================

#Проверим на части, потом на всем
sub_val_dataset = val_dataset

sub_val_dataset = dataset_preprocess(
    sub_val_dataset, 
    tokenizer, 
    answer_pattern = "### answer:\n", 
    max_len_seq = CONFIG['model']['max_length']
)
print(f"len val: {len(sub_val_dataset)}")

# =======================================
# Формирование даталоадера
# =======================================
def custom_collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    answers = [item['answers'] for item in batch]
    return {"input_ids": torch.stack(input_ids),"attention_mask": torch.stack(attention_mask),"labels": torch.stack(labels),"answer": answers}
    

val_dataloader = DataLoader(
    sub_val_dataset,                # Датасет (например, tokenized_dataset)
    batch_size=CONFIG['train']['val_batch'],         # Размер батча
    shuffle=False,         # Перемешивать данные
    num_workers=0,         # Количество потоков для загрузки
    collate_fn=custom_collate_fn,       # Функция для сборки батча
    pin_memory=False,      # Копировать данные в CUDA-память
    drop_last=False,       # Отбрасывать последний неполный батч
    prefetch_factor=None,     # Количество батчей для предварительной загрузки (сколько батчей будет загружено сразу для  ускорения, только в параллельном режиме (когда num_workers > 0)) (None - если не используем)
    persistent_workers=False  # Сохранять рабочие потоки между итерациями
)

print(f"size val dataloader: {len(val_dataloader)}")

# =======================================
# Кастомный логгер
# =======================================

# Для отлова логгирования метрик
class CustomLoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Записываем метрики в лог
            self.logger.log(
            "\n" + "\n".join([f"Step {state.global_step}: {key} = {value}" for key, value in logs.items()]))


class CustomLogger:
    def __init__(self, log_dir):
        self._timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self._log_filename = f"{log_dir}/log_run_{self._timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,  # Уровень логирования
            format='%(asctime)s - %(levelname)s - %(message)s',  # Формат сообщений
            handlers=[
                logging.FileHandler(self._log_filename),  # Запись в файл
                # logging.StreamHandler()  # Вывод в консоль (опционально)
            ],
            encoding='UTF-8'
        )

        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self._log_filename)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        transformers_logger.addHandler(file_handler)

        self._my_logger = logging.getLogger("my_custom_logger") #Перехватить логгер transformers
        self._my_logger.setLevel(logging.INFO)


    def log(self, text_for_log : str):
        self._my_logger.info(text_for_log)


# =======================================
# Функции для вычисления EM и F1 в процессе fine-tuning
# =======================================

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




# =======================================
# Создание логгера
# =======================================

time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
PREFIX_PATH_FOR_MODEL_DIRS = f'{CONFIG['logs']['dir']}/eval__{time_stamp}__{CONFIG['model']['model_name_log']}'

os.makedirs(PREFIX_PATH_FOR_MODEL_DIRS, exist_ok=True)
mylogger = CustomLogger(log_dir=PREFIX_PATH_FOR_MODEL_DIRS)

mylogger.log(f"""Start logger\n------------ CONFIGURATE ------------ \n{CONFIG}\n------------ ------------""")
#Информация о том, какое сохранение оцениваем
mylogger.log(f"MODEL CHECKPOINT EVAL: {checkpoint_full_parh}")

#Вывод информации о модели
mylogger.log(f"\nModel arch:\n{str(model)}\n\nModel config:\n{model.config}\n")


# =======================================
# Запуск оценки
# =======================================

mylogger.log(f"Start eval for model {CONFIG['model']['model_name_log']}")
evaluate_model_for_metrics(
    model, 
    tokenizer, 
    val_dataloader, 
    DEVICE, 
    max_new_tokens=CONFIG['model']['max_new_tokens'], 
    temp=CONFIG['inference']['temp'], 
    logger=mylogger, 
    pfraze_no_answ="No Answer"
)
mylogger.log(f"Finished eval for model {CONFIG['model']['model_name_log']}")














