# Общие
import os
import gc
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import Trainer, TrainingArguments
from torch.cuda.amp import GradScaler, autocast
from trl import SFTTrainer, SFTConfig 
from torch.utils.data import DataLoader
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Для оценки
import re
import string
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm 
from collections import Counter

# =======================================
# Переменные
# =======================================

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
# SQuAD_2_path = '../ft_v1/cdatasets/' # Путь до датасета SQuAD 2.0

# datasetSQUAD2 = load_dataset("rajpurkar/squad_v2", cache_dir=SQuAD_2_path)

# train_dataset = datasetSQUAD2['train']
# val_dataset = datasetSQUAD2['validation']


# print(f"Size train part: {len(train_dataset)}")
# print(f"Size validation part: {len(val_dataset)}")


# =======================================
# Загрузка Domein датасета
# =======================================

train_dataset = load_dataset('json', data_files='../model_ft/cdatasets/custom_gen_dataset_train.json')['train']
val_dataset = load_dataset('json', data_files='../model_ft/cdatasets/custom_gen_dataset_val.json')['train']

print(f"Size train part: {len(train_dataset)}")
print(f"Size validation part: {len(val_dataset)}")

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
tokenizer.pad_token_id = tokenizer.eos_token_id # Токен отступа (Особенность: свой для каждой языковой модели)
tokenizer.padding_side = CONFIG['model']['tokenizer']['padding_size'] #Добавлять до максимальной длинны справа


model = AutoModelForCausalLM.from_pretrained(
    CONFIG['model']['cache_dir'],
    # CONFIG['model']['name'], 
    # cache_dir=CONFIG['model']['cache_dir'],
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config, #Если юзать конфиг квантования - модель автоматом на GPU
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

# #перевод модели в режим тренировки и добавление адаптеров
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# =======================================
# Продолжить обучение с последней сохраненной контрольной точки (С помощью вот этого получилось продолжить обучение )
# оно хотя бы началось, с PeftModel не начиналось (но для оценки модели - используется)
# =======================================

# lora_adapter_path_base = "./saved_models/train__14-06-2025_06-57-34__Llama-3.1-8B-Instruct"
# checkpoint_full_parh = f"{lora_adapter_path_base}/{CONFIG['evaluate_model']['checkpoint']}"

# model.load_adapter(checkpoint_full_parh, adapter_name="default")


# =======================================
# Формирование датасетов
# =======================================

train_dataset = dataset_preprocess_for_SF(
    train_dataset, 
    tokenizer, 
    answer_pattern = "### answer:\n", 
    max_len_seq = CONFIG['model']['max_length'], 
    eval=False
)

# sub_train_dataset = train_dataset.select(range(12800))
#Формируем валидационный датасет, для отслеживания потерь
sub_val_dataset = val_dataset

sub_val_dataset = dataset_preprocess_for_SF(
    sub_val_dataset, 
    tokenizer, 
    answer_pattern = "### answer:\n", 
    max_len_seq = CONFIG['model']['max_length'], 
    eval=True
)

print(f"len train: {len(train_dataset)}")
print(f"len val: {len(sub_val_dataset)}")


# =======================================
# Создание логгера
# =======================================

time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
PREFIX_PATH_FOR_MODEL_DIRS = f'{CONFIG['logs']['dir']}/train__{time_stamp}__{CONFIG['model']['model_name_log']}'

os.makedirs(PREFIX_PATH_FOR_MODEL_DIRS, exist_ok=True)
mylogger = CustomLogger(log_dir=PREFIX_PATH_FOR_MODEL_DIRS)

mylogger.log(f"""Start logger\n------------ CONFIGURATE ------------ \n{CONFIG}\n------------ ------------""")

mylogger.log("train merge model with SQuAD Adapter on Domen dataset")


# Создаем функцию для передачи в SFTrainer
compute_metrics_fn = build_compute_metrics_fn(
    tokenizer=tokenizer, 
    answer_pattern=CONFIG["model"]["tokenizer"]["answer_pattern"], 
    mylogger=mylogger # Передаем твой логгер для отладки
)


answer_pattern_for_collator = CONFIG["model"]["tokenizer"]["answer_pattern"]
response_template_ids = tokenizer.encode(answer_pattern_for_collator, add_special_tokens=False)

collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


trainer = CreateTrainer_SF(
    model, 
    tokenizer, 
    train_dataset, 
    sub_val_dataset,  
    CONFIG, 
    RANDOM_STATE, 
    logger=mylogger,
    data_collator=collator,
    compute_metrics=compute_metrics_fn
)

#Запускаем обучение
mylogger.log(f"Start training for model {CONFIG['model']['model_name_log']}")
#Для продолжения обучения: trainer.train(resume_from_checkpoint=checkpoint_full_parh)
trainer.train()
mylogger.log(f"Finished training for model {CONFIG['model']['model_name_log']}")