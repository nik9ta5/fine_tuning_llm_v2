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
# Конфиг квантования
# =======================================

# Когда модель мерджится - никаких конфигов квантования

# =======================================
# Загрузка токенайзера и модели
# =======================================

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG['model']['cache_dir']
)
tokenizer.pad_token_id = 128001 #tokenizer.eos_token_id # Токен отступа (Особенность: свой для каждой языковой модели)
tokenizer.padding_side = CONFIG['model']['tokenizer']['padding_size'] #Добавлять до максимальной длинны справа


model = AutoModelForCausalLM.from_pretrained(
    CONFIG['model']['cache_dir'],
    torch_dtype=torch.bfloat16,
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

#Мерджим модель с адаптерами (обученными на синтетическом датасете)
lora_adapter_path_base = "./saved_models/train__15-06-2025_14-38-14__base_Llama-3.1-8B-Instruct_15-06-2025_06-26-46_SQuAD_Adapters/"
checkpoint_full_parh = f"{lora_adapter_path_base}/{CONFIG['evaluate_model']['checkpoint']}"

model = PeftModel.from_pretrained(model, checkpoint_full_parh)


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
# Создание логгера
# =======================================

time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
PREFIX_PATH_FOR_MODEL_DIRS = f'{CONFIG['logs']['dir']}/merge__{time_stamp}__{CONFIG['model']['model_name_log']}'

os.makedirs(PREFIX_PATH_FOR_MODEL_DIRS, exist_ok=True)
mylogger = CustomLogger(log_dir=PREFIX_PATH_FOR_MODEL_DIRS)

mylogger.log(f"""Start logger\n------------ CONFIGURATE ------------ \n{CONFIG}\n------------ ------------""")
#Информация о том, какое сохранение оцениваем
mylogger.log(f"MODEL BASE MERGE : {checkpoint_full_parh}")

#Вывод информации о модели
mylogger.log(f"\nModel arch:\n{str(model)}\n\nModel config:\n{model.config}\n")


# =======================================
# Мерджим
# =======================================

model = model.merge_and_unload()


path2save_merge = f"{CONFIG['merge']['dir']}/base_{CONFIG['model']['model_name_log']}_{time_stamp}_SQuAD_Adapters"
os.makedirs(path2save_merge, exist_ok=True)

mylogger.log(f"\nModel arch:\n{str(model)}\n\nModel config:\n{model.config}\n")
# =======================================
# Сохраняем модель
# =======================================

#Сохраняем модель
model.save_pretrained(path2save_merge)
tokenizer.save_pretrained(path2save_merge)

mylogger.log(f"Model saved")
