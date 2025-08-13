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


# ============= Импорты из tools =============
from tools.logService import CustomLogger
from tools.promptService import second_prompt_template, second_prompt_template_dict
from tools.dataset_utils import get_dataset
from tools.quant_utils import create_quant_config
from tools.lora_utils import create_lora_config
from tools.tuning import CreateTrainer_SF
from tools.eval import build_compute_metrics_fn
from tools.preprocess import dataset_preprocess
from tools.eval import evaluate_model_for_metrics

# ============= Отключаем варнинги =============
import warnings
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization", category=UserWarning)
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.",category=UserWarning)


# ============= Переменные =============
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.seed = RANDOM_STATE
torch.set_float32_matmul_precision('high')


# ============= Загрузка конфига =============
with open(f'./configs/train_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)


def system_info_print() -> str:
    system_str = f"""
==================================================================
* SYSTEM INFO *
mem alloc:          {(torch.cuda.memory_allocated()//(1024**2)):.3f} MB
mem reser:          {(torch.cuda.memory_reserved()//(1024**2)):.3f} MB
mem max mem alloca: {(torch.cuda.max_memory_allocated()//(1024**2)):.3f} MB
mem max mem reserv: {(torch.cuda.max_memory_reserved()//(1024**2)):.3f} MB
==================================================================
"""
    return system_str



# ============= Запуск =============

if __name__ == "__main__":

    # ----- Создаем директорию для логов -----
    dir_name_for_log = CONFIG['logs']["dir"] #В КОНФИГ
    os.makedirs(dir_name_for_log, exist_ok=True)


    # ----- Создаем логгер -----
    logger = CustomLogger(
        dir_name_for_log, 
        log_file_name=CONFIG['logs']["finemae"], #В КОНФИГ
        logging_lavel=logging.DEBUG,
        outputConsole=True
    )
    myLogger = logger.getLogger()
    myLogger.debug("test output log")


    # ----- Загрузка датасета SQuAD 2.0 -----
    #SQuAD 2.0
    path2DS_SQUAD2 = '../clear_docs/squad_cache'
    dataSet = get_dataset()


    # ----- Формирование датасета с промптами -----
    SYSTEM_INSTRICTION = """You are a helper, an expert in machine learning and artificial intelligence, designed to help answer questions.
Answer the question using the context provided, without using your knowledge. Ignore any knowledge gained prior to this conversation.
Context is the relevant fragments of documents found by the search that should be used to answer.
The answer should be a word, phrase, or sentence contained in the context.
If the context does not answer the question, answer 'No answer.'"""

    myLogger.debug(SYSTEM_INSTRICTION)

    train_dataset = dataSet['train']
    val_dataset = dataSet['validation']

    os.makedirs(CONFIG['data']["path"], exist_ok=True)

    # train_dataset = train_dataset.map(
    #     lambda x : second_prompt_template_dict(SYSTEM_INSTRICTION, x),
    #     num_proc=1,
    #     cache_file_name=CONFIG['data']["cache_file_name_train"],
    #     remove_columns=train_dataset.column_names
    # )
    val_dataset = val_dataset.map(
        lambda x : second_prompt_template_dict(SYSTEM_INSTRICTION, x),
        num_proc=1,
        cache_file_name=CONFIG['data']["cache_file_name_val"],
        remove_columns=val_dataset.column_names
    )
    # train_dataset = train_dataset.select(range(1000))
    val_dataset = val_dataset.select(range(100))

    # ----- Конфигурация квантования -----
    quant_config = create_quant_config()

    # ----- Загрузка модели и токенайзера -----
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model']['full_path_for_model']
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = CONFIG["model"]["tokenizer"]["padding_size"]
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model']['full_path_for_model'],
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ----- Загрузка LoRA конфига -----
    lora_config = create_lora_config()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.to(DEVICE)

    # ----- Загрузка обученных адаптеров -----
    lora_adapter_path_base = CONFIG['tuning_model']['full_path2adapters']
    checkpoint_full_parh = f"{lora_adapter_path_base}/{CONFIG['tuning_model']['checkpoint']}"

    model = PeftModel.from_pretrained(model, checkpoint_full_parh)

    # print system info
    myLogger.debug(system_info_print())

    myLogger.debug(str(model))


    # =======================================
    # Формирование датасета для оценки
    # =======================================

    val_dataset = dataset_preprocess(
        val_dataset, 
        tokenizer, 
        answer_pattern = CONFIG['model']['tokenizer']['answer_pattern'], 
        max_len_seq = CONFIG['model']['max_length']
    )

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
        val_dataset,                # Датасет (например, tokenized_dataset)
        batch_size=CONFIG['train']['val_batch'],         # Размер батча
        shuffle=False,         # Перемешивать данные
        num_workers=0,         # Количество потоков для загрузки
        collate_fn=custom_collate_fn,       # Функция для сборки батча
        pin_memory=False,      # Копировать данные в CUDA-память
        drop_last=False,       # Отбрасывать последний неполный батч
        prefetch_factor=None,     # Количество батчей для предварительной загрузки (сколько батчей будет загружено сразу для  ускорения, только в параллельном режиме (когда num_workers > 0)) (None - если не используем)
        persistent_workers=False  # Сохранять рабочие потоки между итерациями
    )


    # =======================================
    # Запуск оценки
    # =======================================

    myLogger.debug(f"\n\n\nStart eval for model {CONFIG['model']['model_name_log']}\n\n\n")
    evaluate_model_for_metrics(
        model, 
        tokenizer, 
        val_dataloader, 
        DEVICE, 
        max_new_tokens=CONFIG['model']['max_new_tokens'], 
        temp=CONFIG['inference']['temp'], 
        logger=logger, 
        pfraze_no_answ="No Answer"
    )
    myLogger.debug(f"\n\n\nFinished eval for model {CONFIG['model']['model_name_log']}\n\n\n")














