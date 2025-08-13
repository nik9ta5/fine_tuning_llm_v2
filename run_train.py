import os
import yaml
import torch
import logging

from datasets import load_dataset
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from trl import DataCollatorForCompletionOnlyLM


# ============= Импорты из модулей =============
from tools.logService import CustomLogger
from tools.promptService import second_prompt_template, second_prompt_template_dict
from tools.dataset_utils import get_dataset
from tools.quant_utils import create_quant_config
from tools.lora_utils import create_lora_config
from tools.tuning import CreateTrainer_SF
from tools.eval import build_compute_metrics_fn

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

    train_dataset = train_dataset.map(
        lambda x : second_prompt_template_dict(SYSTEM_INSTRICTION, x),
        num_proc=1,
        cache_file_name=CONFIG['data']["cache_file_name_train"],
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x : second_prompt_template_dict(SYSTEM_INSTRICTION, x),
        num_proc=1,
        cache_file_name=CONFIG['data']["cache_file_name_val"],
        remove_columns=val_dataset.column_names
    )
    train_dataset = train_dataset.select(range(1000))
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

    # ----- Загрузка LoRA конфига -----
    lora_config = create_lora_config()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.to(DEVICE)

    # print system info
    myLogger.debug(system_info_print())

    myLogger.debug(str(model))

       
    # ----- запуск trainer -----

    #Паттерн ответа
    answer_pattern_for_collator = CONFIG['model']['tokenizer']['answer_pattern']
    response_template_ids = tokenizer.encode(answer_pattern_for_collator, add_special_tokens=False)

    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    #Функция для вычисления метрик
    # func_em = build_compute_metrics_fn(
    #     tokenizer,
    #     answer_pattern_for_collator,
    #     logger
    # )
    
    #Создаем trainer
    trainer = CreateTrainer_SF(
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset,  
        CONFIG, 
        RANDOM_STATE, 
        logger=logger,
        data_collator=collator,
        # compute_metrics=func_em
    )

    #Запускаем обучение
    trainer.train()