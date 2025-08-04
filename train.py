import os
import yaml
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from trl import DataCollatorForCompletionOnlyLM

# =======================================
# Импорты из модулей
# =======================================
from logger.custom_logger import CustomLogger # +
from prompts.tuning_prompts import prompt_template # +
from dataset_preprocess.preprocess import prompt_answer_ft_SF_squad, prompt_answer_ft_SF_domen, dataset_preprocess_for_SF # +
from tuning.tuning import CreateTrainArguments_SF, CreateTrainer_SF # +, +

from eval.eval import normalize_answer
from eval.eval import get_tokens
from eval.eval import compute_exact_match
from eval.eval import compute_f1
from eval.eval import build_compute_metrics_fn



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




def model_generate(model, tokenizer, device, prompt):
    tokenizer_prompt = tokenizer(     
        prompt,
        return_tensors="pt", 
        # truncation=True, 
        # max_length=64
    )
    answer_model = model.generate(
        input_ids=tokenizer_prompt['input_ids'].to(device),
        attention_mask=tokenizer_prompt['attention_mask'].to(device),
        # max_new_tokens=32
    )
    answer = tokenizer.batch_decode(answer_model, skip_special_tokens=True)
    return answer




# =======================================
# Запуск
# =======================================

if __name__ == "__main__":

    # =============== Создаем директорию для логов ===============
    dir_name_for_log = "./new_logs"
    os.makedirs(dir_name_for_log, exist_ok=True)

    # =============== Создаем логгер ===============
    logger = CustomLogger(
        dir_name_for_log, 
        prefix_filename = "test_logger"
    )
    logger.log("init")


    # =============== Загрузка датасета SQuAD 2.0 ===============
    #SQuAD 2.0
    # SQuAD_2_path = '../../../ft_v1/cdatasets/' # Путь до датасета SQuAD 2.0
    # datasetSQUAD2 = load_dataset("rajpurkar/squad_v2", cache_dir=SQuAD_2_path)
    # train_dataset = datasetSQUAD2['train']
    # val_dataset = datasetSQUAD2['validation']

    # =============== Загрузка доменного датасета ===============
    domen_data = "../../../model_ft/"
    train_dataset = load_dataset('json', data_files=f'{domen_data}/custom_gen_dataset_train.json')['train']
    val_dataset = load_dataset('json', data_files=f'{domen_data}/custom_gen_dataset_val.json')['train']
    sub_train = train_dataset.select(range(30))
    sub_val = val_dataset.select(range(10))

    # =============== Предобработка датасета ===============
    sub_train = dataset_preprocess_for_SF(
        sub_train, #Датасет
        prompt_answer_ft_SF_domen, #функция для создания промптов
        prompt_template #Шаблон промпта
    )
    sub_val = dataset_preprocess_for_SF(
        sub_val, #Датасет
        prompt_answer_ft_SF_domen, #функция для создания промптов
        prompt_template #Шаблон промпта
    )
    logger.log("data preprocess")
    # =============== Конфигурация квантования ===============
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,                     # Загрузить модель в 8-битном формате
        bnb_8bit_quant_type="int8",            # Тип 8-битного квантования (может быть nf8 или int8)
        bnb_8bit_compute_dtype=torch.bfloat16, # Тип данных для вычислений в 8-битном режиме
        bnb_8bit_use_double_quant=False        # Использовать ли двойное квантование
    )
    
    # =============== Загрузка токенайзера и модели ===============
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model']['cache_dir']
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id # Токен отступа (Особенность: свой для каждой языковой модели)
    tokenizer.padding_side = CONFIG['model']['tokenizer']['padding_size'] #Добавлять до максимальной длинны справа

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model']['cache_dir'],
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config, #Если юзать конфиг квантования - модель автоматом на GPU
        attn_implementation="eager"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    logger.log("model and tokenizer load")
    # =======================================
    # LoRA конфиг
    # =======================================

    lora_config = LoraConfig(
        r=CONFIG['lora']['r'],                            # ранг матриц адаптеров
        lora_alpha=CONFIG['lora']['lora_alpha'],          # коэффициент масштабирования LoRA
        target_modules=CONFIG['lora']['target_modules'],  # Модули для применения LoRA
        lora_dropout=CONFIG['lora']['lora_dropout'],      # Dropout для адаптеров LoRA
        bias="none",                                      # Тип применения смещения (bias)
        task_type="CAUSAL_LM",                            # Тип задачи
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    logger.log("lora config accept")
    # =================== Test for trainer and eval ===================

    #Паттерн ответа
    answer_pattern_for_collator = CONFIG["model"]["tokenizer"]["answer_pattern"]
    response_template_ids = tokenizer.encode(answer_pattern_for_collator, add_special_tokens=False)

    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    logger.log(f"RESPONSE pattern: {answer_pattern_for_collator}")

    #Функция для вычисления метрик
    func_em = build_compute_metrics_fn(
        tokenizer,
        answer_pattern_for_collator,
        logger
    )
    
    #Создаем trainer
    trainer = CreateTrainer_SF(
        model, 
        tokenizer, 
        sub_train, 
        sub_val,  
        CONFIG, 
        RANDOM_STATE, 
        logger=logger,
        data_collator=collator,
        compute_metrics=func_em
    )

    logger.log("trainer created")

    #Запускаем обучение
    logger.log(f"Start training for model {CONFIG['model']['model_name_log']}")
    trainer.train()
    logger.log(f"Finished training for model {CONFIG['model']['model_name_log']}")


    logger.log("exit")