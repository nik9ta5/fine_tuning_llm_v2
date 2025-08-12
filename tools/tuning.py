# =======================================
# Функции для тонкой настройки с SFTrainer
# =======================================

import os
from datetime import datetime
from trl import SFTTrainer, SFTConfig 
from tools.logService import CustomLoggingCallback


def CreateTrainArguments_SF(CONFIG: dict, random_state : int):
    """Функция для создания объекта, агрегирующего в себе параметры для обучения"""
    
    TIMESTAMP = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    path2save = f"{CONFIG['train']['model_save_dir']}/train__{TIMESTAMP}__{CONFIG['model']['model_name_log']}"
    os.makedirs(path2save, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=path2save, #Путь до директории для сохранения
        num_train_epochs=CONFIG['train']['epochs'], #Кол-во эпох
        per_device_train_batch_size=CONFIG['train']['train_batch'], #Размер батча для обучения
        per_device_eval_batch_size=CONFIG['train']['val_batch'], #Размер батча для оценки
        gradient_accumulation_steps=CONFIG['train']['grad_accum'], #Накопление градиентов
        eval_strategy="steps", #Стратегия оценки
        save_strategy="steps", #Стратегия сохранения
        logging_strategy="steps", #Стратегия логирования
        eval_steps=CONFIG['train']['eval_step'], #Кол-во шагов до оценки
        save_steps=CONFIG['train']['eval_step'], #Кол-во шагов до сохранения
        logging_steps=CONFIG['train']['log_step'], #Кол-во шагов для логирования
        learning_rate=CONFIG['train']['lr'], #Скорость обучения
        weight_decay=CONFIG['train']['weight_decay'], #Для регуляризации 
        warmup_steps=0, #Начальный прогрев
        logging_first_step=True,
        fp16=False, #Используем bf16, если GPU поддерживает
        bf16=True,  #Какой тип данных используем
        dataloader_num_workers=0, #Кол-во потоков для загрузки
        load_best_model_at_end=True, #Сохранять лучшую модель в конце
        metric_for_best_model="eval_loss", #Метрика для определения лучшей модели
        greater_is_better=False, 
        report_to="none", #Использование сторонних сервисов логирования
        # gradient_checkpointing=True, #Нужно отключить, если обучать на нескольких GPU
        optim="paged_adamw_8bit", #Какой оптимизатор использовать
        max_grad_norm=0.3,
        seed=random_state, #Сид
        save_total_limit=None, #Ограничение на кол-во сохранений

        max_seq_length=CONFIG["model"]["max_length"], #Максимальная длинна последовательности
        packing=False, #использовать упаковку
    ) 
    return sft_config


def CreateTrainer_SF(model, tokenizer, train_dataset, val_dataset, CONFIG: dict, random_state : int, logger = None, data_collator=None, compute_metrics=None):
    """Функция для создания объекта траинера, для обучения модели
    (Использую optimizer от HF)
    logger - используется кастомный объект, обертка обычного ничего больше
    """
    training_args = CreateTrainArguments_SF(CONFIG, random_state)    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # processing_class=tokenizer,
        dataset_text_field="prompt", #В какой колонке находится текст
        callbacks=[CustomLoggingCallback(logger)],
        data_collator=data_collator,
        compute_metrics=compute_metrics, #РЕАЛИЗОВАТЬ ФУНКЦИЮ
    )
    return trainer