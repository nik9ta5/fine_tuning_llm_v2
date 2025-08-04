# =======================================
# Кастомный логгер
# =======================================

import logging
from datetime import datetime
from transformers import TrainerCallback


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
    def __init__(self, log_dir, prefix_filename = "train_run"):
        self._timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self._log_filename = f"{log_dir}/{prefix_filename}_{self._timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,  # Уровень логирования
            format='%(asctime)s - %(levelname)s - %(message)s',  # Формат сообщений
            handlers=[
                logging.FileHandler(self._log_filename, encoding='UTF-8'),  # Запись в файл
                logging.StreamHandler()  # Вывод в консоль (опционально)
            ],
            encoding='UTF-8'
        )

        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self._log_filename, encoding='UTF-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        for handler in list(transformers_logger.handlers):
            transformers_logger.removeHandler(handler)

        transformers_logger.addHandler(file_handler)

        self._my_logger = logging.getLogger("my_custom_logger") #Перехватить логгер transformers
        self._my_logger.setLevel(logging.INFO)


    def log(self, text_for_log : str):
        self._my_logger.info(text_for_log)