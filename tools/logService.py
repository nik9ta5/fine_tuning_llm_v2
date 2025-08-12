# =======================================
# Кастомный логгер
# =======================================
import sys
import io

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
    def __init__(
        self, 
        path2log_dir, 
        log_file_name, 
        logging_lavel,
        outputConsole = False
        ):
        self.path2log_dir = path2log_dir
        self.log_file_name = log_file_name
        
        full_path2file_log = f"{path2log_dir}/{log_file_name}"

        handlers = [logging.FileHandler(full_path2file_log, encoding='utf-8')]
        
        if outputConsole:
            # Оборачиваем stdout/stderr в UTF-8, чтобы избежать UnicodeEncodeError
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            handlers.append(logging.StreamHandler())  # Добавляем только если нужно

        logging.basicConfig(
            level=logging_lavel,  # Уровень логирования
            format='%(asctime)s - %(levelname)s\n%(message)s',  # Формат сообщений
            handlers=handlers,
            encoding='utf-8'
        )

        self._my_logger = logging.getLogger()

    def getLogger(self):
        return self._my_logger
    
    def log(self, message):
        self._my_logger.info(message)