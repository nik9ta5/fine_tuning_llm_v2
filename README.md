# Репозиторий для тонкой настройки языковых моделей


## Файлы: 
`run_train.py` - основной скрипт для тонкой настройки
`eval_script_model.py` - основной скрипт для оценки настроенной модели


`./tools/` - вспомогательные компоненты для FT
* `dataset_utils.py` - загрузка датасета
* `eval.py` - функции для оценки модели
* `logService.py` - логирование
* `lora_utils.py` - загрузка LoRA конфигурации
* `promptService.py` - формирование промптов
* `quant_utils.py` - загрузка конфигурации квантования
* `tuning.py` - загрузка SFTTrainer


## Запуск
**Версия Python, использованная при разработке:** `3.10.11`

* Клонировать репозиторий
```bash
git clone https://github.com/nik9ta5/fine_tuning_llm_v2.git
cd ./fine_tuning_llm_v2
```
* Создать виртуальное окружение и активировать его
```bash
python -m venv venv
```
#### Windows:
```cmd
venv\Scripts\activate
```
#### Linux:
```bash
source venv/bin/activate
```
* Установить зависимости из файла `requirements.txt`
```bash
pip install -r requirements.txt
```
* Отредактировать файл конфигурации: `./config/AppConfig.yaml`
* Отредактировать инструкцию `SYSTEM_INSTRICTION` для промпта в `run_train.py`
* Запустить ft через `run_train.py`
```bash
python run_train.py
```
* Запустить оценку через `eval_script_model.py`
```bash
python eval_script_model.py
```