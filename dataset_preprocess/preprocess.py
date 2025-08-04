# =======================================
# Для предобработки датасета
# =======================================

#Функция, если юзать SFTrainer
def prompt_answer_ft_SF_squad(batch, prompt_template):
    """Функция для формирования промптов
    Для SQuAD 2.0
    """
    answers = [
        item['text'][0] if item['text'] else "No Answer"
        for item in batch['answers']
    ]
    prompts = [
        prompt_template(context, question, answer) 
        for context, question, answer in zip(batch['context'], batch['question'], answers)
    ]
    return {
        "text": prompts
    }


def prompt_answer_ft_SF_domen(batch, prompt_template):
    """Функция для формирования промптов
    Для доменного датасета
    """
    answers = [
        item if item else "No Answer"
        for item in batch['answers']
    ]
    prompts = [
        prompt_template(context, question, answer) 
        for context, question, answer in zip(batch['context'], batch['question'], answers)
    ]
    return {
        "text": prompts
    }


def dataset_preprocess_for_SF(dataset, func_for_preprocess, prompt_template):
    dataset = dataset.map(
        lambda x: func_for_preprocess(x, prompt_template), #Функция, которая применяется ко всем строкам
        batched=True,        #Использовать батчинг
        num_proc=1,         #Количество процессов
        remove_columns=dataset.column_names,  #Удаляем исходные колонки
    )
    return dataset