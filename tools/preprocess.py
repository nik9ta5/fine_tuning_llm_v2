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


def tokenized_text(batch, tokenizer, answer_pattern = None, max_len_seq = None):
    """ Функция для токенизации промптов

    return: input_ids, attention_mask, labels, answers
    """
    prompts = batch['prompt'] #Промпты на естественном языке
    answers = batch['answer'] #Чисто ответы на естественном языке
    
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=max_len_seq,
        truncation=True,
        padding="max_length"
    ) #input_ids, attention_mask
    
    #Для лейблов - клонируем тензор токенизированного промпта
    labels = encodings['input_ids'].clone()
    for i in range(len(prompts)):
        #Находим индекс слова в промпте, которое считается началом ответа для языковой модели
        answer_start = prompts[i].find(answer_pattern)
        
        #Определяем начало слова ответа в токенизированном формате (длинна тензора отличается от длинны промпта как строки)
        answer_start_id = len(prompts[i])
        if answer_start != -1:
            answer_start_id = tokenizer(prompts[i][:answer_start], return_tensors="pt")['input_ids'].size(1)

        #Сколько токенов паддинга
        padding_tokens = max_len_seq - encodings['attention_mask'][i].sum().item() 

        #Что не является ответом - устанавливаем -100 (defualt в PyTorch and HF)
        labels[i, :padding_tokens + answer_start_id] = -100

    encodings['labels'] = labels #Лейблы - с ответами
    encodings['answers'] = answers #Ответы на естественном языке
    return encodings # input_ids, attention_mask, labels, answers
        

def dataset_preprocess(dataset, tokenizer, answer_pattern = None, max_len_seq = None):    
    dataset = dataset.map(
        lambda x: tokenized_text(x, tokenizer, answer_pattern = answer_pattern, max_len_seq = max_len_seq),
        batched=True,        #Использовать батчинг
        num_proc=1,         #Количество процессов
        remove_columns=dataset.column_names,  #Удаляем исходные колонки
    ) #токенизируем (input_ids, attention_mask, labels, answers)
    return dataset