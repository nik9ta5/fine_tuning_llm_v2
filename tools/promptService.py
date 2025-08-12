# =======================================
# prompts
# =======================================

#Новый промпт
def prompt_template(context, question, answer):
    #Общая инструкция для промпта (Новая инструкция)
    INSTRUCTION = 'Answer the question using context only. The answer must be an exact quote from the context and not include any additional information. If the question cannot be answered using context only, answer "No Answer"'
    return f"""### instructions:\n{INSTRUCTION}\n\n### context:\n{context}\n\n### question:\n{question}\n\n### answer:\n{answer}"""


def second_prompt_template(
    system_instruction : str, 
    context : str, 
    question : str, 
    answer : str
    ):
    prompt_v2 = f"""{system_instruction}

### Context:
{context}

### Question:
{question}

### Answer:\n{answer}"""
    return { "prompt" : prompt_v2, "answer" : answer}


def second_prompt_template_dict(
    system_instruction : str, 
    item : dict
    ):
    #Задать ответ
    answer = item['answers']['text'][0] if item['answers']['text'] else "No answer"

    prompt_v2 = f"""{system_instruction}

### Context:
{item['context']}

### Question:
{item['question']}

### Answer:
{answer}"""
    return { "prompt" : prompt_v2, "answer" : answer}