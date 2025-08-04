# =======================================
# prompts
# =======================================

#Новый промпт
def prompt_template(context, question, answer):
    #Общая инструкция для промпта (Новая инструкция)
    INSTRUCTION = 'Answer the question using context only. The answer must be an exact quote from the context and not include any additional information. If the question cannot be answered using context only, answer "No Answer"'
    return f"""### instructions:\n{INSTRUCTION}\n\n### context:\n{context}\n\n### question:\n{question}\n\n### answer:\n{answer}"""