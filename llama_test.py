from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# 1. Определите путь к вашей GGUF-модели
# Замените этот путь на фактический путь к вашей модели
model_path = "/home/kne.21@free.uni-dubna.ru/Desktop/prog/fine_tuning_v2/merge_models/base_Llama-3.1-8B-Instruct_15-06-2025_06-26-46_SQuAD_Adapters/my-llama-3.1-8b_squad2-q4_k_m.gguf"

# Optional: Callbacks для потокового вывода
# Позволяет видеть генерацию токенов в реальном времени
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# 2. Инициализируйте LlamaCpp LLM
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,  # Температура генерации (креативность). 0.0 - детерминированный, 1.0 - очень креативный
    max_tokens=2048,  # Максимальное количество токенов для генерации
    n_gpu_layers=30,  # Количество слоев модели, которые будут загружены в VRAM GPU.
                      # Установите -1, чтобы загрузить все слои, или 0, чтобы использовать только CPU.
                      # Подбирайте это значение в зависимости от вашей видеопамяти.
    n_batch=512,      # Размер батча для обработки. Больший батч может ускорить, но требует больше VRAM.
                      # Должен быть между 1 и n_ctx.
    n_ctx=4096,       # Размер контекстного окна (количество токенов, которые модель может "видеть").
                      # Важно для обработки длинных запросов и генерации длинных ответов.
    callback_manager=callback_manager, # Используем определенный менеджер колбэков
    verbose=True,     # Включить подробный вывод llama.cpp (полезно для отладки)
)


template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Always answer as helpfully as possible, while being safe. Your answers should not promote discrimination, hatred, or violence.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something incorrect. If you don't know the answer to a question, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# 4. Создайте цепочку (Chain)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 5. Сделайте запрос к модели
question = "Объясни, что такое квантовая запутанность, простыми словами."
print(f"\n--- Запрос: {question} ---")
response = llm_chain.run(question)
print(f"\n--- Ответ: ---\n{response}")

# Пример использования только LLM для простой генерации (без PromptTemplate и Chain)
# print("\n--- Пример прямого вызова LLM ---")
# direct_response = llm("Расскажи о себе.")
# print(f"\n--- Ответ: ---\n{direct_response}")