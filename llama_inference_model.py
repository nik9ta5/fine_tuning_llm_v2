from llama_cpp import Llama

model_path = "./merge_models/base_base_Llama-3.1-8B-Instruct_15-06-2025_06-26-46_SQuAD_Adapters_18-06-2025_18-13-07_domen_Adapters/model.gguf"

llm = Llama(
    model_path=model_path, #Путь до модели
    n_gpu_layers=-1, #Сколько слоев переносить на GPU (-1 -> все слои на GPU)
    n_ctx=2048, #Максимальная длинна контекста
    verbose=True, #Для отображения подробного вывода  
)

while True:
    inp = input(">:")
    if inp.strip().lower() == "exit":
        break
    
    output = llm.create_completion(
        inp, 
        max_tokens=64
    )

    print(output["choices"][0]["text"])
