from peft import LoraConfig


def create_lora_config():
    return LoraConfig(
        r=16,                                   # ранг матриц адаптеров
        lora_alpha=32,                          # коэффициент масштабирования LoRA
        target_modules=["q_proj", "o_proj"],    # Модули для применения LoRA
        lora_dropout=0.05,                      # Dropout для адаптеров LoRA
        bias="none",                            # Тип применения смещения (bias)
        task_type="CAUSAL_LM",                  # Тип задачи
    )