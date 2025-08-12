import torch
from transformers import BitsAndBytesConfig

def create_quant_config():
    return BitsAndBytesConfig(
        # load_in_8bit=True,                     # Загрузить модель в 8-битном формате
        # bnb_8bit_quant_type="int8",            # Тип 8-битного квантования (может быть nf8 или int8)
        # bnb_8bit_compute_dtype=torch.bfloat16, # Тип данных для вычислений в 8-битном режиме
        # bnb_8bit_use_double_quant=False,        # Использовать ли двойное квантование
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )