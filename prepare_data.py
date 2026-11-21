"""Подготовка к занятию «Большие языковые модели» (запускать на компьютере с интернетом).

Сохраняет в папку models модели, нужные ноутбуку, чтобы в аудитории он работал без интернета:
    Qwen/Qwen2.5-0.5B           – базовая модель, около 1 ГБ;
    Qwen/Qwen2.5-0.5B-Instruct  – модель-помощник, около 1 ГБ;
    cointegrated/rubert-tiny2   – кодировщик для задания повышенной сложности (поиск по документам), около 120 МБ.
Запуск:
    python prepare_data.py
Ноутбук сам найдет папку models рядом с собой.
"""
from pathlib import Path

MODELS = [('Qwen/Qwen2.5-0.5B', 'causal'), ('Qwen/Qwen2.5-0.5B-Instruct', 'causal'), ('cointegrated/rubert-tiny2', 'encoder')]

if __name__ == '__main__':
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    root = Path(__file__).parent / 'models'
    for name, kind in MODELS:
        target = root / name.split('/')[-1]
        if (target / 'config.json').exists():
            print('уже есть:', target)
            continue
        print('скачиваю', name)
        AutoTokenizer.from_pretrained(name).save_pretrained(target)
        loader = AutoModelForCausalLM if kind == 'causal' else AutoModel
        loader.from_pretrained(name, torch_dtype='auto').save_pretrained(target)   # формат как в оригинале (bfloat16)
        size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
        print(f'готово: {target} ({size / 2**20:.0f} МБ)')
