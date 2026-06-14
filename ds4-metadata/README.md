---
language: en
pipeline_tag: text-generation
tags:
- mlx
library_name: mlx
---

# mlx-community/DeepSeek-V4-Flash-8bit

Made possible by [Lambda.ai](https://huggingface.co/lambda) ❤️

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/DeepSeek-V4-Flash-8bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
