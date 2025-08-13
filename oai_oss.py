from transformers import pipeline

pipe = pipeline(
    "text-generation",
    # model="openai/gpt-oss-120b",
    model="openai/gpt-oss-20b", # 20b
    torch_dtype="auto",
    device_map="auto",
)
out = pipe(
    [{"role":"user","content":"Please explain the concept of quantum mechanics in a concise manner."}],
    max_new_tokens=256,
)
print(out[0]["generated_text"])