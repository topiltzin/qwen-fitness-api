from fastapi import FastAPI
from pydantic import BaseModel
import torch
from peft import PeftModel
import time 
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER_ID = "Topiltzinfl/qwen3-fitness-tainer-0.6b"
#MODEL_ID = "Qwen/Qwen2.5-3B"
#ADAPTER_ID = "Topiltzinfl/qwen3-fitness-0.6b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Qwen Fitness API")

print(f"Loading model: {MODEL_ID}")
print(f"Device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_ID,
)
model = model.merge_and_unload()

model.to(DEVICE)
model.eval()

print("Model loaded successfully")
SYSTEM_PROMPT = """
Eres un asistente virtual de entrenamiento físico que responde principalmente en español.

Tu objetivo es ayudar al usuario con:
- entrenamiento de fuerza
- hipertrofia
- pérdida de grasa
- cardio
- movilidad
- recuperación
- hábitos relacionados con el entrenamiento

Responde de forma:
- clara
- práctica
- motivadora pero no exagerada
- profesional
- fácil de entender

Cuando propongas un entrenamiento, explica:
- ejercicios
- series
- repeticiones
- descansos
- progresión

Cuando falten datos importantes para personalizar una recomendación,
pregunta al usuario por ellos.

Prioriza:
- buena técnica
- progresión gradual
- recuperación
- seguridad

Si el usuario describe dolor, lesión o síntomas importantes,
no intentes diagnosticarlo.
Recomienda consultar a un profesional de salud cuando corresponda
y evita recomendar entrenar a través del dolor.

No prometas resultados garantizados.
"""    

class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "message": "Qwen Fitness API",
        "model": MODEL_ID,
        "device": DEVICE,
    }

#Tu objetivo es ayudar al usuario con entrenamiento de fuerza, hipertrofia,
#pérdida de grasa, cardio, recuperación, movilidad y hábitos relacionados
#con el entrenamiento.

@app.post("/chat")
def chat(request: ChatRequest):
    start = time.time()
    messages = [
        {
            "role": "system",
            "content": """
Eres un entrenador virtual de fitness que responde principalmente en español.

Responde de forma:
- clara
- práctica
- profesional

Ayuda con:
- hipertrofia
- fuerza
- pérdida de grasa
- cardio
- movilidad
- recuperación

Cuando propongas un entrenamiento incluye ejercicios,
series, repeticiones y descansos.

Si faltan datos importantes, pregunta al usuario.

Si el usuario describe dolor o una lesión, no diagnostiques.
Recomienda consultar a un profesional cuando corresponda.

No muestres razonamientos internos.
Responde directamente con la respuesta final.
""",
        },
        {
            "role": "user",
            "content": request.message,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }
    preprocessing_time = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=124,
            do_sample=False,
            use_cache=True,
#            temperature=0.7,
#            top_p=0.9,
        )
    generation_time = time.time()

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[-1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
    end = time.time()
    print(
        f"Preprocessing: {preprocessing_time - start:.2f}s | "
        f"Generation: {generation_time - preprocessing_time:.2f}s | "
        f"Total: {end - start:.2f}s"
    )
    return {
        "model": MODEL_ID,
        "adapter_model": ADAPTER_ID,
        "response": answer.strip(),
        "total_seconds": round(
            end - start,
            2
        ),        
    }
