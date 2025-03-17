"""
This file contains the code for the generator service, replacing the original GeneratorProcess worker.
It is a simple FastAPI application that can be used to generate text using a pre-trained model.
"""


# frameworks
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI()

class GenerationRequest(BaseModel):
    """
    A request for the generator service.
    """
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float

class GeneratorService:
    """
    A service for generating text using a pre-trained model.
    """
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    async def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """
        Generate text using the pre-trained model concurrently.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# TODO: get filepath from cfg in launch.py
generator_service = GeneratorService("deepseek-ai/DeepSeek-Prover-V1.5-RL")

@app.post("/generate")
async def generate(request: GenerationRequest):
    """
    API endpoint for generating text using the pre-trained model.
    """
    response = await generator_service.generate(
        request.prompt,
        request.max_tokens,
        request.temperature,
        request.top_p
    )
    return {"generated_text": response}

def run_generator_service(port=8000):
    """
    Run the generator service.
    """
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    # TODO: get port from cfg in launch.py
    run_generator_service()