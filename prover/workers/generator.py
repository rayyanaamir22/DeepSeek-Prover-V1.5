import os
import time
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from prover.utils import AttrDict, MODEL_FORMAT

class GeneratorProcess(mp.Process):
    def __init__(self, local_rank, node_rank, model_path, task_queue, request_statuses, lock, args):
        super().__init__()
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.model_path = model_path
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.args = args
        self.prompt_func = MODEL_FORMAT[args.mode]['prompt']
        self.output_func = MODEL_FORMAT[args.mode]['output']
        
    def run(self):
        seed = int(time.time()) % 1000 + (self.node_rank * 8 + self.local_rank) * 1000
        os.environ['LOCAL_RANK'] = str(self.local_rank)
        
        # set random seed for reproducibility
        torch.manual_seed(seed)
        
        # load the model and tokenizer
        device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # configure model loading based on device
        if device.startswith("cuda"):
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use half precision for GPU
                trust_remote_code=True,
                device_map=device
            )
        elif device == "mps":
            # Load model with MPS acceleration for macOS
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # MPS works better with half precision
                trust_remote_code=True
            ).to(device)
        else:
            # CPU fallback
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Use full precision for CPU
                trust_remote_code=True
            ).to(device)
        
        # Configure generation parameters
        generation_config = model.generation_config
        generation_config.max_new_tokens = self.args.max_tokens
        generation_config.temperature = self.args.temperature
        generation_config.top_p = self.args.top_p
        generation_config.do_sample = self.args.temperature > 0
        
        while True:
            inputs = self.task_queue.get()
            if inputs is None:  # Terminate when receiving None
                break
            
            # Process batch
            outputs = []
            
            for _, request_id, item in inputs:
                # Construct prompt
                prompt = ''.join([
                    item.get('_extra_header', str()),
                    self.prompt_func(item),
                    item.get('_extra_prompt', str()),
                ])
                
                # Tokenize input
                input_ids = tokenizer(prompt, return_tensors='pt').to(device)
                
                # Generate text
                with torch.no_grad():
                    output_ids = model.generate(
                        **input_ids,
                        max_new_tokens=self.args.max_tokens,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        do_sample=self.args.temperature > 0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode output
                output_text = tokenizer.decode(output_ids[0][input_ids['input_ids'].shape[1]:], skip_special_tokens=True)
                processed_output = self.output_func(output_text)
                outputs.append(processed_output)
            
            # Update request statuses
            with self.lock:
                for (_, request_id, _), output in zip(inputs, outputs):
                    self.request_statuses[request_id] = output