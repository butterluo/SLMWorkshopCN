import os
import logging
import json
import torch
from transformers import pipeline
from unsloth import FastLanguageModel

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global tokenizer
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "./outputs"
    )
    model_id = "microsoft/phi-4"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id, # Load up `Phi-4 14B`, and set parameters
        # max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
    )
    FastLanguageModel.for_inference(model) 

    model.load_adapter(model_path)
    logging.info("Loaded model.")
    
def run(json_data: str):
    logging.info("Request received")
    data = json.loads(json_data)
    input_data= data["input_data"]
    params = data['params']
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = pipe(input_data, **params)
    generated_text = output[0]['generated_text']
    logging.info("Output Response: " + generated_text)
    json_result = {"result": str(generated_text)}
    
    return json_result