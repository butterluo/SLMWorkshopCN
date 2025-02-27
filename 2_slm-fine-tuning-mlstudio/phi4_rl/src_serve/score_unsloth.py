import os
# os.environ['HF_HOME']="/mnt/batch/tasks/shared/LS_root/mounts/clusters/esus1h100a/code/Cache/HF/"
# print(os.path.normpath(os.path.join(
#         __file__, "..", "..", "..", "./model"
#     )))
# if True:
#     exit(1)
import logging
import json
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
    # model_path = os.path.join(
    #     os.getenv("AZUREML_MODEL_DIR"), "./model"
    # )
    model_path = os.path.normpath(os.path.join(
        __file__, "..", "..", "..", "./model"
    ))

    model_id = "microsoft/phi-4"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0}, torch_dtype="auto", trust_remote_code=True)

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

if __name__ == "__main__":
    init()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    input_data = "你好呀！"
    output = pipe(input_data)
    generated_text = output[0]['generated_text']
    logging.error("Output Response:****** " + generated_text)