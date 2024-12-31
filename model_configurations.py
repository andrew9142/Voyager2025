import os
import time
from dotenv import load_dotenv
load_dotenv()

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the wrapped function
        end_time = time.time()  # Record end time
        print(f"Execution time of {func.__name__}: {end_time - start_time:.6f} seconds")
        return result
    return wrapper

configurations = {
    "gpt-4o": {
        "model_name": "gpt-4o",
        "api_base": os.getenv('AZURE_OPENAI_GPT4O_ENDPOINT'),
        "api_key": os.getenv('AZURE_OPENAI_GPT4O_KEY'),
        "deployment_name": os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT_CHAT'),
        "api_version": os.getenv('AZURE_OPENAI_GPT4O_VERSION'),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_token": 4096
    },
    "gemini-flash" : {
        "api_key": os.getenv('GEMINI_FLASH_KEY')
    }
}

def get_model_configuration(model_version):
    return configurations.get(model_version)

