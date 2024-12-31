

#
# URL https://python.langchain.com/docs/integrations/llms/google_ai/
#
#
# pip install --upgrade --quiet  langchain-google-genai
# pip install --upgrade --quiet langchain_community
# pip install --upgrade --quiet langchain-openai

#from langchain_community.chat_models import AzureChatOpenAI (depreciated)
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.schema import LLMResult
from langchain_google_genai import GoogleGenerativeAI as GenerativeAI

class LLMSelector:
    def __init__(self):
        """
        Initializes the LLMSelector by reading environment variables to determine which LLM to use
        and sets up the respective API key and configuration.
        """
        llm_type=input("Please select LLM model [gemini|aoai]: ")
        if llm_type == "gemini":
            self.llm_provider = 'gemini'
        elif llm_type == "aoai":
            self.llm_provider = 'aoai'
        else:
            # default type
            self.llm_provider = 'aoai'


       
    def get_llm(self):
        """
        Creates and returns the appropriate LLM instance based on the selected provider.
        
        Returns:
            The configured LLM instance.
        """
        #if not self.api_key:
        #    raise ValueError("LLM_API_KEY is not set in environment variables")

        if self.llm_provider == "gemini":
            gemini_config = get_model_configuration("gemini-flash")
            return GenerativeAI(
                model="models/gemini-1.5-flash", 
                google_api_key=gemini_config['api_key'])
        elif self.llm_provider == "aoai":
            gpt_config = get_model_configuration("gpt-4o")
            return AzureChatOpenAI(
                model = gpt_config['model_name'],
                deployment_name = gpt_config['deployment_name'],
                openai_api_key = gpt_config['api_key'],
                openai_api_version = gpt_config['api_version'],
                azure_endpoint = gpt_config['api_base'],
                temperature= gpt_config['temperature']
                )


from model_configurations import get_model_configuration
from model_configurations import measure_time
from langchain_core.output_parsers import StrOutputParser
import traceback

@measure_time
def generate_hw01(llm,question="who are you"):
    
    system = """
            You are a helpful assistant will full knowlege of Taiwan's history.
            
            """

    system2 = """
            You are a helpful assistant.
            Please respond in JSON format.
            The top-level key must be 'Result', and its value must be a list of objects.
            Each object should contain two keys: 'date' (the date of the holiday) and 'name' (the name of the holiday).
            """
    try:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Answer the question based on the below description and only use English or traditional Chinese. Please respond in JSON format."),
                ("human", f"Question: {question} \n "),
            ]
        )

        #json_llm = llm.bind(response_format={"type": "json_object"})
        rag_chain = answer_prompt|llm|StrOutputParser()
        # RAG generation
        answer = rag_chain.invoke({"question": question})
        parser2= JsonOutputParser()
        tmp = parser2.parse(answer)
        print("[answer]",tmp.get("answer","not found"))
        #return answer
    except Exception as e:
        traceback_info = traceback.format_exc()
        print(traceback_info)

import json
class JsonOutputParser(StrOutputParser):
    def parse(self, text: str) -> dict:
        """
        Parses the output text containing JSON wrapped in delimiters.
        """
        try:
            # Extract JSON from text wrapped with ''' json ... '''
            if text.startswith("```json") and text.endswith("```"):
                # Strip the delimiters
                json_string = text[len("```json"): -len("```")].strip()
            else:
                # Assume the text is a plain JSON string
                json_string = text.strip()
            
            # Parse the JSON string
            parsed = json.loads(json_string)
            
            # Validate and extract the desired field, e.g., "answer"
            # You can customize field validation here
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"[here2]Failed to decode JSON: {e}")


class JsonOutputParser2(StrOutputParser):
    def parse(self, text: str) -> dict:
        """
        Parses the output JSON string into a dictionary and extracts fields.
        """
        try:
            # Parse the JSON string
            parsed = json.loads(text)
            # Validate and extract the "answer" field
            if "answer" not in parsed:
                raise ValueError("Field 'answer' not found in the output JSON.")
            return parsed["answer"]
        except json.JSONDecodeError as e:
            raise ValueError(f"[here]Failed to decode JSON: {e}")

@measure_time
def generate_hw02(llm:AzureChatOpenAI,question="who are you"):
        system = """
            You are a helpful assistant will full knowlege of Taiwan's history.
            
            """
        chat_prompt=ChatPromptTemplate.from_messages(
            [
               ("system", system),
                ("human", "Answer the question based on the below description and only use English or traditional Chinese. Please respond in JSON format."),
                ("human", f"Question: {question} \n "),
            ]
        )

        prompt = chat_prompt.format_prompt()
        response = llm.invoke(prompt)
        #print(f"[json1] {response}")
        parser = StrOutputParser()
        try:
            answer = parser.parse(response.content)
            #print("[json2]Extracted Answer ", answer) 
            parser2= JsonOutputParser()
            tmp = parser2.parse(answer)
            print("[answer]",tmp.get("answer","not found"))
        except ValueError as e:
            print(f"Error parsing LLM response: {e}")

@measure_time
def generate_hw03(llm,question="who are you"):
    result: LLMResult = llm.generate(prompts=[question])
    #result: LLMResult = llm.generate([prompt]) (AI is old)
    print("[answer]",result.generations[0][0].text)
#
# Main
#
def main():
    """
    Example of how to use LLMSelector to switch between different LLMs.
    """
    llm_selector = LLMSelector()
    llm = llm_selector.get_llm()

    # Example prompt template and query
    if isinstance(llm, AzureChatOpenAI):
        gpt_config = get_model_configuration("gpt-4o")
        llm = AzureChatOpenAI(
            model = gpt_config['model_name'],
            deployment_name = gpt_config['deployment_name'],
            openai_api_key = gpt_config['api_key'],
            openai_api_version = gpt_config['api_version'],
            azure_endpoint = gpt_config['api_base'],
            temperature= gpt_config['temperature']
            )

        #question = "台灣歷史有多久?"
        #question = "台灣是個獨立國家嗎?"
        #question = "How long is Taiwan's history?"
        question=input("What is your question? ")
        generate_hw01(llm, question)
        #generate_hw02(llm, question)
        
    elif isinstance(llm, GenerativeAI):
        query=input('what is your questions? ')
        generate_hw03(llm,query)

if __name__ == "__main__":
    main()
