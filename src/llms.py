from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chat_models import ChatOllama

load_dotenv()
OPENAI_API_VERSION = '2023-07-01-preview'

"""
These are example functions depicting how we instantiated our LLMs.
You should implement your own functions which return a chat model.
Remember to authenticate before calling these functions.
Its recommended to store your relevant API keys in a .env file.
"""


def get_gpt4() -> BaseChatModel:
    return AzureChatOpenAI(deployment_name='your-gpt-4-deployment', temperature=0,
                           openai_api_version=OPENAI_API_VERSION)


def get_gpt35() -> BaseChatModel:
    return AzureChatOpenAI(deployment_name='your-gpt-3.5-deployment', temperature=0,
                           openai_api_version=OPENAI_API_VERSION)


def get_gemini_pro() -> BaseChatModel:
    return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, convert_system_message_to_human=True)


# Ollama Example
OLLAMA_PARAMS = {'temperature': 0, 'timeout': 120, }
NEURAL_CHAT = ChatOllama(model='neural-chat', **OLLAMA_PARAMS)
