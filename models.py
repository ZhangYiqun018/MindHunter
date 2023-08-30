from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
import configparser


config_name = "config_private.ini"
config = configparser.ConfigParser()
config.read(config_name)

def get_llm(
    temperature: float,
    max_tokens: int = 1000,  # single turn max tokens  
    n: int = 1,
    streaming: bool = False,
) -> AzureChatOpenAI:
    llm: AzureChatOpenAI= AzureChatOpenAI(
        deployment_name    = config.get('AZURE', 'deployment_name'),
        openai_api_base    = config.get('AZURE', 'api_base'),
        openai_api_version = config.get('AZURE', 'api_version'),
        openai_api_key     = config.get('AZURE', 'api_key'),
        temperature        = temperature,
        max_tokens         = max_tokens,
        n                  = n,
        streaming          = streaming,
    )
    return llm


if __name__ == '__main__':
    llm = get_llm(temperature=0.1, max_tokens=3000, streaming=True)
    
    print(llm)
    
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template("{input}")
    chain = LLMChain(
        llm = llm,
        verbose=True,
        prompt=prompt,
    )
    
    print(chain.run('为什么程序猿都喜欢hello world.'))
    

    
