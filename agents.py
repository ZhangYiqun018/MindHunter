import os
import random
from typing import Any, Dict, List

from langchain.agents import AgentOutputParser
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder, PromptTemplate,
                               SystemMessagePromptTemplate)

from models import get_llm


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str):
        llm_output = llm_output.split('Human')[0]
        llm_output = llm_output.replace('AI: ', '')
        llm_output = llm_output.replace('AI:', '')
        llm_output = llm_output.replace('AI： ', '')
        llm_output = llm_output.replace('AI：', '')
        return llm_output


class ChatAgent:
    def __init__(self,
        temperature: float,
        template_path: str,
        verbose: bool = False,
        max_limit_tokens: int = 3000,
    ):
        self.chat_system_template = open(template_path, 'r', encoding='utf-8').read()
        chat_human_template = "{input}"
        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.chat_system_template),
                MessagesPlaceholder(variable_name='history'),
                HumanMessagePromptTemplate.from_template(chat_human_template)
            ]
        )
        
        self.memory = ConversationSummaryBufferMemory(
            llm             = get_llm(temperature=temperature),
            return_messages = True,
            max_token_limit = max_limit_tokens,
        )

        self.conversation = ConversationChain(
            prompt        = chat_prompt,
            llm           = get_llm(temperature=0.1),
            verbose       = verbose,
            memory        = self.memory,
            output_parser = CustomOutputParser(),
        )
        
    def update_prompt(self, knowledge: List=[], act: str = "", check: str = "", search: str = ""):
        if not isinstance(knowledge, List):
            raise AttributeError("the type of knowledge must be list, please check it!")
        knowledge = "\n".join(knowledge)
        try:
            template = self.chat_system_template + \
                    "\n\n## Human的潜在需求:\n" + search + \
                    "\n\n## AI需要采取的行动:\n" + act + \
                    "\n\n## 审计员提供的信息:\n" + check + \
                    "\n\n## 参考知识:\n" + knowledge + \
                    "\n\n"
                    
            self.conversation.prompt.messages[0] = SystemMessagePromptTemplate.from_template(template=template)
        except Exception as e:
            print(e)

    def __call__(self, input):
        return self.send(input)
    
    def send(self, input):
        response = self.conversation.predict(input= input, stop = ["\nHuman:", "Human:"],)
        
        if len(response) == 0:
            return "你的问题可能不符合当地的法律政策与法规，我无法提供服务。"
        return response
    
    def get_memory(self):
        return self.memory.load_memory_variables({})
    
    def get_history(self) -> str:
        """
        获取对话历史，format: str -> human: xxx \n ai: xxx \n
        """
        messages = self.memory.load_memory_variables({})['history']
        history = ""
        for idx, message in enumerate(messages):
            if idx % 2 == 0:
                history += f"Human:{message.content}\n"
            else:
                history += f"AI:{message.content}\n"
        
        return history
    
class CheckAgent:
    def __init__(
        self, 
        template_path: str,
        temperature: float = 0.3, 
        verbose: bool = False
    ):
        template = open(template_path, 'r', encoding='utf-8').read()
        
        prompt = PromptTemplate.from_template(template=template)
        
        self.verbose = verbose
        self.chain = LLMChain(
            llm = get_llm(temperature=temperature),
            prompt = prompt,
            verbose = verbose
        )
        
    def __call__(self, history: str, input: str, knowledge_queue: List = []) -> str:
        # dialog_history = history + f"Human: {input}"
        if not isinstance(knowledge_queue, List):
            raise AttributeError('the TYPE of knowledge queue must be in LIST, please check it!')
        dialog_history = input
        knowledge = []
        for item in knowledge_queue:
            if isinstance(item, str):
                knowledge.append(item)
        
        if len(knowledge) > 0: 
            knowledge = "\n".join(knowledge)
        else:
            knowledge = "目前知识库为空"
            
        response = self.chain.run(
            knowledge = knowledge,
            input = dialog_history
        )
        if self.verbose:
            print('*'*50, f'\nCHECK OUTPUT:\n{response}\n', '*'*50)
        return response


class SubconsciousAgent:
    def __init__(
        self, 
        template_path: str,
        template_fewshot_path: str,
        temperature: float = 0.3,
        verbose: bool = False,
        k: int = 3
    ) -> None:
        self.data = open(template_fewshot_path,'r',encoding='utf-8').read().split("\n\n")
        
        self.k = k
        self.verbose = verbose
        template = open(template_path, 'r', encoding='utf-8').read()
        prompt = PromptTemplate.from_template(template)
        
        self.chain = LLMChain(
            prompt  = prompt,
            llm     = get_llm(temperature=temperature),
            verbose = verbose
        )
        
    def __call__(self, history: str, input: str) -> None:
        dialog_history = history + f"Human: {input}"
        instruct = "\n\n".join(random.choices(self.data, k=self.k))
        
        try:
            response = self.chain.run(
                dialog_history = dialog_history,
                instruct = instruct
            )
        except Exception as e:
            print("隐式需求模块错误!", e)
        if self.verbose:
            print('*'*50, f'\n隐式需求识别模块:{response}\n', '*'*50)
            
        return response


class ActAgent:
    def __init__(
        self,
        template_path: str,
        template_fewshot_path: str,
        temperature: float = 0.1,
        verbose: bool = False,
        k: int = 5
    ):
        self.data = open(template_fewshot_path,'r',encoding='utf-8').read().split("\n\n")
        
        self.k = k
        self.verbose = verbose
        
        template = open(template_path, 'r', encoding='utf-8').read()
        prompt = PromptTemplate.from_template(template)
        
        self.chain = LLMChain(
            llm     = get_llm(temperature=temperature),
            prompt  = prompt,
            verbose = verbose
        ) 
        
    def __call__(self, history: str, input: str) -> str:
        dialog_history = history + f"Human:{input}"
        
        instruct = "\n\n".join(random.choices(
            self.data, k=self.k
        ))
        
        try:
            response = self.chain.run(
                dialog_history = dialog_history,
                instruct = instruct
            )
        except Exception as e:
            print("AI ACT决策错误！", e)
        if self.verbose:
            print('*'*50, f'\nAI ACT决策模块:{response}\n', '*'*50)
            
        return response
    
    
if __name__ == "__main__":
    agent = ChatAgent(temperature=0.4, template_path='./template/chat.txt', verbose=True)
    
    while True:
        Input = input('user:')
        
        response = agent(input = Input)
        
        print(f'assistant: {response}')
    
