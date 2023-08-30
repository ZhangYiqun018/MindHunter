from agents import *

sub_agent = SubconsciousAgent(
    template_path         = './template/sub_template.txt',
    template_fewshot_path = './template/sub_fewshot_template.txt',
    temperature           = 0.1,
    verbose               = True
)

act_agent = ActAgent(
    template_path         = './template/act_template.txt',
    template_fewshot_path = './template/act_fewshot_template.txt',
    temperature           = 0.1,
    verbose               = True
)

chat_agent = ChatAgent(
    temperature   = 0.1,
    template_path = './template/chat.txt',
    verbose       = True,
)

def dialog_loop(input: str):
    history = chat_agent.get_history()
    sub = sub_agent(history=history, input=input)
    act = act_agent(history=history, input=input)

    chat_agent.update_prompt(act=act, search=sub)
    
    response = chat_agent.send(input = input)
    
    return response
    

if __name__ == '__main__':
    while True:
        Input = input("user:")
        output = dialog_loop(input = Input)
    
        print(f"assistant:{output}")