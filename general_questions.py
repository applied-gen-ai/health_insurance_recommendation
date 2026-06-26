import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


load_dotenv()


def read_data():
    with open('insurance.txt', 'r') as file:
        data = file.read()  # Reads the entire file into a string
        return data




mixtral = 'mixtral-8x7b-32768'
llama = 'llama-3.3-70b-versatile'

model = ChatAnthropic(temperature=0, model_name='claude-3-5-sonnet-20240620') 

prompt_str = "you are helpful ai assistant that assits humans on health insurance in india based on below reference context. Very carefully analyse the context and give output as you are acting as SME. Give output like you are suggesting on health insurance. Based on your expertise other agents will extract relavant data."

template = ChatPromptTemplate.from_messages([
        ("system", f"{prompt_str}\n{'{context}'}  .please look heading before giving advises to the user and give advises along with headings. PLEASE answer as per question. Take reference from context and just answer for what is asked without any verbose."),
        ("human", "{question}")
    ])

gen_txt = read_data()

chain = (
        RunnableMap({
            "context": lambda x: gen_txt,  # Use retrieved context
            "question": lambda x: x["question"]
        })
        | template 
        | model 
        | StrOutputParser()
    )

def agent_1(user_input):
        response = chain.invoke({"question": user_input})  #{"user_input": "Who is sourav ganguly?"}
        return response