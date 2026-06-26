
import time
from urls import urls

import re
import json

from pymongo import MongoClient
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from fuzzy_check import fuzzy_match

from langchain_groq import ChatGroq

from typing import Literal, Dict
from typing import Literal, Dict, Optional
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from search_policy import search_with_metadata_key_value

from scrape_pdf import read_pdf, insert_text

load_dotenv()

import os
from langchain_huggingface import HuggingFaceEmbeddings
from react_agent import get_policy_url


embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )




llama = 'llama-3.3-70b-versatile'
model = ChatGroq(temperature=0, model_name=llama)

prompt_str = '''
You are a helpful AI assistant that parses given question on health insurance policy and extracts provider name and policy name.
Just give me output without any verbose or additional text.
Every word in the ouput has to be seperated by - and should be in lower case
Output format:
{{"provider" : "<provider name>",
"insurance_name": "<policy name>"}}

'''

template = ChatPromptTemplate.from_messages([
        ("system", f"{prompt_str}"),
        ("human", "{question}")
    ])

chain = (
    RunnableMap({
        "question": lambda x: x["question"]
    })
    | template 
    | model 
    | StrOutputParser()
)



prompt_str_final = '''
You are a helpful AI assistant that assists humans on health insurance policy related questions
Please answer within 100 words if possible. Just provide answer only without additional text.

Get the knowledge from below knowledge base:
'''

template_final = ChatPromptTemplate.from_messages([
        ("system", f"{prompt_str_final}\n{'{context}'}"),
        ("human", "{question}")
    ])


#query = "Do we have air ambulance facility in icici lombard max protect classic policy"


def agent_3(query):

    x = chain.invoke({"question": query})
    d = eval(x)
    print(d)

    pr = fuzzy_match(d)

    provider = pr['matched_provider']
    matched_insurance_name = pr['matched_insurance_name']

    results = search_with_metadata_key_value(query, provider, matched_insurance_name)
    print(results)
    print('-------------------------------------------------------------------------------------------------------------')
    if len(results)==0:
        que = "get latest health insurance policy pdf url of "+provider+' '+matched_insurance_name + " policy."
        print(que)
        url = get_policy_url(que)
        print(url)
        txt = read_pdf(url)
        insert_text(txt, provider, matched_insurance_name)
        results = search_with_metadata_key_value(query, provider, matched_insurance_name)
        formatted_docs = str(i.page_content for i in results)
        chain_final = (
                RunnableMap({
                    "context": lambda x: formatted_docs,  # Use retrieved context
                    "question": lambda x: x["question"]
                })
                | template_final 
                | model 
                | StrOutputParser()
            )
        final_answer = chain_final.invoke({"question":query})
        return final_answer
    else:
        formatted_docs = str(i.page_content for i in results)

        chain_final = (
                RunnableMap({
                    "context": lambda x: formatted_docs,  # Use retrieved context
                    "question": lambda x: x["question"]
                })
                | template_final 
                | model 
                | StrOutputParser()
            )
        
        final_answer = chain_final.invoke({"question":query})
        return final_answer