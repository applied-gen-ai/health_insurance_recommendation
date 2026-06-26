from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import re

import os
load_dotenv()

prompt = hub.pull("hwchase17/react")

tools = [TavilySearchResults(max_results=5)]

mixtral = 'mixtral-8x7b-32768'
llama = 'llama-3.3-70b-versatile'

model = ChatGroq(temperature=0.2, model_name=llama)

#model = ChatAnthropic(temperature=0.4, model_name='claude-3-5-sonnet-20240620')

agent = create_react_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

def get_policy_url(q):
    text = agent_executor.invoke({"input": q})

    print("raw text", text['output'])
    pdf_url_pattern = r"https?://[^\s]+\.pdf"

    match = re.search(pdf_url_pattern, text['output'])
    if match:
        extracted_url = match.group()
        return extracted_url
    else:
        return "no match"
