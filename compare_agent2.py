
import time
from urls import urls

import re
import json

from pymongo import MongoClient
import os
import io
import sys
from contextlib import redirect_stdout
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq

from typing import Literal, Dict
from typing import Literal, Dict, Optional
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()



client = MongoClient("mongodb://localhost:27017/")
db = client["insurance_db"]
collection = db["health_insurance"]

mixtral = 'mixtral-8x7b-32768'
llama = 'llama3-70b-8192'

#model = ChatGroq(temperature=0, model_name=llama) 

model = ChatAnthropic(temperature=0.4, model_name='claude-3-5-sonnet-20240620') 

template = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert pymongo code generator that generates code for given question by the user. DONT again instatiate client and collection in the output. I DONT NEED ANY VERBOSE IN THE OUTPUT. JUST NEED CODE
Below is the schema of mongo db table where provider is name of health insurance company and insurance_name is name of health insurance policy.
In below schema there are 10 features from copayment to no claim bonus. For a insurance policy below schema tells whether the feature in this policy is good/bad/neutral/NA.
Like wise you have lot of data related to lot of providers. policy name will be in lower case seperated by -
    
MONGO DB details:
client = MongoClient("mongodb://localhost:27017/")
db = client["insurance_db"]
collection = db["health_insurance"]
     
     
SCHEMA:
{{'insurance_name': 'united-india-individual-gold-plan',
  'provider': 'united-india',
  'description': 'United India Individual Gold Plan is a comprehensive health insurance policy that covers various medical expenses, including hospitalization, day care treatments, and alternative medicine.',
  'features': {{
      'co-payment': {{'rating': 'good', 'details': 'No co-payment, the insurer will bear the entire cost of treatment up to the sum insured.'}},
      'Room-rent-limit': {{'rating': 'bad', 'details': 'Room rent limit of 1% of the sum insured for normal rooms and 2% for ICU.'}},
      'Disease-sub-limit': {{'rating': 'bad', 'details': 'Disease-wise sub-limits for certain diseases like Cataracts, Hernia, Hysterectomy, etc.'}},
      'Pre-Post-hospitalization': {{'rating': 'good', 'details': 'Pre and post hospitalization expenses covered up to 10% of sum insured for 30 days before and 60 days after discharge.'}},
      'critical-illness': {{'rating': 'NA', 'details': 'NA'}},
      'Low-Waiting-period': {{'rating': 'neutral', 'details': 'Reasonable waiting period of 3 years for pre-existing diseases.'}},
      'Daycare-Treatment-cover': {{'rating': 'good', 'details': 'Day care treatments covered, including minor procedures like dialysis, chemotherapy, and minor surgeries.'}},
      'Restoration-Benefit': {{'rating': 'bad', 'details': 'No restoration benefit, the policy does not restore the cover after a claim is made.'}},
      'maternity-coverage': {{'rating': 'NA', 'details': 'NA'}},
      'no-claim-bonus': {{'rating': 'bad', 'details': 'No bonus for being healthy and not claiming insurance.'}}
  }}
}}


"""),
    ("human", '''
     RULES FOR CODE GENERATION:
     1) Below is the list of insurance providers as stored in mongo db. Given any question related to any provider, PLEASE MAP to below names. This is MANDATORY.
     acko, aditya-birla, bajaj-allianz, bharti-axa, care, hdfc-ergo, icici-lombard, iffco-tokio, manipal-cigna, new-india-assurance, niva-bupa-erstwhile-max-bupa, oriental-insurance, reliance-general, royal-sundaram, sbi, star-health, tata-aig, united-india
     2) whenever a question is asked about a certain policy under a provider, remember policy name will be in lower case where multiple words are seperated by -.
     3) Always dont do a direct match of policy name rather check for partial match ex- LIKE in SQL. Always include insurance name in the output. I need ALL the matching documents.
     4) I will be using exec command to execute your output, so please print results in your code.
     5) When filtering fields in MongoDB, DO NOT use Python dictionary comprehensions directly in the projection parameter. MongoDB projections cannot process Python expressions like "{{k: v for k, v in "$features".items()}}".
     6) To filter subdocuments or embedded objects (like features with specific ratings), either:
            - Use the appropriate MongoDB operators in the query
            - OR retrieve the complete field and filter it in Python after getting the results
     7) I need ONLY pymongo CODE as output WITHOUT any verbose and WITHOUT any other explanation and i DONT even need, here is your code start text.
     8) *** If any question is related to recommend top N policies, you need to retrieve data related to ALL the policies based on the parameters you know.**

NOTE**:
    - If question is related to comparing policies, JUST filter what is good for each policy in terms of co-payment, Room-rent-limit, Disease-sub-limit, Pre-Post-hospitalization, Low-Waiting-period,
    Daycare-Treatment-cover, Restoration-Benefit, no-claim-bonus
     - If user explicitly asks for maternity then add it to query
     - While filtering for good/bad, please look at schema.
     - *** If any question is related to recommend top N policies, you need to retrieve data related to ALL the providers and policies based on the parameters you know.**

{question}
     ''')
])

# Fix the RunnableMap implementation
chain = (
    RunnableMap({
        "question": lambda x: x["question"]
    })
    | template 
    | model 
    | StrOutputParser()
)

def agent_2(q):
    response = chain.invoke({"question": q}).replace('```', '')
    f = io.StringIO()
    with redirect_stdout(f):
        exec(response)
    output = f.getvalue()  # Contains "Hello, world!\n15\n"
    return output