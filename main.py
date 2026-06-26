from compare_agent2 import agent_2
from general_questions import agent_1
from parse_question import agent_3

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool, create_tool_calling_agent, create_self_ask_with_search_agent
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

from risk import agent_4


model = ChatAnthropic(temperature=0.4, model_name='claude-3-5-sonnet-20240620') 


tools_data = [
    
        Tool(
            name="compare tool",
            func=agent_2,
            description='''
            This tool helps to answer questions related to  co-payment, Room-rent-limit, Disease-sub-limit, Pre-Post-hospitalization, Low-Waiting-period,
            Daycare-Treatment-cover, Restoration-Benefit, maternity-coverage, no-claim-bonus. This tool also helps in COMPARING multiple health insurance policies based on the mentioned factors.

            ''',
        ),
        
        Tool(
            name="other questions tool",
            func=agent_3,
            description='''
            This tool helps to answer any other question EXCEPT co-payment, Room-rent-limit, Disease-sub-limit, Pre-Post-hospitalization, Low-Waiting-period,
            Daycare-Treatment-cover, Restoration-Benefit, maternity-coverage, no-claim-bonus from any other insurance policy.
            '''
        ),

                Tool(
            name="risk tool",
            func=agent_4,
            description='''
            This tool helps to asses risk of an insurance provider based on ICR and CSR ratios
            ''',
        )
        
]


prompt_template = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
        template="""
    You are an intelligent task dispatcher. Your job is to analyze the given input and decide which of 3 specialized tools should handle the task.
    Once you get all the required data, please organize it in organised way to the user.

    If the question is ONLY related to risk tool then only call risk tool. no need to call other tool.
    If the question is ONLY related to COMPARE tool then only call COMPARE tool. no need to call other tools. Query sent to factor agent goes as input to text to sql agent, so please frame query accordingly.
    If the question is only related to other questions tool then only call other questions tool. no need to call other tool


    ** If question is related to comparing multiple policies and recommending then please call risk tool after compare tool. Regarding ICR , good ICR is between 60-70%. If ICR is more than 80%, it means policy provider is not financially healthy.

    ** For a given question, if you need to answer that question by calling multiple tools, please call different tools one after the other.
    If certain information in user question cannot be provided by any of the tools, you can ignore that subquestion and mention same to user.

    *** Vey carefully analyse the answers provided by all the tools and give a reasoning on what made you take the final decision. This is very important
    
    ** If a question is pointed to certain policy on what is covered, etc then use other questions agent.

    While answer be very crisp in output answers and properly provide reasoning of your decision.


    Invoke the agents that are related to question. 
    Input: {input}

    Available tools: {tool_names}
    {tools}

    To make your decision:
    1. Carefully analyze the user question.
    2. Based on what is asked by user, properly call right tools with very right prompts
    3. Collate the answers and crisply share it with end user
    4. DONT give long answers


    Use the following format:

    Thought: Consider the task and which tool would be most appropriate
    Action: The name of the tool to use
    Action Input: The exact input to pass to the tool
    Observation: The result of the action
    Thought: I now know the final answer
    Final Answer: The final answer to the original input question

    {agent_scratchpad}

    """
    )

agent = create_react_agent(model, tools_data, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools_data, verbose=True, handle_parsing_errors=True)

result = agent_executor.invoke({"input": '''
    Can you please compare ICICI lombard max protect classic policy with hdfc ergo optima restore policy and suggest which one is better?
    '''
    })

print(result)
# Can you please compare ICICI lombard max protect classic policy with hdfc ergo optima restore policy and suggest which one is better?

# I dont know how to choose a health insurance, can you please recommend top 3 health insurance policies to buy?

# What kind of kidney diseases are covered in ICICI lombard max protect classic policy