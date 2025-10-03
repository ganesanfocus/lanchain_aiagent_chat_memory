import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import TavilySearchResults


load_dotenv()
# TAVILY_API_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Tool 1 simple QA Tool
qa_prompt = PromptTemplate.from_template("Answer clearly: {question}")
qa_chain  = LLMChain(llm=llm, prompt=qa_prompt)

qa_tool = Tool(
    name="Simple QA",
    func=qa_chain.run,
    description="Answer factual questions clearly"
)

#  Verbose meaning communicate with agents accepted, finished

# # Tool 2: Summarizer Tool
summary_prompt = PromptTemplate.from_template("Summarize the following text: \n\n{text}")
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
summary_tool = Tool(
    name="Summarizer",
    func=summary_chain.run,
    description="Summarizes long paragraph or text content"
) 

# # Tool 3: WebSearch Tavily Tool
search_tool = Tool(
    name="Web Search",
    func=TavilySearchResults(max_result=3).run,
    description="Search the internet for current and live information"
)

tools = [qa_tool, summary_tool, search_tool]

# Initialize Agent
agent_executor = initialize_agent(
    tools= tools,
    llm=llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True # Show reasoning process details
)

# Run user queries
queries = [
    "What is LangGraph in LangChain?",
    "Summarize this: LangChain is a framework to build LLM aps using prompts, memory, tools and agent.",
    "Latest news about OpenAI GPT-4o"
]

for query in queries:
    print("\n User Query:\n", query)
    response = agent_executor.run(query)
    print("\n Agent Response:\n", response)
