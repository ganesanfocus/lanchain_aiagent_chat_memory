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

# Tool 2: Summarizer Tool
summary_prompt = PromptTemplate.from_template("Summarize the following text: \n\n{text}")
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
summary_tool = Tool(
    name="Summarizer",
    func=summary_chain.run,
    description="Summarizes input text"
) 

# Tool 3: WebSearch Tavily Tool
search_tool = Tool(
    name="Web Search",
    func=TavilySearchResults(max_result=3).run,
    description="Search the internet for current information"
)


# Tool usage examples
qa_query="What is LangGraph in LangChain?"
summary_text = """
LangGraph is a library built on top of Langchain that helps developers create stateful, multi-stp agents
as graphs. Each node represents a step like calling an LLM or a tool. It's ideal for advanced AI workflows.
"""
search_query = "Latest updates on GPT-4o by OpenAI"


# Run each tools manually
print("\n Simple QA Tool Output:\n", qa_tool.run({"question": qa_query}))
print("\n Summarizer Tool Output:\n", summary_tool.run({"text": summary_text}))
print("\n Web Search Tool Output:\n", search_tool.run({"text": search_query}))