# %%
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import dotenv_values

config = dotenv_values(".env") 


# %%

# %%
import requests  # used to make HTTP requests
from langchain.tools import StructuredTool  # this helps define a tool that performs a specific task
import pandas as pd  # used to handle tabular data like dataframes
from jobspy import scrape_jobs  # A function from the jobspy library that scrapes job listings

def find_jobs_lookup(search_term: str, location: str = "India", results_wanted: int = 20) -> str:  # Returns a formatted string of job listings

    # Scrape job listings using the jobspy library
    # Calls scrape_jobs to get job listings from different platforms
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
        search_term=search_term,
        google_search_term=f"{search_term} jobs near {location} since yesterday",  # A search string tailored for Google Jobs
        location=location,
        results_wanted=results_wanted,
        hours_old=72,
        country_indeed='India',  # Limits Indeed search results to jobs in India
    )

    # Filter the necessary columns
    filtered_jobs = jobs[['site', 'job_url', 'title', 'company', 'location', 'description']]

    # Only include jobs with descriptions (non-empty descriptions)
    filtered_jobs = filtered_jobs[filtered_jobs['description'].notna()]

    # Sample 10 random jobs
    filtered_jobs = filtered_jobs.sample(n=10)

    # Convert the DataFrame to a string with only the first 20 words of the description
    job_listings_str = ""
    for _, job in filtered_jobs.iterrows():
        # Limit description to first 20 words
        description_words = job['description'].split()[:20]
        limited_description = " ".join(description_words)  # Joins the first 20 words (from the list description_words) back into a single string, with each word separated by a space.

        job_listings_str += f"Title: {job['title']}\n"
        job_listings_str += f"Company: {job['company']}\n"
        job_listings_str += f"Location: {job['location']}\n"
        job_listings_str += f"Description: {limited_description}...\n"
        job_listings_str += f"Link: {job['job_url']}\n\n"

    return job_listings_str.strip()  # Remove any trailing whitespace

# A tool in LangChain is something that the language model can "call" to perform tasks, such as fetching data from a database, scraping the web, or processing information

find_jobs_tool = StructuredTool.from_function(  # Converts the find_jobs_lookup function into a LangChain tool
    func=find_jobs_lookup,  # The function to run (i.e., find_jobs_lookup)
    name="tool_find_jobs_information",  # Explains what the tool does
    description="""A tool designed to gather job listings""",  # Automatically handles any errors during execution
    handle_tool_error=True
)



# %%
tools = [find_jobs_tool] # putting find_jobs_tool into a list

# %%
from langchain_community.tools import DuckDuckGoSearchResults

#DuckDuckGoSearchResults is a pre-built LangChain tool that allows you to perform web searches using the DuckDuckGo search engine

search = DuckDuckGoSearchResults() #This creates an instance (or object) of the DuckDuckGoSearchResults class



# %%
web_tool = [search] # list of tool

# %%
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessage
from langgraph.graph.message import add_messages # A function used to add messages to a list

class State(TypedDict):
  messages:Annotated[list,add_messages]

# %%
from langgraph.graph import StateGraph,START,END # importing three components from the langgraph library that are related to building and managing a graph of states

# %%
from langchain_groq import ChatGroq # it's a class provided by langchain_groq and used as an interface with language model (rapper class)

# %%

groq_api_key=config['GROQ_API_KEY_PLACEMENT']

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192") # Creating a Groq Language Model (LLM) Instance

# Refers to a version of the LLaMA model . The "70b" indicates the model has 70 billion parameters.

# "8192" refers to the model's token limit

llm

# %%
llm_with_tools=llm.bind_tools(tools=tools) # This method is used to bind or attach additional tools to the language model
llm_with_web = llm.bind_tools(tools =web_tool)

# %%
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# %%
def role_extractor (state:State):
  
  prompt=f"""
  and following are intrests of user  :
  {state["messages"]}

  just recommend a job role answer should be a single job role no explanation

  your aim is to just provide a role only role , you dont need to search just return a role
  """
  return {"messages":[llm.invoke(prompt)]}

# %%
from langgraph.prebuilt import ToolNode,tools_condition

# %%
def job_finder (state:State):
  job_role = state["messages"][-1].content
  job_listing = find_jobs_tool.run(job_role)
  prompt=f"""
  folowing are details about user :-
  {read_file('report.txt')}
  and following are job listings :
  {job_listing}

  give the following report contining list of job in following format :
  ___________________________________________________________________________________________________________________________________________________
  1. job role
  2. company name
  3. description
  4. link


  NOTE : do not use any tools just answer on the basis of this prompt

  '
  """
  return {"messages":[llm.invoke(prompt)]}

# %%
def job_recomender (state:State):
  job_listing = state["messages"][-1].content

  prompt=f"""
  for the following job listings :
    {job_listing}
  find courses needed for above job listing on the internet using web tool ( the web tool returns title , snippet and link)
  display in following format NOTE ONLY ONE SEARCH IS ENOUGH !!! (search only one time )

  SECTION 2 :
  ___________________________________________________________________________________

  | title | company | (link)"""


  return {"messages":[llm_with_web.invoke(prompt)]}


# %%
def combiner(state : State) :
  job_listings = state["messages"][2].content
  courses = state["messages"][-1].content
  about = read_file('report.txt')

  prompt=f"""
  for the following job listings :
    {job_listings}
  and the following {courses}

  please ensure that you give the links in the table 

  display both in a marked down table ,

  
  also the following is additional details about user : 

  {about}

  from your side recoomend the courses and the jobs that you find in the list would be good for the user based on thier 
  additional details what you think and why that perticular course would be beificial / perticular jobs would be good / easy to ace 

  SECTION 1 (jobs):
  ________________________________
  table for jobs

  it should contain the following columns 
  | Job Title |	Company | 	Skills | 	Link | 

  SECTION 2 (courses)  :

  it should contain the following columns 

  | Course Title | Link | Snippet |

  Section 3 (ai suggestion): 

  your insight ( normal para )
  
  table for courses

  """

  return {"messages":[llm.invoke(prompt)]}

# %%
tool_node = ToolNode(tools=web_tool)
job_node = ToolNode(tools=tools)

def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, ToolMessage):
        return "combiner"
    if last_message.tool_calls:
        return "web_tools"

    else :
        return END

# %%
graph_builder= StateGraph(State) # A graph is created to manage the flow of tasks. The State defines the structure (state) that the graph will use

graph_builder.add_node("job_finder",job_finder)
graph_builder.add_node("role_extractor",role_extractor)
graph_builder.add_node("job_lister",job_recomender)
graph_builder.add_node("web_tools", tool_node)
graph_builder.add_node("combiner" , combiner)
graph_builder.add_node("tools" , job_node)

graph_builder.add_edge(START,"role_extractor")
graph_builder.add_conditional_edges("job_finder" , tools_condition )

graph_builder.add_edge("role_extractor", "job_finder")
graph_builder.add_edge("tools" , "job_finder")
graph_builder.add_edge( "job_finder" ,"job_lister")
graph_builder.add_conditional_edges("job_lister", should_continue, ["web_tools" , "combiner" , END])
graph_builder.add_edge("web_tools","combiner" )


graph_builder.add_edge("combiner",END)


# %%
graph=graph_builder.compile()

# %%
class IIIcell:
    def __init__(self):
        self.intrest = ''  # Initialize the interest variable in the constructor

    def add_intrest(self, intrest):
        self.intrest = intrest  # Set the user's interest

    def get_job_report(self):
        # Assuming `graph.stream` is correctly configured
        events = graph.stream(
            {"messages": [("user", self.intrest)]}, stream_mode="values"
        )

        last_event = None  # To keep track of the last event
        for event in events:
            event['messages'][-1].pretty_print()
            last_event = event  # Store the last event

        # Optionally print the last event's message
        if last_event:
           result =  last_event["messages"][-1]


        from langchain_core.pydantic_v1 import BaseModel, Field 
        from typing import List 

        class Course(BaseModel):
            Course_Title: str = Field(description="The Title of Course")
            Link: str = Field(description="The link for the Course")
            Snippet : str = Field(description = "small snippet of the course")

        class Jobs(BaseModel):
            Job_Title: str = Field(description="The Title of Job")
            Company: str = Field(description="The Company of Job")
            Skills: str = Field(description="The Skills required for the Job")
            Link: str = Field(description="The link for the Job")

        class Info(BaseModel):
            Job_Listings: List[Jobs] = Field(description="List of Jobs")
            Courses: List[Course] = Field(description="List of Courses")
            suggestion : str = Field(description="the suggestion given by  the ai ")



        structured_llm = llm.with_structured_output(Info)

        struct_answer = structured_llm.invoke(f"""

        given the following output 

        {last_event["messages"][-1].content}


        format it in a structured format in json form  
        """)

        return struct_answer.dict()