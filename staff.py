# %%
from langchain_openai import ChatOpenAI
import os
from crewai_tools import PDFSearchTool
from crewai.tools  import tool
from crewai.process import Process
from crewai import Crew
from crewai import Task
from crewai import Agent
from dotenv import dotenv_values

config = dotenv_values(".env") 


# %%
os.environ['GROQ_API_KEY'] = config['GROQ_API_KEY_STAFF']

# %%
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ['GROQ_API_KEY'],
    model_name="groq/llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1000,
)

# %%

from langchain_community.retrievers import WikipediaRetriever

@tool('web_search_tool')
def web_search_tool(question: str) -> str:
    """
    Tool for conducting web searches on wikipedia and retrieving summarized content.

    **Inputs**:
      - `question` (str): A **plain string** representing a concise search query
        (e.g., "latest AI trends", "Python web scraping").
      - **Note**: `question` should be a **string only**, not a dictionary or complex object.

    **Input Type**:
      - `question` is expected to be a **string**.

    """
    retriever = WikipediaRetriever()
    docs = retriever.invoke(question)
    info = ""
    for word in docs[0].page_content.split(' ')[0:600]:
        info += word + " "
    return info

# %%
def intialize_crew(rag_tool):
    # Topic Provider Agent
    topic_provider = Agent(
        role="Instructor",
        goal=(
            "Analyze the topic '{input}' and identify all key study areas necessary for a full understanding and readiness for job roles related to this topic. "
            "Break down the topic into clear, essential subjects and subtopics that will provide a thorough foundation. "
            "Each topic should be relevant to the job requirements and enable comprehensive preparation."
            "do not go too far from topic"
        ),
        backstory="A knowledgeable instructor providing a structured outline of essential topics and subtopics for effective study.",
        verbose=True,
        llm=llm
    )



    # PDF Content Extractor Agent
    pdf_content_extractor = Agent(
        role="PDF Content Extractor",
        goal=(
            "add on to the information of instructor llm  by finding '{input}' in pdf "
            "Pass input as a simple text query ('query') and retrieve informations from pdf  "
            "try to search a lot "
        ),
        backstory="An instructor specialized in extracting comprehensive knowledge directly from a PDF document for full topic coverage.",
        verbose=True,
        llm=llm
    )

    # Web Content Extractor Agent
    web_content_extractor = Agent(
        role="Web Content Extractor",
        goal=(
            "do a search for tpoics on '{input}' given by the topic provider , just one seach is okay "
            "Tool input will be :  question : ' just a plain string' "
            "return what you found related to the topics provided by the web"
            "your final output should be  some conetnt from web"
        ),
        backstory="An instructor collecting extensive web-based knowledge on each subtopic to fill any gaps in PDF content.",
        verbose=True,
        llm=llm
    )


    # Information Combiner Agent
    info_combiner = Agent(
        role="Information Combiner",
        goal=(
            "Combine the detailed content gathered from both the PDF and web sources into a single, cohesive study guide. "
            "Organize and synthesize the information to ensure each subtopic is explained in depth, covering all angles required for job preparedness. "
            " not you have to give comrehensive explanation theoretical explanations"
            "Deliver the final material in a structured format, ensuring it is clear, logical, and aligns with the requirements of the topic and job description."
        ),
        backstory="A tutor combining insights from multiple sources to create a well-rounded and thorough study guide for the user.",
        verbose=True,
        llm=llm
    )


    # Topic Summary Task
    topic_summary_task = Task(
        description="Provide a structured TL;DR summary outlining the key topics, subjects  necessary to prepare for and excel in the role described in the job description. Focus on the skills and knowledge required for success in this role.",
        agent=topic_provider,
        expected_output="combined informations gathered from searching"
    )


    # PDF Search Task
    pdf_search_task = Task(
        description="Locate and extract information in the PDF document for topic.",
        agent=pdf_content_extractor,
        tools=[rag_tool],
        context=[topic_summary_task],
        expected_output="A list of  contexts (contexts size sould be 40-30 words) from the PDF that address each subtopic. No web content; PDF-only focus.",
    )

    # Web Search Task
    web_search_task = Task(
        description=" extract a brief, relevant section from the web to add to the info ",
        agent=web_content_extractor,
        tools=[web_search_tool],
        context=[topic_summary_task],
        expected_output="some  relevant output should be  some conetnt from web"
    )


    # Information Combination Task
    info_combination_task = Task(
        description=(
            "Your task is to combine the information gathered by the PDF content extractor and web search extractor agents. "
            "Provide exactly **10 comprehensive and extensive points** related to the study topic. "
            "Each point should be a distinct subtopic or key aspect, followed by a thorough and detailed explanation. "
            "You should provide only the points and explanations, without any source information or metadata. "
            "Each point should be fully explained to give a complete understanding of the topic. "
            "Return the points as a plain list, without any formatting or structure other than the list itself."
        ),
        agent=info_combiner,
        expected_output=(
            "Provide 10 comprehensive and extensive points. Each point should be a distinct subtopic or key aspect, followed by a detailed explanation. "
            "Do not include any sources, metadata, or JSON formatting. Just return the points in plain text."
        ),
    )

    crew = Crew(
        agents=[topic_provider, pdf_content_extractor, web_content_extractor , info_combiner],
        tasks=[topic_summary_task, pdf_search_task, web_search_task , info_combination_task],
        process=Process.sequential,
        Max_RPM=2,
        verbose=True,
    )

    return crew


# %%
import typing_extensions as typing
import google.generativeai as genai
import json


genai.configure(api_key=config['GEMINI_API_KEY'])


def convert_to_json(result: str) :
      from langchain_core.pydantic_v1 import BaseModel, Field 
      from typing import List 
      from langchain_groq import ChatGroq # it's a class provided by langchain_groq and used as an interface with language model (rapper class)
      class content(BaseModel):
        context_title: str = Field(description="topic from which question has been asked")
        question: str = Field(description="question generated from the context")

      class Format(BaseModel):
          format : List[content] = Field(description="array of questions")

      groq_api_key= config['GROQ_API_KEY_STAFF']
      llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192")
      structured_llm = llm.with_structured_output(Format)
      struct_result = structured_llm.invoke(f""" given context : 
                                     
          {result} 
          
           your task is to generate 10 question from the context for each topic in json format cover all the topics gievn in the result 
           the questions should be length , at each question either ask to explain with example , or explain briefly , explain all the parts etc...


           "for some questions , where you think real life examples would work , then ass practical questions with examples from real life "
          """)

      return struct_result.dict()


# %%

import google.generativeai as genai


def evelauate(answer_sheet) : 
      from langchain_core.pydantic_v1 import BaseModel, Field 
      from typing import List 
      from langchain_groq import ChatGroq # it's a class provided by langchain_groq and used as an interface with language model (rapper class)
      groq_api_key=config['GROQ_API_KEY_STAFF']
      llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192")
      class Marks(BaseModel):
            question_no: int = Field(description="the question number")
            title: str = Field(description="The title of the concept")
            marks : int = Field(description = "marks allocated")
            question : str = Field(description = "the question")
            reason : str = Field(description="the Reason for assigning such marks")

      class Report(BaseModel):
            report : List[Marks] = Field(description="list of all the questions evaluated")
            review : str = Field(description="short overall review / description of the evaluation report in 3-4 words for eg : could have don better , good work , excellent etc..")


      structured_llm = llm.with_structured_output(Report)

      struct_answer = structured_llm.invoke( f"""



          you are an experienced instructor who evaluates answers with fairness, 
          offering constructive feedback and insights into the candidate's comprehension level be a little harsh


          Evaluate each answer based on accuracy, relevance, completeness, and clarity. For each question, assign marks out of the maximum allowed.
            Explain the reasoning behind the marks awarded and provide insight into the candidate's current level of understanding for each question.

          {answer_sheet}

           the following should be mentioned in the output

        -  The question number.
        - Marks awarded out of the total possible for that question.
        - Explanation for why the specific marks were awarded, considering accuracy, relevance, and completeness.
        - Observations about the candidate's current understanding based on their answer.
        - each question is of 10 marks 

            give answer in the following json format : 

            
    [{{
               question_no : int  ,
               title : str  , 
               marks : int  , 
               question : str , 
               reason : str  , 
             }} , 
             {{
               question_no : int ,  
               title : str  , 
               marks : int  , 
               question : str , 
               reason : str 
             }} , 
             ]
            
             

             for example : 
             
            {{
            report : [{{
              question_no : 1
              title : 'fundamentals of motion'
              marks : 7
              question : 'explain fundamentals of velocity and acceleration along with examples '
              reason : 'while the concepts of velocity, displacement, and acceleration are correct but no explanation is provided'
             }} , 
             {{
              question_no : 2
              title : 'equation of motion'
              marks : 10
              question : 'derive all three equations of motion'
              reason : 'correctly explained the derivation along with steps '          
             }}
             .....
             ]
            review : "well done"
            }}
             
             
        """)

      return struct_answer.dict()



import cohere
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
import json

DB_DIR = r"./rag_db"  
DB_PATH = os.path.join(DB_DIR, "faiss_index")  
CHUNKS_PATH = os.path.join(DB_DIR, "chunks.json")  

def pdf_vector_space(path_to_pdf):
    """Processes a PDF, generates embeddings, stores FAISS index and chunks in a JSON file."""
    
    # Load PDF and extract text
    loader = PyPDFLoader(path_to_pdf)
    pages = loader.load()
    content = [page.page_content for page in pages]

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", " "], 
        chunk_size=1000,                                  
        chunk_overlap=1                                 
    )
    
    chunks = []
    for page_content in content:
        chunks.extend(splitter.split_text(page_content))

    # Initialize Cohere client
    co = cohere.ClientV2(config['COHERE_API_KEY_STAFF'])
    
    # Generate embeddings
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="classification",
        embedding_types=["float"],
    )

    embeddings = response.embeddings.dict()['float']
    vectors = np.array(embeddings)         
    dim = vectors.shape[1]                 

    # Create FAISS index and add vectors
    index = faiss.IndexFlatL2(dim)         
    index.add(vectors)

    # Ensure the directory exists
    os.makedirs(DB_DIR, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, DB_PATH)
    print(f"FAISS index saved at {DB_PATH}")

    # Save chunks to JSON file
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    
    print(f"Chunks saved at {CHUNKS_PATH}")

# Example Usage
# pdf_vector_space("path/to/your/file.pdf")  # To process and store index + chunks


# %%

import cohere
import faiss
import numpy as np
import os

co = cohere.ClientV2(config['COHERE_API_KEY_STAFF'])  # Replace with your API key


@tool('rag_tool')
def rag_tool(query :str):

    """ 
    This tool extracts relevant content from a PDF using Retrieval-Augmented Generation (RAG). 
    It searches for the most relevant sections of the document based on the query and returns the extracted content. 
    
    **Inputs:**  
    - query (str): A natural language question or phrase to search for within the PDF.   

    **Output:**  
    - A string containing the most relevant extracted content from the PDF.  
      Example: `"React is a framework built on JavaScript that allows for the development of interactive UIs using a component-based architecture..."`  
      
    """
    index = faiss.read_index("./rag_db/faiss_index" )
    top_k = 3 
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="classification",
        embedding_types=["float"],
    )
    query_embedding = np.array(response.embeddings.dict()['float']).astype(np.float32)
    indices , doc = index.search(query_embedding, top_k)
    with open("./rag_db/chunks.json", "r", encoding="utf-8") as f:
        chuncks = json.load(f)
    data_return = ""
    for i in doc[0] :
        data_return += f""" 
          {chuncks[i]}
         """ 
    return data_return


class Teacher:
    def __init__(self):
        self.path = None
        self.answer_sheet = ""

    def upload_pdf(self , path ):
        self.path = path


    def ask(self, query):
        from langchain_groq import ChatGroq
        groq_api_key=config['GROQ_API_KEY_STAFF']
        llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192")
        return llm.invoke( f"""

            you are a helpful teacher that solves students doubt 

            answer the following query : {query}
        """).content    
    

    def generate_test(self, topic):
        os.mkdir('./rag_db')
        pdf_vector_space('context.pdf')    
        inputs ={"input": topic}
        crew = intialize_crew(rag_tool)
        result = crew.kickoff(inputs=inputs)
        out = convert_to_json(result.raw)

        import shutil
        shutil.rmtree('./rag_db')
   
        return out

    def upload_answer_sheet(self, path):
        with open(path, 'r') as file:
            self.answer_sheet = file.read()

    def give_evaluation_report(self):
      return evelauate(self.answer_sheet)



