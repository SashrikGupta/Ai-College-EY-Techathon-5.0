import json
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import yt_dlp
from exa_py import Exa

# Initialize Exa with your API key
exa = Exa(api_key="bc5a029b-9ea2-4d09-baeb-7985349c75ea")

# Initialize the LLM
llm = ChatGroq(
    api_key='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',
    model="llama-3.3-70b-versatile"
)

# Load course data
with open('courses.json', 'r') as f:
    data = json.load(f)

def search_youtube(query, max_results=3):
    search_url = f"ytsearch{max_results}:{query}"
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        result = ydl.extract_info(search_url, download=False)
    return [
        {"title": video["title"], "link": video["webpage_url"]}
        for video in result["entries"]
    ]

def search_on_net(topic, query):
    """
    Search for study materials online using two sources:
      - YouTube tutorials
      - PDF resources via Exa search (using the exa_py package)
    """
    yt_results = search_youtube(f"tutorials on {query}")
    
    # Perform the Exa search for PDFs
    result = exa.search(f"topic: {topic} pdfs to study: {query}", category="pdf", type="auto")
    
    # Extract PDF materials using the provided snippet logic
    pdf_material = []
    for ex in result.results:
        pdf_material.append({
            "title": ex.title,
            "link": ex.url
        })
    
    return {
        "video": yt_results,
        "pdf": pdf_material
    }

def give_gen_courses(department, branch, intrest, top=3, total_time=60):
    class course(BaseModel):
        index: int = Field(description="The index of the course from the list")
        name: str = Field(description="The name of the course")

    class Courses(BaseModel):
        content: List[course] = Field(description="List of all the courses")

    structured_llm = llm.with_structured_output(Courses)
    list_data = ""
    for i, name in enumerate(data[department]['departments'][branch]):
        list_data += f"{i} : {name['title']}\n"

    answer = structured_llm.invoke(
        f"given course infos {list_data} find top {top} courses that would be suitable for user with intrest in {intrest} note all the cours must be different also arrange them in a order for student to learn sequentially"
    ).dict()

    for ans in answer['content']:
        ans['description'] = data[department]['departments'][branch][ans['index']]['description']

    mark_down = ""
    for ans in answer['content']:
        mark_down += f"\n# {ans['name']}\n"

    class Duration(BaseModel):
        duration: List[int] = Field(description="Give the duration of the course in days")

    schedule_llm = llm.with_structured_output(Duration)
    schedule = schedule_llm.invoke(
        f"""
        given the following courses :
        {mark_down}
        your task is to allocate days according to the requirement of course in days
        note the given duration of collection of all topics is {total_time} days
        """
    ).dict()

    for i, ans in enumerate(answer['content']):
        ans['duration'] = schedule['duration'][i]

    class Activity(BaseModel):
        topic_name: str = Field(description="Names of the topics that will be covered")
        start_time: int = Field(description="Give the start time of the activity")
        end_time: int = Field(description="Give the end time of the activity")
        activity_type: int = Field(description="'1 for Library', '2 for Test', '3 for Research'")
        description: str = Field(description="Description of the activity")
        course_index: int = Field(description="Index of the course")

    class TimeLine(BaseModel):
        timelines: List[Activity] = Field(description="Timeline of the course")

    context = '======================================================================================================='
    for i, ans in enumerate(answer['content']):
        context += f"\ncourse index : {i}\ncourse : {ans['name']}\nduration : {ans['duration']} days\ndescription : {ans['description']}\n"
    context += """
    ===============================================================================================================
      For each course, our platform provides three key learning activities:

    1. **Library Session:**
      - Students can upload and access study material.
      - The session is dedicated to learning fundamental concepts.

    2. **Test Session:**
      - Students can generate and take tests to reinforce their understanding.

    3. **Research Session (if applicable):**
      - Students can upload research papers and explore advanced topics in-depth.

    **Task:**

    Your objective is to create a structured day-wise learning plan for each course, ensuring the following:

    - Each day should be allocated to **only one activity** (Library, Test, or Research).
    - The sequence should be:

      - Start with **Library sessions** to cover the fundamental concepts.
      - Follow up with **Test sessions** to assess understanding.
      - (The above two can be repeated according to the breakdown of chapters/content in the course)
      - Include **Research activities** if the course involves advanced topics.
      - **Always conclude with a Final Test on the last day.**
    - The breakdown should be **structured and easy to follow.**

    **Expected Output Format:**

    For each course, provide a **detailed schedule** similar to the example below:

    ---

    ### **Example Schedule for Course: Reinforcement Learning (Duration: 10 days)**

    - **Day 1-3:**  topic_name : reinforcement learning, dynamic programming  Library – Study **Introduction to Reinforcement Learning** and **Dynamic Programming (DP) techniques**.  // describe briefly what to study
    - **Day 4-4:** topic_name :  reinforcement learning, dynamic programming  Test – Assess knowledge on **DP, RL, and Monte Carlo methods**. // describe briefly what to study
    - **Day 5-6:** topic_name :  deep q networks  Library - read about Deep Q networks and temporal difference
    - **Day 7-7:** topic_name :  deep q networks , Test - Master **Deep Q networks and temporal difference**
    - **Day 8-9:** topic_name :  trending topics in RL , Research – Explore **trending topics in RL** and conduct research.  // describe the research topic briefly
    - **Day 10:** topic_name :  final test  Final Test – Comprehensive evaluation of the entire course.

    ---

    **Repeat this process for each course with appropriate topics and scheduling.**

    **Note:** The test will be of only one day.

    ===============================================================================================================
    """

    timeline_llm = llm.with_structured_output(TimeLine)
    ouput = timeline_llm.invoke(context).dict()

    for i, ans in enumerate(answer['content']):
        course_time_line = []
        for timeline_item in ouput['timelines']:
            if timeline_item['course_index'] == i:
                course_time_line.append(timeline_item)
        ans['timeline'] = course_time_line

    for ans in answer['content']:
        for timeline in ans['timeline']:
            if timeline['activity_type'] == 1:
                study_material = []
                study_material.append(search_on_net(intrest, timeline['topic_name']))
                timeline['study_material'] = study_material

    return answer
