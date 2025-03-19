from staff import Teacher
from tpo import IIIcell
from library import Librarian
from astrapy import DataAPIClient
from datetime import datetime
import uuid 
import requests
from datetime import datetime
from flask import Flask , request , jsonify
from flask_cors import CORS
import requests
import os
from dotenv import dotenv_values
from coursegen import give_gen_courses
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List 

config = dotenv_values(".env") 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

######################### Libraries ################################################

client = DataAPIClient(config['ASTRA_DB_CLIENT'])
db = client.get_database_by_api_endpoint(config['ASTRA_DB_ENDPOINT'])
student_collection = db.get_collection("student")
test_collection = db.get_collection("test_report_db")
course_collection = db.get_collection("courses")

teacher = Teacher()
librarian = Librarian()
new_cell = IIIcell()

BASE_API_URL = config['LANGFLOW_BASE_API_URL']
LANGFLOW_ID = config['LANGFLOW_ID']
FLOW_ID = config['FLOW_ID']
APPLICATION_TOKEN = config['LANGFLOW_APPLICATION_TOKEN']


def db_bot(userid , query) : 

    student = student_collection.find_one({"_id": userid})
    mark_down_report = f"following are the activities of user {student['name']} \n"
    mark_down_report += f" ================ tests ================= \n" 
    mark_down_report += f" |time|title|marks|report|\n"
    for activity in student['activity']:
            if 'activity_name' in activity and isinstance(activity['activity_name'], dict):
                if activity['activity_name'].get('name') == 'test':
                    mark_down_report += f" | {activity.get('time_stamp', '')}"

                    if 'title' in activity['activity_name']:
                        mark_down_report += f" | {activity['activity_name']['title']}"

                    if 'result' in activity['activity_name']:
                        mark_down_report += f" : {activity['activity_name']['result']}"

                    if 'report' in activity['activity_name']:
                        mark_down_report += f" => {activity['activity_name']['report']}"

                    mark_down_report += " |\n"



    mark_down_report += " \n================ library ================= \n"
    for activity in student['activity']:
            if(activity['activity_name']['name'] == 'library'):
                mark_down_report += f"{activity['time_stamp']} \n"

    mark_down_report+= f" \n=============== Job Search ================= \n"

    for activity in student['activity']:
            if(activity['activity_name']['name'] == 'Job Search'):
                mark_down_report += f"{activity['time_stamp']} \n"


    mark_down_report+= f" \n=============== Study Sessions ================= \n"

    for activity in student['activity']:
            if(activity['activity_name']['name'] == 'Study Session'):
                mark_down_report += f"| {activity['time_stamp']} | {activity['activity_name']['topic']}  \n"




    from langchain_groq import ChatGroq # it's a class provided by langchain_groq and used as an interface with language model (rapper class)

    llm=ChatGroq(groq_api_key=config['GROQ_API_KEY_STAFF'],model_name="llama3-70b-8192")

    res = llm.invoke(f"""
        you are an ai academic advisor your task is to advise students 
        
        the user : {student['name']}
        has the following query :  {query}

        
        from the following info from database  : {mark_down_report}

        your job is to do following 

        
        now for the given query and information fetched from the database 
        1st answer the doubt of student and then advise to do furter things based on activity 

        note : if in the information if there is mentioned the stuendent id do not mention in your response 
        
        """)

    return res.content

def register(username, email, image_url):
    # Check if a user with the given email already exists
    existing_user = student_collection.find_one({"email": email})
    
    if existing_user:
        # Update the user's name, email, and image_url; leave 'activity' unchanged
        student_collection.update_one(
            {"_id": existing_user["_id"]},
            {"$set": {"name": username, "email": email, "image_url": image_url}}
        )
        # Retrieve and return the updated user document
        updated_user = student_collection.find_one({"_id": existing_user["_id"]})
        return updated_user
    else:
        # Create a new user document
        new_student = {
            "name": username,
            "email": email,
            "image_url": image_url,
            "activity": []
        }
        # Insert the new user document
        inserted_id = student_collection.insert_one(new_student).inserted_id
        # Retrieve and return the newly inserted user document
        new_user = student_collection.find_one({"_id": inserted_id})
        return new_user


def login(email):
    student = student_collection.find_one({"email": email})
    return student



def store_librarian(user_id):
    import os
    student = student_collection.find_one({"_id": user_id})
    if not student:
        return {"error": "Student not found"}
    try : 
        librarian.add_material('content')


        print(librarian.index)
    except Exception as e :
        print(f"Teacher error: {e}")
        return {"error": "Teacher error"}
    print("hi")
    activity_log = {
        "time_stamp": datetime.utcnow().isoformat(),
        "activity_name": {"name": "library"}
    }
    student_collection.update_one(
        {"_id": user_id},
        {"$push": {"activity": activity_log}}
    )

def ask_teacher(user_id , prompt) : 
    student = student_collection.find_one({"_id": user_id})
    if not student:
        return {"error": "Student not found"}
    try : 
        response = teacher.ask(prompt)
    except Exception as e :
        print(f"Teacher error: {e}")
        return {"error": "Teacher error"}
    activity_log = {
        "time_stamp": datetime.utcnow().isoformat(),
        "activity_name": {"name": "doubt" , "title" : response['title'] }
    }
    student_collection.update_one(
        {"_id": user_id},
        {"$push": {"activity": activity_log}}
    )
    return response['answer']



def generate_test_paper(prompt) :
    import os
    response = teacher.generate_test(prompt)
    return response


def evaluate(data):
   answer_sheet = """ """

   for answer in data : 
      answer_sheet+=f"""  
      title : {answer['context_title']}
      question : {answer['question']}
      answer : {answer['answer']}"""


   report = teacher.give_evaluation_report(answer_sheet)
   for i , q in enumerate(report['report']) : 
      q['answer'] = data[i]['answer']
      q['question'] = data[i]['question']

   return report 


def evaluate_test(user_id , data) : 
    student = student_collection.find_one({"_id": user_id})
    if not student:
        return {"error": "Student not found"}
    response = evaluate(data)

    total_marks = 0
    for question in response['report']:
        total_marks+= question['marks']
        
    activity_log = {
        "time_stamp": datetime.utcnow().isoformat(),
        "activity_name": {"name": "test" , "result" : total_marks , "report" : response['review'] , "title" : response['title']}
    }
    student_collection.update_one(
        {"_id": user_id},
        {"$push": {"activity": activity_log}}
    )
    test_collection.insert_one({
        "student_id": user_id , 
        "report" : response ,
        "time_stamp" : datetime.utcnow().isoformat()
    })
    return response

def ask_placement(userid , search):

    new_cell.add_intrest(search)
    report = new_cell.get_job_report()

    activity_log = {
        "time_stamp": datetime.utcnow().isoformat(),
        "activity_name": {"name": "Job Search" }
    }

    student_collection.update_one(
        {"_id": userid},
        {"$push": {"activity": activity_log}}
    )

    return report





################### API GATEWAY ##################################################






app = Flask(__name__)
CORS(app)


@app.route('/register', methods=['POST'])
def register_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    username = data.get("username")
    email = data.get("email")
    image_url = data.get("image_url", "")  # Default to empty string if not provided

    if not username or not email:
        return jsonify({"error": "Missing required fields: username and email"}), 400

    try:
        user = register(username, email, image_url)
        # Convert ObjectId to string for JSON serialization
        user['_id'] = str(user['_id'])
        return jsonify(user), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        if not data or 'email' not in data:
            return jsonify({'error': 'Email is required'}), 400
        email = data.get('email')
        if not isinstance(email, str) or '@' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        user = login(email)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        return jsonify(user), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



import json

@app.route('/dbchat' , methods=['POST'])
def ask_db():
    try : 
        data = request.get_json()
        userid = data.get('userid')
        prompt = data.get('prompt')
        response = db_bot(userid , prompt)
        return jsonify({'response' : response}) , 200
    except Exception as e : 
        print(e)
        return jsonify({'error' : str(e)}) , 500



@app.route('/librarian/add_material', methods=['POST'])
def add_material():
    try:
        # Create the 'content' folder if it doesn't exist
        if not os.path.exists('content'):
            os.mkdir('content')
        
        # Check for files in the request
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')  # Get multiple files
        saved_files = []

        for file in files:
            if file.filename == '':
                continue  # Skip empty files
            
            file_path = os.path.join('content', file.filename)
            file.save(file_path)
            saved_files.append(file.filename)

        # Instead of request.get_json(), get the JSON string from the form field 'data'
        data_str = request.form.get('data')
        if not data_str:
            return jsonify({'error': 'No JSON data provided in form field "data"'}), 400
        
        # Parse the JSON string
        data = json.loads(data_str)
        userid = data.get('userid')
        if not userid:
            return jsonify({'error': 'User ID is required in JSON data'}), 400

        store_librarian(userid)

        if not saved_files:
            return jsonify({'error': 'No valid files uploaded'}), 400

        return jsonify({'message': 'Files uploaded successfully', 'files': saved_files}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    


@app.route('/librarian/query_material', methods=['POST'])
def query_material():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        response = librarian.query_material(prompt).content
        return jsonify(response), 200
    except Exception as e:

        return jsonify({'error': str(e)}), 500
    

@app.route('/librarian/delete', methods=['GET'])
def delete_material():
    try:
        librarian.remover_material()
        return jsonify({'message': 'Material deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/teacher/ask', methods=['POST'])
def ask_teacher_api():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        response = ask_teacher(data.get('userid') , prompt)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
PDF_PATH = 'context.pdf'


@app.route('/teacher/generate_test', methods=['POST'])
def generate_test_api():
    try:
        # Check for uploaded file
        if 'pdf' not in request.files:
            return jsonify({'error': 'PDF file is required'}), 400

        pdf_file = request.files['pdf']
        pdf_file.save(PDF_PATH)  # Save the file as context.pdf in the main folder

        # Process JSON data
        data_str = request.form.get('data')
        data = json.loads(data_str)
        prompt = data.get('prompt')
        

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Generate test paper using the provided prompt
        response = generate_test_paper(prompt)

        # Delete the stored PDF before returning response
        if os.path.exists(PDF_PATH):
            os.remove(PDF_PATH)

        return jsonify(response), 200

    except Exception as e:
        # Ensure PDF is deleted even if an error occurs
        print(e)
        if os.path.exists(PDF_PATH):
            os.remove(PDF_PATH)

        return jsonify({'error': str(e)}), 500



ANSWERSHEET_PATH = 'answersheet.txt'  # Path for storing the uploaded answer sheet

@app.route('/teacher/evaluate_test', methods=['POST'])
def evaluate_test_api():
    try:
        data = request.get_json()
        userid = data.get('userid')
        answer_sheet = data.get('answer_sheet')
        # Evaluate the test
        response = evaluate_test(userid , answer_sheet)


        return jsonify(response), 200

    except Exception as e:
        # Ensure answersheet is deleted even if an error occurs
        if os.path.exists(ANSWERSHEET_PATH):
            os.remove(ANSWERSHEET_PATH)

        return jsonify({'error': str(e)}), 500
    



REPORT_PATH = 'report.txt'  # Path for storing the uploaded report file

@app.route('/placement/search', methods=['POST'])
def search_placement():
    try:
        # Ensure at least one file is uploaded
        if len(request.files) == 0:
            return jsonify({'error': 'Report file is required'}), 400

        # Get the first uploaded file and save it as 'report.txt'
        file = list(request.files.values())[0]
        file.save(REPORT_PATH)

        # Process JSON data
        data_str = request.form.get('data')
        data = json.loads(data_str)
        userid = data.get('userid')
        prompt = data.get('prompt')

        if not userid or not prompt:
            return jsonify({'error': 'User ID and prompt are required'}), 400

        # Search placement details
        response = ask_placement(userid, prompt)

        # Delete the stored report before returning response
        if os.path.exists(REPORT_PATH):
            os.remove(REPORT_PATH)

        return jsonify(response), 200

    except Exception as e:
        # Ensure report.txt is deleted even if an error occurs
        if os.path.exists(REPORT_PATH):
            os.remove(REPORT_PATH)

        return jsonify({'error': str(e)}), 500

@app.route('/test_reports', methods=['POST'])
def test_reports():
    try:
        data = request.get_json()
        userid = data.get('userid')
        cursor = test_collection.find({"student_id": userid})
        response = list(cursor)
        for doc in response:
            doc['time_stamp'] = datetime.fromisoformat(doc['time_stamp'])  
        response.sort(key=lambda x: x['time_stamp'], reverse=True)
        for doc in response:
            doc['time_stamp'] = doc['time_stamp'].isoformat()

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/generate_course", methods=["POST"])
def generate_course():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        required_fields = ["userid", "department", "branch", "interest", "duration", "n_course"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required parameters: {', '.join(missing_fields)}"}), 400
        
        userid = data["userid"]
        department = data["department"]
        branch = data["branch"]
        interest = data["interest"]
        duration = data["duration"]
        n_course = data["n_course"]
        remarks = data["remarks"]
        
        if not isinstance(n_course, int) or n_course <= 0:
            return jsonify({"error": "n_course must be a positive integer"}), 400
        import inspect
        print(inspect.getfile(give_courses))
        print(remarks)
        courses = give_gen_courses(department, branch, interest, remarks , n_course , duration  )
        
        course_collection.insert_one({
            "userid": userid,
            "name": interest,
            "course": courses
        })
        
        return jsonify({"userid": userid, "interest": interest, "courses": courses}), 201
    
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route('/get_courses', methods=['POST'])
def give_courses():
    data = request.get_json()
    userid = data.get("userid")

    if not userid:
        return jsonify({"error": "Missing userid"}), 400

    cursor = course_collection.find({"userid": userid})
    courses = list(cursor)

    # Convert ObjectId to string
    for course in courses:
        course["_id"] = str(course["_id"])

    return jsonify(courses) , 200 

@app.route("/update_notes", methods=["POST"])
def update_study_material_notes():
    data = request.json
    courseid = data.get("courseid")
    subcourseidx = data.get("subcourseidx")
    topicidx = data.get("topicidx")
    study_index = data.get("study_index")
    notes_text = data.get("notes_text")

    # Fetch the course document
    course = course_collection.find_one({"_id": courseid})
    user_id = course['userid']
    if not course:
        return jsonify({"error": "Course not found"}), 404

    try:
        # Navigate to the specific study material
        pdf_material_len = len(course['course']['content'][subcourseidx]['timeline'][topicidx]["study_material"][0]['pdf'])
        
        if study_index >= pdf_material_len:
            study_type = "video"
            study_index = study_index - pdf_material_len
        else:
            study_type = "pdf"

        study_material = course['course']['content'][subcourseidx]['timeline'][topicidx]["study_material"][0][study_type][study_index]
        
        # Update or add the 'notes' field
        study_material['notes'] = notes_text

        # Update the database with the modified study material
        course_collection.update_one(
            {"_id": courseid},
            {"$set": {f"course.content.{subcourseidx}.timeline.{topicidx}.study_material.0.{study_type}.{study_index}": study_material}}
        )
        activity_log = {
            "time_stamp": datetime.utcnow().isoformat(),
            "activity_name": {"name": "Study Session" , "topic" : course['course']['content'][subcourseidx]['name']} 
        }
        student_collection.update_one(
            {"_id": user_id},
            {"$push": {"activity": activity_log}}
        )

        return jsonify({"message": "Notes updated successfully"})
    
    except (KeyError, IndexError) as e:
        return jsonify({"error": "Invalid indices or structure issue", "details": str(e)}), 400

#change by aagam
@app.route('/api/groq', methods=['POST'])
def groq_endpoint():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        class Questions(BaseModel):
            questions: List[str] = Field(description="List of all the questions") 

        llm = ChatGroq(groq_api_key=config['GROQ_API_KEY_STAFF'], model_name="llama-3.3-70b-versatile")
        structured_llm = llm.with_structured_output(Questions)

        structured_output = structured_llm.invoke(prompt).dict()
        
        return jsonify(structured_output), 200  # Return structured output as JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port =5000
    app.run(debug=False)