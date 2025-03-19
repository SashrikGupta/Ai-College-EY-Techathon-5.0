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

config = dotenv_values(".env") 

######################### Libraries ################################################

client = DataAPIClient(config['ASTRA_DB_CLIENT'])
db = client.get_database_by_api_endpoint(config['ASTRA_DB_ENDPOINT'])
student_collection = db.get_collection("student")

teacher = Teacher()
librarian = Librarian()
new_cell = IIIcell()

BASE_API_URL = config['LANGFLOW_BASE_API_URL']
LANGFLOW_ID = config['LANGFLOW_ID']
FLOW_ID = config['FLOW_ID']
APPLICATION_TOKEN = config['LANGFLOW_APPLICATION_TOKEN']


def db_bot(message):
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{FLOW_ID}"

    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
        "session_id" : uuid.uuid4().hex
    }
    headers = None

    headers = {"Authorization": "Bearer " + APPLICATION_TOKEN, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    response = response.json()
    response = response['outputs'][0]['outputs'][0]['results']['message']['text']

    return response

def register(username, email, age, gender, semester, weakness, strength, resume):
    new_student = {
        "name": username,
        "email": email,
        "age": age,
        "gender": gender,
        "semester": semester,
        "strength": strength,
        "weakness": weakness,
        "resume": resume,
        "activity": []
    }
    
    response = student_collection.insert_one(new_student) 
    
    return response.inserted_id


def login(email):
    student = student_collection.find_one({"email": email})
    return student



def store_librarian(user_id):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    student = student_collection.find_one({"_id": user_id})
    if not student:
        return {"error": "Student not found"}
    librarian.add_material('content')
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
    response = teacher.ask(prompt)
    activity_log = {
        "time_stamp": datetime.utcnow().isoformat(),
        "activity_name": {"name": "doubt"}
    }
    student_collection.update_one(
        {"_id": user_id},
        {"$push": {"activity": activity_log}}
    )
    return response



def generate_test_paper(prompt) :
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
    response = teacher.generate_test(prompt)
    return response



def evaluate_test(user_id) : 
    student = student_collection.find_one({"_id": user_id})
    if not student:
        return {"error": "Student not found"}
    teacher.upload_answer_sheet('answer_sheet.txt')
    response = teacher.give_evaluation_report()

    total_marks = 0
    for question in response['report']:
        total_marks+= question['marks']
        
    activity_log = {
        "time_stamp": datetime.utcnow().isoformat(),
        "activity_name": {"name": "test" , "result" : total_marks , "report" : response['review']}
    }
    student_collection.update_one(
        {"_id": user_id},
        {"$push": {"activity": activity_log}}
    )
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
def register_user():
    try:
        data = request.get_json()
        required_fields = ['username', 'email', 'age', 'gender', 'semester', 'weakness', 'strength', 'resume']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        username = data.get('username')
        email = data.get('email')
        age = data.get('age')
        gender = data.get('gender')
        semester = data.get('semester')
        weakness = data.get('weakness')
        strength = data.get('strength')
        resume = data.get('resume')

        if not isinstance(age, int) or age <= 0:
            return jsonify({'error': 'Invalid age'}), 400

        user_id = register(username, email, age, gender, semester, weakness, strength, resume)
        
        return jsonify({'user_id': str(user_id)}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500




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
        response = db_bot(f""" 
        the user with _id : {userid}
        has the following doubt : 
        {prompt}
        solve the query of user with respect to its id 
        """)
        return jsonify({'response' : response}) , 200
    except Exception as e : 
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
        if os.path.exists(PDF_PATH):
            os.remove(PDF_PATH)

        return jsonify({'error': str(e)}), 500



ANSWERSHEET_PATH = 'answersheet.txt'  # Path for storing the uploaded answer sheet

@app.route('/teacher/evaluate_test', methods=['POST'])
def evaluate_test_api():
    try:
        # Ensure at least one file is uploaded
        if len(request.files) == 0:
            return jsonify({'error': 'Answer sheet file is required'}), 400

        # Get the first uploaded file and save it as 'answersheet.txt'
        file = list(request.files.values())[0]
        file.save(ANSWERSHEET_PATH)

        # Process JSON data
        data_str = request.form.get('data')
        data = json.loads(data_str)
        userid = data.get('userid')

        if not userid:
            return jsonify({'error': 'User ID is required'}), 400

        # Evaluate the test
        response = evaluate_test(userid)

        # Delete the stored answer sheet before returning response
        if os.path.exists(ANSWERSHEET_PATH):
            os.remove(ANSWERSHEET_PATH)

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


if __name__ == "__main__":
    port =5000
    app.run(host='0.0.0.0', port=port, debug=False)
