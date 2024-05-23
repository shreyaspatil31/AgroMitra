from flask import jsonify
from mysql import connector

db = connector.connect(
    host="localhost",
    user="root",
    password="Patil@3152",
    database="majorProject"
)

cursor = db.cursor()

def get_solution(crop_type, disease_name):
    query = "SELECT COMPANY, CONTENT, MAKE_USE FROM SOLUTION WHERE CROP_NAME = %s AND DISEASE_NAME = %s"
    cursor.execute(query, (crop_type, disease_name))
    solution = cursor.fetchone()
    
    if solution:
        solution_data = {
            'company': solution[0],
            'content': solution[1],
            'makeUse': solution[2]
        }
        return solution_data
    else:
        return {"error": "Error: Solution Not Found!"}
