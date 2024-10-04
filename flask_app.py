from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import threading
from assistant import (
    load_and_split_documents, 
    create_summary,
    save_to_pickle,
    load_from_pickle,
    default_FAISS,
    create_docstore,
    chat_gen,
    generate_follow_up_answer,
    evaluate_answer
)

from db import (init_db, 
store_query_metadata, 
store_response_metadata,  
store_knowledge_evaluation_metadata,
store_document_metadata, 
get_response_metadata)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the database is initialized when the Flask app starts
init_db()

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable to track knowledge base creation status
kb_status = "idle"

def create_knowledge_base(file_path):
    global kb_status
    kb_status = "creating"
    try:
        # Load, process, and store document metadata
        loaded_docs = load_and_split_documents(file_paths=[file_path])
        print(f"Loaded and split {len(loaded_docs)} documents.")

        doc_summary, extra_chunks = create_summary(loaded_docs)
        print(doc_summary, '\n')

        convstore = default_FAISS()
        docstore = create_docstore(extra_chunks, loaded_docs)

        save_to_pickle(convstore, docstore)
        
        kb_status = "ready"
    except Exception as e:
        print(f"Error creating knowledge base: {str(e)}")
        kb_status = "failed"

@app.route('/upload', methods=['POST'])
def upload_file():
    global kb_status
    # Clear the uploads folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Check for file in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        store_document_metadata(filename, file_path)
        
        # Start knowledge base creation in a separate thread
        threading.Thread(target=create_knowledge_base, args=(file_path,)).start()
        
        return jsonify({"message": "File uploaded successfully. Knowledge base creation started."}), 202

@app.route('/kb_status', methods=['GET'])
def get_kb_status():
    global kb_status
    return jsonify({"status": kb_status})

@app.route('/query', methods=['POST'])
def process_query():
    global kb_status
    if kb_status != "ready":
        return jsonify({"error": "Knowledge base is not ready"}), 400

    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Load convstore and docstore
    convstore, docstore = load_from_pickle()
    
    if not convstore or not docstore:
        return jsonify({"error": "Knowledge base not available"}), 500
    
    # Generate answer, bullet points, and follow-up question
    final_answer, bullet_points, follow_up_question = chat_gen(query, convstore, docstore)
    # Ensure bullet_points is a string
    if isinstance(bullet_points, list):
        bullet_points = "\n".join(bullet_points)  # Convert list to string, each bullet on a new line
    
    if isinstance(follow_up_question, list):
        follow_up_question = "\n".join(follow_up_question)  # Convert list to string, each bullet on a new line

    
    follow_up_answer = generate_follow_up_answer(follow_up_question, convstore, docstore)

    if isinstance(follow_up_answer, list):
        follow_up_answer = "\n".join(follow_up_answer)
    
    # Store metadata
    store_query_metadata(query)
    
    question_id = store_response_metadata(final_answer, bullet_points, follow_up_question, follow_up_answer)
    print("Answer, Bullet Points, Test Question, and Follow-up Answer Stored")

    return jsonify({
        "answer": final_answer,
        "bullet_points": bullet_points,
        "test_question": follow_up_question,
        "test_question_id": question_id
    }), 200

@app.route('/evaluate', methods=['POST'])
def evaluate_user_answer():
    user_answer = request.json.get('answer')
    test_question_id = request.json.get('test_question_id')
    
    if not user_answer or not test_question_id:
        return jsonify({"error": "No answer or test question ID provided"}), 400
    
    # Get the test question and answer from the database
    test_question_data = get_response_metadata(test_question_id)
    
    if not test_question_data:
        return jsonify({"error": "Invalid test question ID"}), 400
    
    # Evaluate user's answer using cosine similarity
    evaluation = evaluate_answer(user_answer, test_question_data['test_answer'])
    
    # Store evaluation metadata
    store_knowledge_evaluation_metadata(
        test_question_id,
        user_answer,
        evaluation['knowledge_understood'],
        evaluation['knowledge_confidence']
    )
    
    return jsonify(evaluation), 200

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running!"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)