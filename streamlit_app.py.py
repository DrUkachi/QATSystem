import streamlit as st
import requests
import io
from db import init_db

def handle_response(response, success_message, error_message):
    """Handles API response for uploads, queries, and evaluations."""
    if response.status_code == 200:
        st.success(success_message)
        return response.json()
    else:
        st.error(error_message)
        return None

def main():
    # Initialize the database only once
    if 'db_initialized' not in st.session_state:
        init_db()
        st.session_state.db_initialized = True

    # Initialize session state variables
    if 'document_uploaded' not in st.session_state:
        st.session_state.document_uploaded = False
    if 'query_submitted' not in st.session_state:
        st.session_state.query_submitted = False
    if 'evaluation_completed' not in st.session_state:
        st.session_state.evaluation_completed = False

    st.title("Document Q&A and Understanding Evaluation")

    # Upload document
    uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf"], key="file_uploader")

    if uploaded_file is not None and not st.session_state.document_uploaded:
        bytes_data = uploaded_file.getvalue()
        files = {"file": ("document.pdf", io.BytesIO(bytes_data), "application/pdf")}

        response = requests.post("http://localhost:8080/upload", files=files)
        result = handle_response(response, "Document successfully uploaded and knowledge base created.", "Failed to upload document. Please try again.")
        
        if result:  
            st.session_state.document_uploaded = True  

    if st.session_state.document_uploaded:  
        query = st.text_input("Enter your question about the document:")

        if st.button("Submit Query"):
            if query:
                response = requests.post("http://localhost:8080/query", json={"query": query})
                result = handle_response(response, "Query processed successfully.", "Failed to process query. Please try again.")

                if result:
                    # Display answer and bullet points
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    st.subheader("Key Points:")
                    st.write(result['bullet_points'])
                    st.subheader("Test Question:")
                    st.write(result["test_question"])

                    # Store the test_question_id in session state
                    st.session_state.test_question_id = result["test_question_id"]
                    st.session_state.query_submitted = True  

        # User answer input
        user_answer = st.text_input("Your answer to the test question:")

        if st.button("Submit Answer") and user_answer:
            eval_response = requests.post("http://localhost:8080/evaluate", 
                                                      json={"answer": user_answer, 
                                                            "test_question_id": st.session_state.test_question_id})

            evaluation_result = handle_response(eval_response, "Answer submitted for evaluation.", "Failed to evaluate answer. Please try again.")

            if evaluation_result:  
                st.subheader("Evaluation:")
                st.write(f"Understanding: {'Yes' if evaluation_result['knowledge_understood'] else 'No'}")
                st.write(f"Confidence: {evaluation_result['knowledge_confidence']}%")
                st.session_state.evaluation_completed = True
     
        else:
            st.warning("Please enter a question before submitting.")
    else:
        st.warning("Please enter a question before submitting.")

if __name__ == "__main__":
    main()
