# QAT System Chatbot

## Overview

The QAT System Chatbot is an advanced conversational AI application designed to assist users by answering questions based on retrieved documents. Utilizing state-of-the-art natural language processing (NLP) techniques and vector storage via FAISS, this chatbot is capable of generating responses and follow-up questions that enhance user understanding. The application is built using LangChain, Hugging Face embeddings, and various document loaders for flexible file handling.

## Features

- **Document Retrieval**: Efficiently loads and splits documents from various formats (PDF, DOCX, TXT).
- **Conversational Memory**: Maintains context through conversation history to provide relevant answers.
- **Dynamic Follow-Up Questions**: Generates follow-up questions based on user queries and chatbot responses.
- **Bullet Point Summaries**: Provides concise bullet points elaborating on answers for better comprehension.
- **Similarity Evaluation**: Evaluates user responses against expected answers using cosine similarity.

## Installation

To get started with the Document Chatbot, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/DrUkachi/qatsystem.git
   cd document-chatbot
   ```

2. **Install required packages**:

   Create a virtual environment (recommended) and install the necessary dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

   Ensure you have the following libraries installed:

   ```plaintext
   langchain
   flask
   streamlit
   langchain_huggingface
   langchain_community
   faiss-cpu  # or faiss-gpu for GPU support
   torch  # Install compatible version based on your system
   ```

3. **Set Up the GROQ API Key**:

   To access the GROQ API, you will need an API key. Follow these steps to set it up:

   - Sign up for a [GROQ](https://console.groq.com/keys) account and obtain your API key.
   - Add the API key to your environment variables:

     - On Linux/MacOS:
       ```bash
       export GROQ_API_KEY='your_api_key_here'
       ```

     - On Windows:
       ```bash
       set GROQ_API_KEY='your_api_key_here'
       ```

   Make sure to replace `your_api_key_here` with your actual GROQ API key.

4. **Download Hugging Face model**:

   The project uses the `sentence-transformers/all-mpnet-base-v2` model for embeddings. Ensure you have access to Hugging Face models by installing the `transformers` library if not already included in `requirements.txt`.

## Usage

### Ensure you have Postgres Installed as the project will be based on the database

1.**Install PostgreSQL**:

   - **On Ubuntu**:
     ```bash
     sudo apt update
     sudo apt install postgresql postgresql-contrib
     ```

   - **On macOS** (using Homebrew):
     ```bash
     brew install postgresql
     ```

   - **On Windows**:
     - Download the installer from the [PostgreSQL official website](https://www.postgresql.org/download/windows/).
     - Follow the installation instructions and configure your PostgreSQL environment.


After installation, start the PostgreSQL service:

   - **On Ubuntu**:
     ```bash
     sudo service postgresql start
     ```

   - **On macOS**:
     ```bash
     brew services start postgresql
     ```

   - **On Windows**:
     Start the PostgreSQL service using the services management console.

3. **Create a PostgreSQL Database**:

   Open the PostgreSQL shell and create a new database for your project:

   ```bash
   sudo -u postgres psql
   CREATE DATABASE document_chatbot_db;
   ```

   You can exit the PostgreSQL shell by typing
   `\q`.

4. **Start the Flask server**:
   ```bash
   python flask_app.py
   ```

   You can decide to use the provide notebook with the `requests` library to test the created endpoints or use the `streamlit` interface
   that came with the project to further test and see the interaction.
   
5. **Start the Streamlit app**
   You can choose the port number you prefer for this just indicate it in the command.
   ```bash
   streamlit run --server.port 8501 QATSystem/streamlit_app.py
   ```
   

## How it works
1. Upload a Document using the upload interface
2. The document will be used to create a docstore or knowledge database - this will take an average of 3-5 mins to load depending on whether you are using a GPU or CPU
3. Once the docstore is created you can then go ahead to send in your query or ask a question related to the document you uploaded.
4. When the query is sent a response is returned with 3 key bullet points and a follow-up question
5. Then user is asked to answer the question and then an evaluation of the response is provided.


## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact:

- **Your Name**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [yourusername](https://github.com/yourusername)

---

Thank you for your interest in the Document Chatbot! We hope you find it useful for your document processing and conversational AI needs.
```

Feel free to replace `yourusername` and `your.email@example.com` with your actual information. Let me know if you need any additional modifications!
