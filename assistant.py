from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import json
import pickle
import numpy as np
from pprint import pprint
import torch

# Use GPU if available, else default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embed_dims = len(embeddings.embed_query("test"))

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    verbose=True,
    streaming=True
)

def save_to_pickle(convstore, docstore, filename='store_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({'convstore': convstore, 'docstore': docstore}, f)
    print(f"Saved convstore and docstore to {filename}.")

def load_from_pickle(filename='store_data.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded convstore and docstore from {filename}.")
    return data['convstore'], data['docstore']

def RPrint(preface=""):
    """Returns a runnable that prints its input with an optional preface."""
    def print_and_return(x):
        if preface:
            print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(print_and_return)

def docs2str(docs, title="Document"):
    """Convert document chunks into a formatted string."""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        out_str += f"[Quote from {doc_name}] {doc.page_content}\n"
    return out_str

def load_and_split_documents(file_paths):
    """Load and split documents from given file paths."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " "]
    )
    
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        loaded_docs = loader.load()
        for doc in loaded_docs:
            content = doc.page_content
            if "References" in content:
                content = content[:content.index("References")]
                doc.page_content = content
            # Split the document into chunks and filter short chunks
            doc_chunks = text_splitter.split_documents([doc])
            doc_chunks = [chunk for chunk in doc_chunks if len(chunk.page_content) > 200]
            documents.extend(doc_chunks)
    
    return documents

def create_summary(docs):
    """Create a summary string and metadata for the documents."""
    doc_string = "Available Documents:"
    doc_metadata = []
    
    for doc in docs:
        metadata = getattr(doc, 'metadata', {})
        doc_string += f"\n - {metadata.get('Title', 'Unknown Title')}"
        doc_metadata.append(str(metadata))
    
    return doc_string, [doc_string] + doc_metadata

def default_FAISS():
    """Create an empty FAISS vectorstore."""
    return FAISS(
        embedding_function=embeddings,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    """Aggregate multiple FAISS vector stores into one."""
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

def create_docstore(extra_chunks, docs):
    """Create and return a document store."""
    print("Constructing Vector Stores")
    vecstores = [FAISS.from_texts(extra_chunks, embeddings)]
    vecstores += [FAISS.from_documents(docs, embeddings)]

    docstore = aggregate_vstores(vecstores)
    print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")
    return docstore

def save_memory_and_get_output(d, vstore):
    """Save input/output to conversation store and return output."""
    vstore.add_texts([f"User previously responded with {d.get('input')}",
                      f"Agent previously responded with {d.get('output')}"])
    return d.get('output')

def initialize_chat_prompt():
    """Initialize and return the chat prompt template."""
    return ChatPromptTemplate.from_messages([(
        "system",
        """You are a document chatbot. Your task is to assist the user by answering questions based on the retrieved documents.
        User asked: {input}
        We have retrieved the following potentially useful information from the documents: 
        Document Retrieval:{context}
        Please provide an answer based only on the retrieved information. Ensure your response is clear, conversational, and directly addresses the user's query."""
    ),
    ('user', '{input}')])

def create_retrieval_chain(convstore, docstore):
    """Create and return the retrieval chain."""
    long_reorder = RunnableLambda(LongContextReorder().transform_documents)
    return (
        {'input': (lambda x: x)}
        | RunnableAssign({'history': itemgetter(
            'input') | convstore.as_retriever() | long_reorder | docs2str})
        | RunnableAssign({'context': itemgetter(
            'input') | docstore.as_retriever() | long_reorder | docs2str})
    )

def generate_follow_up_question(message, final_answer):
    """Generate a follow-up question based on the context and previous answers."""
    follow_up_prompt = f"""
    You just provided the following answer: "{final_answer}"
    Based on the user's previous question: "{message}",
    generate a follow-up question to test the user's understanding 
    based on the answer you provided.
    Don't start with "Sure, here\'s a follow-up question to test the user\'s understanding:\n\n"
    Just state the question.
    """
    
    # Use the LLM to generate the follow-up question
    follow_up_question = llm.invoke(follow_up_prompt)
    return follow_up_question.content

def chat_gen(message, convstore, docstore, follow_up=True):
    """Generate a chat response, bullet point and a follow-up question based on the user's message."""
    
    bullet_points = ""

    chat_prompt = initialize_chat_prompt()
    stream_chain = chat_prompt | llm | StrOutputParser()
    retrieval_chain = create_retrieval_chain(convstore, docstore)

    # Perform the retrieval based on the input message and history
    retrieval = retrieval_chain.invoke(message)

    # Run the stream_chain to get the final answer
    final_answer = stream_chain.invoke(retrieval)

    # Save the chat exchange to the conversation memory buffer
    save_memory_and_get_output({'input': message, 'output': final_answer}, convstore)

    if follow_up:
        # Generate a follow-up question based on retrieved context and the last answer
        follow_up_question = generate_follow_up_question(message, final_answer)
        # Generate Bullet Points
        bullet_points = generate_bullet_points(final_answer, retrieval, message)
        
        return final_answer, bullet_points, follow_up_question
    return final_answer

    

def generate_bullet_points(final_answer, retrieval, message):
    """Generate bullet points elaborating on the provided answer."""
    context = retrieval['context']

    bullet_point_prompt = f"""
    You just provided the following answer: "{final_answer}"
    based on the question {message} provided by the user and the context retrieved from the document: "{context}".
    
    Now, please provide 3 bullet points to elaborate on the answer given, formatted as a list:
    - Bullet Point 1: 
    - Bullet Point 2: 
    - Bullet Point 3:
    """
    
    bullet_points = llm.invoke(bullet_point_prompt)
    
    return bullet_points.content

def generate_follow_up_answer(follow_up_question, convstore, docstore):
    """Generate a follow-up answer for the provided question."""
    follow_up_answer = chat_gen(follow_up_question, convstore, docstore, follow_up=False)
    return follow_up_answer

def evaluate_answer(user_answer: str, test_answer: str) -> dict:
    """Evaluate the user's answer against the test answer using cosine similarity."""
    user_embedding = embeddings.embed_query(user_answer)
    test_embedding = embeddings.embed_query(test_answer)
    
    # Reshape the embeddings to 2D for cosine similarity calculation
    user_embedding = np.array(user_embedding).reshape(1, -1)
    test_embedding = np.array(test_embedding).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(user_embedding, test_embedding)[0][0]
    
    # Determine if knowledge is understood based on the similarity score
    knowledge_understood = bool(similarity > 0.6)  # Adjust threshold as needed
    knowledge_confidence = int(similarity * 100)  # Convert to a percentage

    return {
        "knowledge_understood": knowledge_understood,
        "knowledge_confidence": knowledge_confidence
    }