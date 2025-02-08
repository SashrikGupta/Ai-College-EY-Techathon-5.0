# Import all required libraries for loading documents, processing audio, and handling embeddings
import requests                                       # HTTP requests for API interactions
from langchain_community.document_loaders import PyPDFLoader  # PDF loader from LangChain Community
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitter for chunking text
from langchain_google_genai import GoogleGenerativeAI  # Google Generative AI model via LangChain
import google.generativeai as genai                   # Google Generative AI API library
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
import numpy as np                                    # Numerical computing with arrays
import faiss                                          # Vector indexing and similarity search
import os                                             # Operating system utilities
import assemblyai as aai                              # AssemblyAI for audio transcription
from moviepy.editor import VideoFileClip              # Video processing and audio extraction
from transformers import AutoTokenizer, AutoModel     # Transformers library for NLP models
import torch                                          # PyTorch for deep learning and embeddings
from tqdm import tqdm                                 # Progress bar for loops
from langchain_groq import ChatGroq
import cohere
from dotenv import dotenv_values

config = dotenv_values(".env") 

# Initialize the Cohere client (make sure to replace 'YOUR_API_KEY' with your actual key)
co = cohere.Client(config['COHERE_API_KEY_LIB'])

def embeder(text):
    # Use Cohere's embedding model
    response = co.embed(
        model='embed-english-v2.0',  # Use the embedding model
        texts=[text]                # Send the text in a list
    )
    return response.embeddings[0]  # Return the first (and only) embedding



chunks = []                    # Initialize a global list to store text chunks
index = None                   # Global variable for the FAISS index
API_KEY = config['GEMINI_API_KEY']   # Google API key for LLM services
aai.settings.api_key = config['ASSEMBLY_AI_API_KEY']  # AssemblyAI API key
genai.configure(api_key=API_KEY)  # Configure Google Generative AI with API key


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parallel_embedding(chunks):
    embeddings = []  # List to store embeddings
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(embeder, chunk): chunk for chunk in chunks}  # Submit all tasks
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating embeddings"):
            embeddings.append(future.result())  # Retrieve results
    return embeddings  # Return all embeddings


def pdf_vector_space(path_to_pdf):
    global chunks  # Use global chunks list
    loader = PyPDFLoader(path_to_pdf)  # Load PDF with LangChain's PyPDFLoader
    pages = loader.load()              # Load each page's content
    content = [page.page_content for page in pages]  # Extract text from pages

    # Split text into manageable chunks for embeddings
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", " "],  # Custom separators for splitting text
        chunk_size=1000,                                  # Maximum chunk size
        chunk_overlap=1                                  # Overlap between chunks for context
    )

    for page_content in content:
        chunks.extend(splitter.split_text(page_content))  # Split text and add to chunks

    embeddings = parallel_embedding(chunks)             # Calculate embeddings

    # Remove None entries if there are any
    embeddings = [emb for emb in embeddings if emb is not None]

    vectors = np.array(embeddings)         # Convert list of embeddings to NumPy array
    dim = vectors.shape[1]                 # Get the dimension of vectors for FAISS index
    index = faiss.IndexFlatL2(dim)         # Create FAISS index for L2 distance
    index.add(vectors)                     # Add vectors to FAISS index
    return index                           # Return index

def video_vector_space(path_to_video):
    global chunks
    chunks = []

    # Extract audio from video file
    video = VideoFileClip(path_to_video)             # Load video file
    audio_path = "temp_audio.wav"                    # Temporary file to store extracted audio
    video.audio.write_audiofile(audio_path)          # Write audio from video to temporary file

    try:
        # Transcribe audio using AssemblyAI
        transcriber = aai.Transcriber()              # Initialize transcriber
        transcription = transcriber.transcribe(audio_path)  # Transcribe audio
        text = transcription.text                    # Extract transcribed text

        # Split transcribed text into chunks
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", " "],
            chunk_size=2000,
            chunk_overlap=1
        )
        chunks.extend(splitter.split_text(text))     # Add split text to chunks list

        embeddings = parallel_embedding(chunks)    # Calculate embeddings

        # Remove None embeddings if any
        embeddings = [emb for emb in embeddings if emb is not None]
        vectors = np.array(embeddings)               # Convert embeddings list to array

        dim = vectors.shape[1]                       # Dimension for FAISS index
        index = faiss.IndexFlatL2(dim)               # Initialize FAISS index with L2 distance
        index.add(vectors)                           # Add vectors to index

    finally:
        # Ensure temporary audio file is deleted after processing
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return index                                     # Return FAISS index


def data_give(index, search_query):
    search_vector = embeder(search_query)        # Get embedding of the search query
    search_vector = np.array(search_vector).reshape(1, -1)  # Reshape for FAISS compatibility
    distance, loc = index.search(search_vector, k=3)       # Search top 60 similar vectors

    data_to_be_given = ""
    for i in loc[0]:                             # Retrieve corresponding chunks for query
        data_to_be_given += chunks[i]

    # Generate prompt for LLM
    prompt = "You are given the following data as context : " + data_to_be_given + " now answer the following query on the basis of context: " + search_query
    prompt += " The answer should be highly comprehensive."
    groq_api_key=config['GROQ_API_KEY_LIB']
    llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192")
    return llm.invoke(prompt)                          # Invoke LLM with prompt


import glob  # Import the glob module to find all the PDF files in the specified directory
import os  # Import os to work with file paths
import numpy as np  # Import numpy for handling arrays
import faiss  # Import FAISS for similarity search and clustering

def process_pdf_directory(directory_path):
    """
    This function processes all the PDF files in the specified directory, extracts text,
    splits it into chunks, generates embeddings, and creates a FAISS index for fast similarity search.
    """

    # Get all PDF files in the directory using glob
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    # If no PDF files are found, raise an error
    if not pdf_files:
        raise ValueError("No PDF files found in the specified directory.")

    # Initialize a global list to store text chunks from all PDFs
    global chunks
    chunks = []

    # Initialize a list to store embeddings for each chunk of text
    all_embeddings = []

    # Loop through each PDF file found in the directory
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}")  # Print the current file being processed

        # Load the PDF content using the PyPDFLoader
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()  # Load the pages of the PDF
        content = [page.page_content for page in pages]  # Extract the content of each page

        # Initialize a text splitter with custom parameters
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", " "],  # Split the text at these separators
            chunk_size=2000,  # Define maximum chunk size as 200 characters
            chunk_overlap=1  # Allow 1 character overlap between chunks
        )

        # Loop through each page content and split it into smaller chunks
        for page_content in content:
            chunks.extend(splitter.split_text(page_content))  # Add the resulting chunks to the global list

    # Generate embeddings for each text chunk using a parallel embedding function
    embeddings = parallel_embedding(chunks)

    # Filter out any invalid embeddings (None values)
    embeddings = [emb for emb in embeddings if emb is not None]

    # Ensure that we have generated valid embeddings
    if not embeddings:
        raise ValueError("No valid embeddings generated from PDF files.")

    # Convert the embeddings list into a numpy array for processing
    vectors = np.array(embeddings)

    # Get the dimensionality of the embeddings (the number of features in each vector)
    dim = vectors.shape[1]

    # Initialize a FAISS index to store the vectors for fast nearest neighbor search
    index = faiss.IndexFlatL2(dim)

    # Add the vectors to the FAISS index
    index.add(vectors)

    # Return the FAISS index containing the vectors for fast similarity search
    return index



import glob
import os
from moviepy.editor import VideoFileClip


def process_files_in_directory(directory_path):
    global chunks
    chunks = []

    # Collect all files (PDFs and videos) in the directory using glob
    files = glob.glob(os.path.join(directory_path, "*"))

    all_embeddings = []

    # Iterate over each file in the directory
    for file_path in files:

        # Process PDF files
        if file_path.endswith(".pdf"):
            print(f"Processing PDF: {file_path}")

            # Load the PDF content
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = [page.page_content for page in pages]  # Extract text from each page

            # Initialize the text splitter with custom chunking settings
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", "!", "?", " "],  # Define chunk separators
                chunk_size=2000,  # Max chunk size in characters
                chunk_overlap=1  # Overlap of 1 character between chunks
            )

            # Split content from each page into manageable chunks and add to global chunks list
            for page_content in content:
                chunks.extend(splitter.split_text(page_content))

        # Process video files (MP4, AVI, MOV)
        elif file_path.endswith((".mp4", ".avi", ".mov")):  # Add more formats as needed
            print(f"Processing Video: {file_path}")

            # Extract audio from video file
            audio_path = "temp_audio.wav"  # Temporary audio file to store extracted audio
            video = VideoFileClip(file_path)
            video.audio.write_audiofile(audio_path)

            try:
                # Transcribe the audio using AssemblyAI
                transcriber = aai.Transcriber()  # Assuming a valid AssemblyAI API client is available
                transcription = transcriber.transcribe(audio_path)
                text = transcription.text  # Get the transcribed text

                # Initialize the text splitter for the transcribed text
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", "!", "?", " "],
                    chunk_size=2000,
                    chunk_overlap=1
                )

                # Split the transcribed text into chunks and add to global chunks list
                chunks.extend(splitter.split_text(text))

            finally:
                # Clean up by deleting the temporary audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        else:
            print(f"Unsupported file type: {file_path}")  # Handle unsupported file types

    # Generate embeddings for all text chunks
    embeddings = parallel_embedding(chunks)  # Assuming a predefined embedding function is available
    embeddings = [emb for emb in embeddings if emb is not None]  # Filter out invalid embeddings

    # Ensure that embeddings were successfully generated
    if not embeddings:
        raise ValueError("No valid embeddings generated from files.")

    # Convert the embeddings into a numpy array for processing
    vectors = np.array(embeddings)

    # Get the dimensionality of the embeddings (number of features)
    dim = vectors.shape[1]

    # Initialize a FAISS index for similarity search (L2 distance metric)
    index = faiss.IndexFlatL2(dim)

    # Add the embeddings to the FAISS index
    index.add(vectors)

    # Return the FAISS index containing all the processed vectors for fast similarity search
    return index



def give_vectors(index, search_query):
    search_vector = embeder(search_query)        # Get embedding of the search query
    search_vector = np.array(search_vector).reshape(1, -1)  # Reshape for FAISS compatibility
    distance, loc = index.search(search_vector, k=3)       # Search top 60 similar vectors

    data_to_be_given = ""
    for i in loc[0]:                             # Retrieve corresponding chunks for query
        data_to_be_given += chunks[i]

    return data_to_be_given                          # Invoke LLM with prompt

class Librarian:
    def __init__(self):
        self.index = None                    # Initialize empty index

    def add_material(self, path_to_material):
        self.index = process_files_in_directory(path_to_material)  # Process directory for files

    def query_material(self, prompt):
        return data_give(self.index, prompt)  # Query data from index
    
    def provide_vector(self , query):
        return  give_vectors(self.index , query )

    def remover_material(self):
        # also delete the /content folder if exists
        import shutil 
        if os.path.exists('content'):
            shutil.rmtree('content')
        self.index = None                     # Remove index to clear data