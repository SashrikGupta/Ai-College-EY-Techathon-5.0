import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
import os
import assemblyai as aai
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from langchain_groq import ChatGroq
import cohere
from dotenv import dotenv_values
import glob
import shutil

# Load configuration from .env file
config = dotenv_values(".env") 

# Initialize the Cohere client
co = cohere.Client(config['COHERE_API_KEY_LIB'])

def embeder(text):
    """
    Get an embedding for the provided text using Cohere.
    """
    response = co.embed(
        model='embed-english-v2.0',
        texts=[text]
    )
    return response.embeddings[0]

def parallel_embedding(chunks):
    """
    Calculate embeddings in parallel for a list of text chunks.
    """
    embeddings = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(embeder, chunk): chunk for chunk in chunks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating embeddings"):
            embeddings.append(future.result())
    return embeddings

def pdf_vector_space(path_to_pdf):
    """
    Process a PDF file to build a FAISS index and return the associated text chunks.
    """
    local_chunks = []
    loader = PyPDFLoader(path_to_pdf)
    pages = loader.load()
    content = [page.page_content for page in pages]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", " "],
        chunk_size=1000,
        chunk_overlap=1
    )
    for page_content in content:
        local_chunks.extend(splitter.split_text(page_content))
    embeddings = parallel_embedding(local_chunks)
    embeddings = [emb for emb in embeddings if emb is not None]
    vectors = np.array(embeddings)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, local_chunks

def video_vector_space(path_to_video):
    """
    Process a video file (by extracting and transcribing audio) to build a FAISS index.
    """
    local_chunks = []
    audio_path = "temp_audio.wav"
    video = VideoFileClip(path_to_video)
    video.audio.write_audiofile(audio_path)
    try:
        transcriber = aai.Transcriber()
        transcription = transcriber.transcribe(audio_path)
        text = transcription.text
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", " "],
            chunk_size=2000,
            chunk_overlap=1
        )
        local_chunks.extend(splitter.split_text(text))
        embeddings = parallel_embedding(local_chunks)
        embeddings = [emb for emb in embeddings if emb is not None]
        vectors = np.array(embeddings)
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return index, local_chunks

def data_give(index, search_query, chunks):
    """
    Given a FAISS index, search query, and corresponding text chunks, find the most relevant context,
    then generate and return a comprehensive answer using LLM (via ChatGroq).
    """
    search_vector = embeder(search_query)
    search_vector = np.array(search_vector).reshape(1, -1)
    distance, loc = index.search(search_vector, k=3)
    data_to_be_given = ""
    for i in loc[0]:
        data_to_be_given += chunks[i]
    prompt = (
        "You are given the following data as context: " + data_to_be_given +
        " now answer the following query on the basis of context: " + search_query +
        " The answer should be highly comprehensive."
    )
    groq_api_key = config['GROQ_API_KEY_LIB']
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    return llm.invoke(prompt)

def process_pdf_directory(directory_path):
    """
    Process all PDFs in the directory to create a FAISS index and corresponding text chunks.
    """
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    if not pdf_files:
        raise ValueError("No PDF files found in the specified directory.")
    local_chunks = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        content = [page.page_content for page in pages]
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", " "],
            chunk_size=2000,
            chunk_overlap=1
        )
        for page_content in content:
            local_chunks.extend(splitter.split_text(page_content))
    embeddings = parallel_embedding(local_chunks)
    embeddings = [emb for emb in embeddings if emb is not None]
    if not embeddings:
        raise ValueError("No valid embeddings generated from PDF files.")
    vectors = np.array(embeddings)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, local_chunks

def process_files_in_directory(directory_path):
    """
    Process all supported files (PDFs and videos) in the directory, build embeddings,
    and create a FAISS index for fast similarity search.
    """
    local_chunks = []
    files = glob.glob(os.path.join(directory_path, "*"))
    for file_path in files:
        if file_path.endswith(".pdf"):
            print(f"Processing PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = [page.page_content for page in pages]
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", "!", "?", " "],
                chunk_size=2000,
                chunk_overlap=1
            )
            for page_content in content:
                local_chunks.extend(splitter.split_text(page_content))
        elif file_path.endswith((".mp4", ".avi", ".mov")):
            print(f"Processing Video: {file_path}")
            audio_path = "temp_audio.wav"
            video = VideoFileClip(file_path)
            video.audio.write_audiofile(audio_path)
            try:
                transcriber = aai.Transcriber()
                transcription = transcriber.transcribe(audio_path)
                text = transcription.text
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", "!", "?", " "],
                    chunk_size=2000,
                    chunk_overlap=1
                )
                local_chunks.extend(splitter.split_text(text))
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        else:
            print(f"Unsupported file type: {file_path}")
    embeddings = parallel_embedding(local_chunks)
    embeddings = [emb for emb in embeddings if emb is not None]
    if not embeddings:
        raise ValueError("No valid embeddings generated from files.")
    vectors = np.array(embeddings)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, local_chunks

def give_vectors(index, search_query, chunks):
    """
    Retrieve the aggregated text chunks corresponding to the top similar vectors for a query.
    """
    search_vector = embeder(search_query)
    search_vector = np.array(search_vector).reshape(1, -1)
    distance, loc = index.search(search_vector, k=3)
    data_to_be_given = ""
    for i in loc[0]:
        data_to_be_given += chunks[i]
    return data_to_be_given

# Final class remains in the same format as before
class Librarian:
    def __init__(self):
        self.index = None        # FAISS index
        self.chunks = []         # Associated text chunks

    def add_material(self, path_to_material):
        """
        Process all files in the given directory and build the FAISS index and chunks.
        """
        self.index, self.chunks = process_files_in_directory(path_to_material)

    def query_material(self, prompt):
        """
        Query the material using the provided prompt. Returns an answer generated by the LLM.
        """
        if self.index is None:
            raise ValueError("No material has been added.")
        return data_give(self.index, prompt, self.chunks)

    def provide_vector(self, query):
        """
        Return the concatenated text chunks corresponding to the query.
        """
        if self.index is None:
            raise ValueError("No material has been added.")
        return give_vectors(self.index, query, self.chunks)

    def remover_material(self):
        """
        Remove indexed material and delete the 'content' folder if it exists.
        """
        if os.path.exists('content'):
            shutil.rmtree('content')
        self.index = None
        self.chunks = []

    def embedder(self, text):
        return embeder(text)
