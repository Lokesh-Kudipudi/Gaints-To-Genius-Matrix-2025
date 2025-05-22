# app.py - Basic Insights + YouTube Search + DuckDuckGo Search + Chat with PDF
import streamlit as st
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from googleapiclient.discovery import build
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check for API keys
# google_api_key = os.getenv("GOOGLE_API_KEY")
# youtube_api_key = os.getenv("YOUTUBE_API_KEY")

google_api_key = st.secrets["GOOGLE_API_KEY"]
youtube_api_key = st.secrets["YOUTUBE_API_KEY"]

# Check if Google Gemini API key is provided, if not, prompt the user to input it
if not google_api_key:
  google_api_key = st.sidebar.text_input(
    "Enter Google Gemini API Key:",
    type="password",
    help="Get your API key from https://console.cloud.google.com/"
  )

# Check if YouTube Data API key is provided, if not, prompt the user to input it
if not youtube_api_key:
  youtube_api_key = st.sidebar.text_input(
    "Enter YouTube Data API Key:",
    type="password",
    help="Get your API key from https://console.cloud.google.com/"
  )

# If Google Gemini API key is still not provided, display a warning and stop execution
if not google_api_key:
  st.warning("⚠️ Please provide your Google Gemini API key to continue!")
  st.stop()

# Cache the loading of resources to optimize performance
@st.cache_resource
def load_resources():
  # Initialize the language model (LLM) and embeddings using the Google Gemini API key
  llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2, google_api_key=google_api_key)
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
  
  # Initialize the DuckDuckGo search tool
  search = DuckDuckGoSearchResults(output_format="list")
  
  # Return the loaded resources
  return llm, embeddings, search

# Load the LLM, embeddings, and search tool
llm, embeddings, search = load_resources()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
  text = ""
  # Open the PDF file using PyMuPDF (fitz) and extract text from each page
  with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
    for page in doc:
      text += page.get_text()
  return text

# Function to generate streaming content based on input text and a prompt template
def generate_streaming_content(text, prompt_template):
  # Create a prompt using the provided template
  prompt = PromptTemplate.from_template(prompt_template)
  
  # Create a processing chain with the prompt, LLM, and output parser
  chain = prompt | llm | StrOutputParser()
  
  # Stream the generated content chunk by chunk
  for chunk in chain.stream({"text": text}):
    yield chunk

# Function to handle file change events (e.g., when a new file is uploaded)
def handleFileChange():
  # Reset session state variables to regenerate content
  st.session_state.generated_content = False
  st.session_state.markdown_output = ""

# Common function to process PDF
def process_pdf(file):
  text = extract_text_from_pdf(file)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_text(text)
  vectorstore = FAISS.from_texts(splits, embeddings)
  return vectorstore

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode:", ["Generate Insights", "Chat with PDF"])

# Generate Insights Mode
if app_mode == "Generate Insights":
  st.title("📚 Lecture Insight Generator")
  st.markdown("Upload lecture PDF or paste text to get insights!")

  # Input options
  input_option = st.radio("Choose input type:", ("PDF Upload", "Text Input"), key="insights_input_option")

  input_text = ""
  if input_option == "PDF Upload":
    uploaded_file = st.file_uploader("Upload PDF lecture slides", type=["pdf"], 
                                      key="insights_pdf_uploader", on_change=handleFileChange)
    if uploaded_file:
      input_text = extract_text_from_pdf(uploaded_file)
  else:
    input_text = st.text_area("Paste lecture text here:", height=200, key="insights_text_area")

  # Processing tasks (same as before)
  tasks = [
    {
      "title": "📝 Summary",
      "prompt": """Generate a concise summary of the following lecture content.\nFocus on main concepts and key information. Keep it under 200 words.\n\nContent: {text}"""
    },
    {
      "title": "🔑 Key Takeaways",
      "prompt": """Extract 5-7 key bullet points from this lecture content.\nMake them clear and concise. Use bullet points.\n\nContent: {text}"""
    },
    {
      "title": "❓ FAQ",
      "prompt": """Generate 5-7 important questions students might ask about this content along with their answers.\nFormat as Question:\n\n[question]\n\nAnswer: [answer]\n\n---\n\nContent: {text}"""
    },
    {
      "title": "📝 Practice Questions",
      "prompt": """Create 5-7 practice exam questions with answers based on this content.\nFormat each as:\n\nQuestion: [question]\n\nAnswer: [answer]\n\n---\n\nContent: {text}"""
    }
  ]
  
  if 'generated_content' not in st.session_state:
    st.session_state.generated_content = False
  if 'markdown_output' not in st.session_state:
    st.session_state.markdown_output = ""
  
  if input_text:
  
    if st.session_state.generated_content:
      st.markdown(st.session_state.markdown_output)
    
    if not st.session_state.generated_content:
      markdown_output = "" 
      st.session_state.generated_content = True
      
      # Process main tasks
      for task in tasks:
        st.subheader(task["title"])
        placeholder = st.empty()
        st.write("---")
        full_response = ""
       
        for chunk in generate_streaming_content(input_text, task["prompt"]):
          full_response += chunk
          placeholder.markdown(full_response)
        markdown_output += f"## {task['title']}\n{full_response}\n\n"
        
      # Check if YouTube API key is provided
      if youtube_api_key:
        try:
          # Placeholder for displaying YouTube video results
          yt_placeholder = st.empty()
            
          # Generate a search query for YouTube videos based on the input content
          query_prompt = PromptTemplate.from_template(
            """Generate a concise search query for YouTube videos based on this content.
            Focus on main topics and key terms. Return only the query text, nothing else.
            Content: {text}"""
          )
        
          # Use the language model to create the search query
          search_query = query_prompt | llm | StrOutputParser()
          query = search_query.invoke({"text": input_text})
            
          # Use the YouTube Data API to search for videos
          youtube = build('youtube', 'v3', developerKey=youtube_api_key)
          search_response = youtube.search().list(
            q=query,  # Search query generated by the model
            part='snippet',  # Retrieve video details
            maxResults=5,  # Limit the number of results
            type='video',  # Search only for videos
            videoDuration='medium'  # Exclude short videos
          ).execute()
        
          # Display the search results as a list of video links
          yt_content = "--- \n## 🎥 Related YouTube Videos\n"
          for item in search_response.get('items', []):
            video_id = item['id']['videoId']  # Extract video ID
            title = item['snippet']['title']  # Extract video title
            url = f"https://www.youtube.com/watch?v={video_id}"  # Construct video URL
            yt_content += f"- [{title}]({url})\n"  # Add video link to the content
            yt_placeholder.markdown(yt_content)  # Update the placeholder with the content
        
          # Append the YouTube video links to the markdown output
          markdown_output += yt_content
        except Exception as e:
          # Display an error message if the YouTube search fails
          st.error(f"Error searching YouTube: {str(e)}")
  
      duckduckgoSearchQuery_prompt = PromptTemplate.from_template(
        """Generate a concise google search query based on this content.
        Focus on main topics and key terms it should be related to the lecture content.
        But not the Lecture author, origin, or any other irrelevant information. 
        Return only the query text, nothing else.
        Content: {text}"""
      )
      search_query_duckduckgo = duckduckgoSearchQuery_prompt | llm | StrOutputParser()
      query_duckduckgo = search_query_duckduckgo.invoke({"text": input_text})
      search_results = search.invoke(query_duckduckgo, output_format="list")
      
      st.write("## 🌐 Related Web Search Results")
  
      markdown_output += f"---\n## 🌐 Related Web Search Results\n"
      for result in search_results:
        st.markdown(f"- [{result['title']}]({result['link']})")
        markdown_output += f"- [{result['title']}]({result['link']})\n"
      
      st.session_state.markdown_output = markdown_output
  
    # Download button
    st.download_button(
      label="Download Insights as Markdown",
      data=st.session_state.markdown_output.encode("utf-8"),
      file_name="lecture_insights.md",
      mime="text/markdown"
    )  

# Chat with PDF Mode
elif app_mode == "Chat with PDF":
    st.title("💬 Chat with PDF")
    
    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # PDF Uploader for chat mode
    uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"], key="chat_pdf_uploader")
    
    if uploaded_file and (st.session_state.vectorstore is None or 
                        "uploaded_file" not in st.session_state or 
                        st.session_state.uploaded_file != uploaded_file):
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.session_state.uploaded_file = uploaded_file
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask questions about the PDF..."):
        if not st.session_state.vectorstore:
            st.error("Please upload a PDF document first!")
            st.stop()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare RAG chain
        retriever = st.session_state.vectorstore.as_retriever()
        template = """Answer the question based only on the following context:
        Context: {context}

        Question: {question}
        """
        prompt_rag = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_rag
            | llm
            | StrOutputParser()
        )

        # Generate response
        with st.chat_message("assistant"):
            response = chain.invoke(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})