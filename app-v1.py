# app.py - Basic Insights Generator
import streamlit as st
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check for API key in .env or ask in sidebar
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    google_api_key = st.sidebar.text_input(
        "Enter Google Gemini API Key:",
        type="password",
        help="Get your API key from https://console.cloud.google.com/"
    )

if not google_api_key:
    st.warning("⚠️ Please provide your Google Gemini API key to continue!")
    st.stop()

# Initialize Gemini model with the provided key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=google_api_key
)

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def generate_streaming_content(text, prompt_template):
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    for chunk in chain.stream({"text": text}):
        yield chunk

# Streamlit UI
st.title("📚 Lecture Insight Generator")
st.markdown("Upload lecture PDF or paste text to get insights!")

# Input options
input_option = st.radio("Choose input type:", ("PDF Upload", "Text Input"))


def handleInputChange():
    st.session_state.generated_content = False

input_text = ""
if input_option == "PDF Upload":
    uploaded_file = st.file_uploader("Upload PDF lecture slides", type=["pdf"], on_change=handleInputChange) 
    if uploaded_file:
        input_text = extract_text_from_pdf(uploaded_file)
else:
    input_text = st.text_area("Paste lecture text here:", height=200)

if 'generated_content' not in st.session_state:
    st.session_state.generated_content = False
if 'markdown_output' not in st.session_state:
    st.session_state.markdown_output = ""

if input_text:

  # Define processing tasks
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
          "prompt": """Create 5 practice exam questions with answers based on this content.\nFormat each as:\n\nQuestion: [question]\n\nAnswer: [answer]\n\n---\n\nContent: {text}"""
      }
  ]

  markdown_output = ""

  if st.session_state.generated_content:
    st.markdown(st.session_state.markdown_output)
  
  if not st.session_state.generated_content:
    for task in tasks:
      st.subheader(task["title"])
      placeholder = st.empty()
      st.write("---")
      full_response = ""
      
      # Stream the response
      for chunk in generate_streaming_content(input_text, task["prompt"]):
        full_response += chunk
        placeholder.markdown(full_response)

      markdown_output += f"## {task['title']}\n{full_response}\n\n"
    st.session_state.generated_content = True
    st.session_state.markdown_output = markdown_output

  
  download_button =  st.download_button(
  label="Download Insights as Markdown",
  data=markdown_output.encode("utf-8"),
  file_name="lecture_insights.md",
  mime="text/markdown"
  )