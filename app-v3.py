# app.py - Basic Insights + Youtube References + DuckDuckGo Search
import streamlit as st
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
from langchain_community.tools import DuckDuckGoSearchResults


# Load environment variables
load_dotenv()

# Check for API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

if not google_api_key:
  google_api_key = st.sidebar.text_input(
    "Enter Google Gemini API Key:",
    type="password",
    help="Get your API key from https://console.cloud.google.com/"
  )

if not youtube_api_key:
  youtube_api_key = st.sidebar.text_input(
    "Enter YouTube Data API Key:",
    type="password",
    help="Get your API key from https://console.cloud.google.com/"
  )

if not google_api_key:
  st.warning("⚠️ Please provide your Google Gemini API key to continue!")
  st.stop()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
  model="gemini-2.0-flash-lite",
  temperature=0.2,
  google_api_key=google_api_key
)

search = DuckDuckGoSearchResults(output_format="list")

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

def handleFileChange():
  st.session_state.generated_content = False
  st.session_state.markdown_output = ""

# Streamlit UI
st.title("📚 Lecture Insight Generator")
st.markdown("Upload lecture PDF or paste text to get insights!")

# Input options
input_option = st.radio("Choose input type:", ("PDF Upload", "Text Input"))

input_text = ""
if input_option == "PDF Upload":
  uploaded_file = st.file_uploader("Upload PDF lecture slides", type=["pdf"], on_change=handleFileChange)
  if uploaded_file:
    input_text = extract_text_from_pdf(uploaded_file)
else:
  input_text = st.text_area("Paste lecture text here:", height=200)

# Processing tasks
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
      st.write("---")
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

