# Lecture Insights and PDF Chat Application

This project is a **Streamlit-based application** that provides tools for generating insights from lecture content and interacting with PDF documents. It integrates advanced AI capabilities, including Google Gemini, DuckDuckGo search, and YouTube Data API, to enhance the user experience.

## Features

1. **Generate Insights**:

- Upload a PDF or paste text to generate summaries, key takeaways, FAQs, and practice questions.

## Additional Features

- **Markdown Export**: Users can download generated insights in Markdown format for easy sharing and editing.
- **YouTube Video Recommendations**: The application suggests relevant YouTube videos based on the analyzed content.
- **Web Resource Links**: Provides curated links to websites related to the uploaded content for further exploration.
- **Interactive Query Support**: Users can ask follow-up questions about the PDF content for deeper understanding.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install the required dependencies using `pip install -r requirements.txt`.
- Obtain API keys for:
  - **Google Gemini API**
  - **YouTube Data API** (optional)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your API keys:
   Youtube API KEY is optional \
   or you can enter the keys in the sidebar also.

```env
GOOGLE_API_KEY="your_google_gemini_api_key"
YOUTUBE_API_KEY="your_youtube_data_api_key"
```

### Running the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

### Generate Insights

1. Select the "Generate Insights" mode from the sidebar.
2. Upload a PDF or paste text.
3. View summaries, key takeaways, FAQs, and practice questions.
4. Download the insights as a Markdown file.

### Chat with PDF

1. Select the "Chat with PDF" mode from the sidebar.
2. Upload a PDF document.
3. Ask questions about the content in the chat interface.

## Technologies Used

- **Streamlit**: For building the web interface.
- **LangChain**: For AI-powered text processing.
- **FAISS**: For vector-based document retrieval.
- **DuckDuckGo Search API**: For web search results.
- **YouTube Data API**: For related video recommendations.
- **PyMuPDF**: For extracting text from PDFs.

## Live Link

[Lecture Insights Generator](https://gaints-to-genius-matrix-2025.streamlit.app/)
