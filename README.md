# Article Recommendation System
## Contributors
- [muratergn](https://github.com/muratergn)

## Overview
This project provides article recommendations based on users' interests using Natural Language Processing (NLP) with the Krapivin and Inspec datasets. Users can give feedback on these recommendations and search for articles.

## Project Structure
- **API File**: [`Zakuska_AI/PythonAPI/FastAPI.py`](Zakuska_AI/PythonAPI/FastAPI.py)
- **Machine Learning and AI**: Implemented using Python.
- **Web Development**: Implemented using ASP.NET Core.

## Features
- **Article Recommendations**: Tailored to user interests using NLP.
- **Feedback Mechanism**: Users can provide feedback on the recommendations.
- **Article Search**: Users can search for specific articles.

## Technology Stack
- **Python**: For machine learning and artificial intelligence.
- **ASP.NET Core**: For web development.

## Getting Started

### Prerequisites
- Python 3.9
- ASP.NET Core 7

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/furkankarlidag/Article-Recommendation-System-with-NLP.git
    ```

2. **Navigate to the API directory:**
    ```sh
    cd Zakuska_AI/PythonAPI
    ```

3. **Install the required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the FastAPI server:**
    ```sh
    uvicorn FastAPI:app --reload
    ```

5. **Set up the ASP.NET Core project:**
    Open the `Zakuska_AI.sln` solution file in Visual Studio and run the project.

### Usage

1. **Start the API server:**
    Make sure the FastAPI server is running as described in the installation steps.

2. **Run the ASP.NET Core application:**
    Launch the ASP.NET Core application to access the web interface.

3. **Interact with the system:**
    - Search for articles.
    - Receive personalized recommendations.
    - Provide feedback on the recommendations.
