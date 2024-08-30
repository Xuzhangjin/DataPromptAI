# DataPromptAI
DataPromptAI is a Python package designed to streamline data analysis using the power of AI. With DataPromptAI, you can easily load multiple datasets from a local path or URL and interact with them using natural language prompts. The package leverages the OpenAI API to generate Python code snippets based on your input, enabling you to perform complex data operations with minimal effort.

## Key Features
Seamless Data Loading: Load datasets from various sources, whether local files or URLs, and handle large data using chunking.
Natural Language Processing: Describe your data operations in natural language, and DataPromptAI will generate the necessary Python code.
Interactive Execution: Review the generated code and choose whether to execute it directly within your environment.
AI-Powered: Built on top of OpenAI's language models, DataPromptAI brings the power of LLMs to your data analysis workflow.
DataFrame Support: Tailored for operations on pandas DataFrames, making it ideal for data science tasks.

## Data Privacy
DataPromptAI is designed with user privacy in mind. The package ensures that your data remains local and is never uploaded to ChatGPT's servers. All operations, including data processing and prompt-based code generation, are performed on your local machine. This approach safeguards your sensitive information and complies with data privacy regulations, making DataPromptAI a secure choice for working with your data.



## Installation

To install the package, clone the repository and use pip:

```bash
git clone https://github.com/Xuzhangjin/DataPromptAI.git
cd DataPromptAI
pip install .
