# Julien's GPT4ALL Interface

## Overview

This project integrates embeddings with an open-source Large Language Model (LLM) to answer questions about Julien GODFROY. It uses the `langchain` library in Python to handle embeddings and querying against a set of documents (e.g., CV of Julien GODFROY). The project includes a Streamlit web interface for easy interaction.

## Screenshots

![Application Screenshot1](/Screenshots/Q2_hiring.png)
![Application Screenshot2](/Screenshots/Q2_teamwork.png)

## Features

- Query processing using `langchain` library.
- Integration with GPT4All for answering questions.
- Streamlit interface for user interaction.
- Specialized in answering queries related to Julien GODFROY's professional profile.

## Installation

Before running the application, ensure you have Python installed on your system. Then, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/jugodfroy/GPT4ALL-langchain-embeddings.git
   ```
2. Navigate to the cloned repository:
   ```bash
   cd GPT4ALL-langchain-embeddings
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the application, run the following command in the terminal:

```bash
streamlit run main.py
```

This will open the Streamlit interface in your default web browser. Here, you can type your questions about Julien GODFROY in the provided text area and submit them to receive answers.

## Contributing

Contributions to improve this project are welcome. Please ensure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contact

- Julien GODFROY - [GitHub](https://github.com/jugodfroy)
- Project Link: https://github.com/jugodfroy/GPT4ALL-langchain-embeddings

## Acknowledgements

- [LangChain](https://github.com/nickloewen/langchain)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
