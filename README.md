# LangChain Documentation Helper

This project is a **LangChain Documentation Helper** that provides users with an AI-powered chatbot capable of answering queries related to LangChain documentation. The application utilizes advanced natural language processing (NLP) techniques to enable efficient documentation exploration and query handling. While primarily designed for LangChain, the framework can be extended to support other documentation domains.

## Features

- **Query-Based Documentation Support**: Responds to user queries with relevant answers extracted from the LangChain documentation.
- **Embedding Model**: Uses `text-embedding-3-small` for generating embeddings and storing them in a Pinecone index.
- **Cosine Similarity Evaluation**: Measures the relevance of responses with an average cosine similarity score of 0.69 across ten queries.
- **Extensibility**: Adaptable to other technical documentation datasets.

## Project Structure

```
LangChain_Doc_Helper/
├── langchain-docs      # Contains HTML documentation files, Preprocessed and chunked data
├── modules/
│   ├── ingestion.py    # Loads documentation files to pinecone after preprocessing
│   ├── core.py         # Creation of llm function to instantiate the chatbot
│   ├── main.py         # Streamlit Deployment
├── README.md           # Project overview
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mohsina-bilal/LangChain_Doc_Helper.git
   cd LangChain_Doc_Helper
   ```

2. Configure environment variables by creating a `.env` file:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   ```

## Usage

1. **Prepare the Data:** Place your LangChain documentation files in the `data/raw_data/` folder.

2. **Preprocess and Generate Embeddings:**

   Run the following script to preprocess the data and generate embeddings:

   ```bash
   python ingestion.py
   ```

3. **Start the Chatbot:**

   Create the function initializing the chatbot based on OpenAI API:

   ```bash
   python core.py
   ```

4. **Chatbot User Interface:**

   Create a streamlit interface for the chatbot:

   ```bash
   python main.py
   ```

## Results

- Achieved an average cosine similarity score of **0.75** over ten queries.
- Demonstrated effective query handling for fact-based questions.
- Identified areas for improvement in complex or context-dependent queries.

## Future Work

- Expand support to other technical documentation domains.
- Fine-tune the embedding model for improved response accuracy.
- Incorporate advanced evaluation metrics for richer insights.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For inquiries or suggestions, please reach out to:

- **Name**: Mohsina Bilal
- **GitHub**: [mohsina-bilal](https://github.com/mohsina-bilal)
