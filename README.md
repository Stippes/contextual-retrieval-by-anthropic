# Contextual Retrieval with Llama-Index

This project demonstrates contextual retrieval as described by Anthropic. Documents are indexed in a local [ChromaDB](https://www.trychroma.com/) store and served through FastAPI and Streamlit.

## Quick start

1. **Clone the repository**
   ```bash
   git clone https://github.com/RionDsilvaCS/contextual-retrieval-by-anthropic.git
   cd contextual-retrieval-by-anthropic
   ```

2. **Run the Apache Tika server**
   ```bash
   docker run -p 9998:9998 apache/tika:3.2.0.0-full
   ```

   Or currently on Windows, in the cmd:
   '''
   cd desktop
   java -jar tika-server-standard-3.2.2.jar
   '''


3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   This installs [Unstructured](https://github.com/Unstructured-IO/unstructured) and other required packages.

4. **Prepare your documents**
   ```bash
   mkdir data
    # add .pdf, .docx, .pptx, .xlsx or .txt files inside this folder
   ```

5. **Configure environment variables**
   Copy the template and then edit it:
   ```bash
   cp .env.example .env
   ```
   Fill in the values. Important variables are:
   - `BASE_PATH` – root directory for data and database files
   - `DATA_DIR` – location of your documents relative to `BASE_PATH`
   - `SAVE_DIR` – folder where the database is stored
   - `COLLECTION_NAME` – name of the ChromaDB collection
   - `TIKA_URL` – URL of the Tika server (default `http://localhost:9998`)
   - `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL` – credentials for OpenAI. `OPENAI_MODEL` sets the chat model name used in requests.
   - `AZURE_API_KEY`, `AZURE_ENDPOINT`, `AZURE_DEPLOYMENT_NAME`, `AZURE_API_VERSION` – credentials for Azure OpenAI (optional when using OpenAI)

   You can run the app with only an OpenAI API key by providing `OPENAI_API_KEY` and the model names while leaving the Azure variables empty. Either Azure or OpenAI credentials must be supplied.

6. **Create the vector store**
   ```bash
   python create_save_db.py
   ```
   This command ingests any files placed in `DATA_DIR`, including PowerPoint and Excel documents. See [`tests/test_ingestion_office.py`](tests/test_ingestion_office.py) for an example of validating `.pptx` and `.xlsx` ingestion.

7. **Start services**
   ```bash
   ollama serve            # run in a separate terminal
   python app.py           # start the FastAPI server
   streamlit run main.py   # launch the web UI
   ```

Once running, open the Streamlit page and begin asking questions about your documents.

## API endpoints

- `POST /rag-chat` – submit a question and receive an answer with document sources.
- `POST /upload` – upload a new document. The file is saved to `DATA_DIR` and the vector store is rebuilt automatically.
