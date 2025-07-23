# Contextual Retrieval with Llama-Index

This project demonstrates contextual retrieval as described by Anthropic. Documents are indexed in a local [ChromaDB](https://www.trychroma.com/) store and served through FastAPI and Streamlit.

## Quick start

1. **Clone the repository**
   ```bash
   git clone https://github.com/RionDsilvaCS/contextual-retrieval-by-anthropic.git
   cd contextual-retrieval-by-anthropic
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your documents**
   ```bash
   mkdir data
   # add .pdf, .docx or .txt files inside this folder
   ```

4. **Configure environment variables**
   Copy `.env.example` to `.env` and fill in the values.
   Important variables are:
   - `BASE_PATH` – root directory for data and database files
   - `DATA_DIR` – location of your documents relative to `BASE_PATH`
   - `SAVE_DIR` – folder where the database is stored
   - `COLLECTION_NAME` – name of the ChromaDB collection
   - `API_URL` – URL of the FastAPI endpoint
   - `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL` – credentials for OpenAI
   - `AZURE_API_KEY`, `AZURE_ENDPOINT`, `AZURE_DEPLOYMENT_NAME`, `AZURE_API_VERSION` – credentials for Azure OpenAI (optional when using OpenAI)

   You can run the app with only an OpenAI API key by providing `OPENAI_API_KEY` and the model names while leaving the Azure variables empty. Either Azure or OpenAI credentials must be supplied.

5. **Create the vector store**
   ```bash
   python create_save_db.py
   ```

6. **Start services**
   ```bash
   ollama serve            # run in a separate terminal
   python app.py           # start the FastAPI server
   streamlit run main.py   # launch the web UI
   ```

Once running, open the Streamlit page and begin asking questions about your documents.
