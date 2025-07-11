# Contextual Retrieval with Llama-Index (Anthropic)
![banner_img](./img/dataflow.png)

This repository provides an implementation of [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval), a novel approach that enhances the performance of retrieval systems by incorporating chunk-specific explanatory context. By prepending contextual information to each chunk before embedding and indexing, this method improves the relevance and accuracy of retrieved results.

### Key Technologies

- **Llama-Index:** A powerful framework for building semantic search applications.
- **Ollama:** A local LLMs serving solution, using the gemma2:2b model.
- **Streamlit:** A Python framework for building interactive web applications.
- **FastAPI:** A high-performance API framework for building web applications.
- **ChromaDB:** A vector database for efficient storage and retrieval of high-dimensional embeddings.

### Examples

Example 1:
![prompt_1_f](./img/prompt_1_f.png)
![prompt_1_b](./img/prompt_1_s.png)

Example 2:
![prompt_3_f](./img/prompt_2_f.png)
![prompt_3_b](./img/prompt_2_s.png)

### Setup

Clone the GitHub repo
```shell
git clone https://github.com/RionDsilvaCS/contextual-retrieval-by-anthropic.git
```
```shell
cd contextual-retrieval-by-anthropic
```

Create Python `env` and run `requirements.txt` file
```shell 
pip install -r requirements.txt
```

Create a directory `data` and add your documents here. Supported file types are
`.pdf`, `.docx`, and `.txt`.
```shell
mkdir data
```

Copy `.env.template` to `.env` and update the values. `BASE_PATH` defines the
drive or root directory you want to store all data under. Other paths are
appended to this base location.
```shell
cp .env.template .env
# then edit .env
```

Run Python file `create_save_db.py` to create ChromaDB and BM25 databases 
```shell
python create_save_db.py
```

### Run the application 

Begin with running `Ollama` in separate terminal 
```shell
ollama serve
```

Run the python file `app.py` to boot up FastAPI server
```shell
python app.py
```

Run the python file main.py to start streamlit app
```shell
streamlit run main.py
```

### Docker Deployment

Build the image and mount a drive for `BASE_PATH` so that all data and logs are
stored on your chosen volume.

```shell
docker build -t contextual-rag .
docker run -p 8000:8000 -p 8501:8501 \
  -v /my/drive:/data \
  -e BASE_PATH=/data \
  --env-file .env \
  contextual-rag
```

The Streamlit UI will be available on port `8501` and the FastAPI endpoint on
port `8000`.

### Additional Information

- **Contextual Embedding:** The process of prepending chunk-specific explanatory context to each chunk before embedding.
- **Contextual BM25:** A modified version of BM25 that incorporates contextual information for improved relevance scoring.

----
### Follow me

>GitHub [@RionDsilvaCS](https://github.com/RionDsilvaCS)  ·  Linkedin [@Rion Dsilva](https://www.linkedin.com/in/rion-dsilva-043464229/)   ·  Twitter [@Rion_Dsilva_CS](https://twitter.com/rion_dsilva_cs)
