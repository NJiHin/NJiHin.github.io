An extremely eye opening experiencing creating my first RAG and how deep into each aspect of RAG you can go into when optimising it. 
# Architecture
A very straightforward pipeline/ architecture was implemented, with a focus on implementation using LangChain  and locally hosted LLMs downloaded from HuggingFace.

Use local model from HF quantised with Bitsandbytes
↓
Ingest pdf with some preprocessing using PyMuPDF
↓
chunk the text
↓
run through a local embeddings model, also downloaded from HF
↓
Embed the pdf and save into ChromaDB vectorstore
↓
define prompt
↓
retrieval and generation using LangChain

---
# Starting off
From the very start, I knew that I wanted to run the LLMs locally. I started started with Ollama, but knowing that i would need the finer grain control, switched to downloading models straight from HuggingFace. 

Wanting to maximise my hardware capabilities, I opted for `Qwen/Qwen2.5-7B-Instruct`, a 7 billion parameter LLM that could be ran on my Nvidia RTX 3060Ti (8GB VRam) with 4-bit quantisation.
## Quantisation
Without Ollama, I had to learn the basics of quantisation and implementation through BitsandBytes.
## Ingestion
Initially, PyPDF was used for PDF ingestion, but switched to PyMUPDF due to better performance. PDFPlumber was also considered due to it's ability to handle table data better, but skimming through my knowledge base shows there is not much tables and hence PyMUPDF was selected in the end.
## Chunking
Chunk sizing and chunk overlap was played around with, and some thought was given into implementing more advanced chunking algorithms, such as Semantic Chunking or Recursive Chunking. Ultimately, I settled on Fixed-size chunking for simplicity as the focus for this leg of the project was to understand the basic RAG structure. 
## Embeddings
Some research was put into potentially finding fine-tuned embeddings from HuggingFace just for customisability, but I ended sticking with the reliable `sentence-transformers/all-MiniLM-L6-v2` as my embeddings model.
## VectorStore
Initially created a very simple rag using FAISS, but ChromaDB had native persistent storage, so I pivoted to ChromaDB. It is also more suitable for my use case where retrieval speed is not of utmost importance.

Alot of consideration is taken for the type of data ingested into the knowledge base. There are two main criteria:
1. The documents had to be publicly downloadable in pdf
2. Must not require extensive domain knowledge to verify

Initially, I wanted to do a pharmaceutical/ medical RAG, but realised that it would be hard to verify the correctness and accuracy of the answer. Upon more exploration, I decided that a suitable knowledge base would be MATLAB syntax. 
## LangChain
I was most interested in learning and using LangChain, experimenting with its components. Keeping things clean, I used few components in my chain for a lightweight RAG.
```python
rag_chain = (
    {"context": retriever 
    | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

# Learning Points
With almost endless customisations depending on the model and parameters, I found it easy to be caught up with the minute details, such as finding high quality documents for my knowledge base. I had to remind myself the main goal of this project - which was to understand the basic architecture and components of RAG. Of course, when implementing a real project, great though has to be put into the knowledge base, but in this context, it was just to test out things and see how to implement and use these tools.
# Moving Forward
I intend to keep building up this RAG while learning more advanced techniques, such as  Reranking and MMR, Evaluation using LLM as a judge, as well as creating a local demo using FastAPI and StreamLit. Ultimately, I would want to learn deployment with containerisation and cloud platforms.

# References
 https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html