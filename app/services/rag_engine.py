import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import json

class RAGEngine:
    def _classify_text(self, text_snippet):
        prompt = f"""
    Analyze the following research text. Create a single, short, descriptive 
    category tag (e.g., 'Marine Plastic Detection', 'Threat Intelligence').
    Do not use complex sentences. Return ONLY the tag.
    
    Text: {text_snippet[:1000]}
    """
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    def __init__(self, data_dir, db_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0)
        self.vector_store = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        self.bm25_index = None
        self.chunks = []

    def initialize(self):
        import json
        print(f"DEBUG: Scanning directory: {self.data_dir}")
        files = [f for f in os.listdir(self.data_dir) if f.endswith(('.pdf', '.txt'))]
        cache_file = os.path.join(self.data_dir, "category_cache.json")
        
        file_categories = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                file_categories = json.load(f)

        # Check existing data
        data = self.vector_store.get(include=['metadatas', 'documents'])
        
        # Determine if we need to ingest (If DB is empty OR files were added)
        if data['documents'] and not files:
             # Just load what's there
            self.chunks = [Document(page_content=doc, metadata=meta) for doc, meta in zip(data['documents'], data['metadatas'])]
        else:
            print("Ingesting or Refreshing documents...")
            raw_docs = []
            for loader_cls, glob in [(TextLoader, "**/*.txt"), (PyPDFLoader, "**/*.pdf")]:
                loader = DirectoryLoader(self.data_dir, glob=glob, loader_cls=loader_cls)
                raw_docs.extend(loader.load())
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            self.chunks = splitter.split_documents(raw_docs)
            
            print("Categorizing and indexing documents...")
            for doc in self.chunks:
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                
                # If file not in cache, categorize it
                if source not in file_categories:
                    text_sample = doc.page_content[:800]
                    category = self._classify_text(text_sample).strip().replace("'", "").title()
                    file_categories[source] = category
                    print(f"DEBUG: Classified {source} as {category}")
                
                doc.metadata['file_name'] = source
                doc.metadata['category'] = file_categories[source]

            with open(cache_file, 'w') as f:
                json.dump(file_categories, f)

            # CRITICAL: If data exists, clear it to avoid duplicate indices before adding new
            if data['documents']:
                self.vector_store.delete_collection()
                # Re-initialize vector store after deletion
                self.vector_store = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            
            self.vector_store.add_documents(self.chunks)
            print(f"Indexed {len(self.chunks)} chunks.")

        self._rebuild_indices()

    def _rebuild_indices(self):
        tokenized_corpus = [c.page_content.lower().split() for c in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def query(self, user_question: str, filter_dict: dict = None):
        # 1. Hybrid Search
        search_kwargs = {"k": 5}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            filter_key = list(filter_dict.keys())[0]
            filter_value = filter_dict[filter_key]
            filtered_chunks = [
                c for c in self.chunks 
                if c.metadata.get(filter_key, "").lower() == filter_value.lower()
            ]
    
            if not filtered_chunks:
                print(f"DEBUG: No chunks matched {filter_dict}")
        
        vector_docs = self.vector_store.similarity_search(user_question, **search_kwargs)
        
        # 2. BM25 Retrieval using self.chunks
        bm25_docs = []
        if self.bm25_index:
            query_tokens = user_question.lower().split()
            if filter_dict:
                filename_to_match = filter_dict.get('file_name')
                filtered_chunks = [c for c in self.chunks if c.metadata.get('file_name') == filename_to_match]
                if not filtered_chunks:
                    print(f"DEBUG: No chunks found for filter {filter_dict}, skipping BM25.")
                else:
                    temp_tokenized = [c.page_content.lower().split() for c in filtered_chunks]
                    temp_bm25 = BM25Okapi(temp_tokenized)
                    bm25_docs = temp_bm25.get_top_n(query_tokens, filtered_chunks, n=5)
            else:
                bm25_docs = self.bm25_index.get_top_n(query_tokens, self.chunks, n=5)

        # 3. Combine and Deduplicate
        combined_docs = list({d.page_content: d for d in (vector_docs + bm25_docs)}.values())
        
        if not combined_docs:
            return {"answer": "I do not have enough information to answer this question.", "metadata": []}

        # 4. Context Preparation
        combined_context = "\n---\n".join([
            f"Source File: {d.metadata.get('file_name', 'Unknown')}\nContent: {d.page_content}" 
            for d in combined_docs
        ])
        print(f"DEBUG: Retrieved {len(combined_docs)} docs.")
        for i, d in enumerate(combined_docs):
            print(f"DEBUG: Doc {i} Source: {d.metadata.get('file_name')}")
        
        # 5. LLM Prompting
        top_docs = combined_docs[:3] 
        combined_context = "\n---\n".join([d.page_content for d in top_docs])
        # Updated LLM Prompt
        # In your RAGEngine.query() method:
        prompt = ChatPromptTemplate.from_template("""
    You are a research assistant. 
    Question: {question}
    
    Context: {context}
    
    INSTRUCTION: If the question asks for "promising niches", 
    look for keywords like "gender", "inequality", "peace", or "consumption".
    Answer precisely based on the context.
    """)
        response = self.llm.invoke(prompt.format(context=combined_context, question=user_question))
        
        return {"answer": response.content, "metadata": [d.metadata for d in combined_docs]}