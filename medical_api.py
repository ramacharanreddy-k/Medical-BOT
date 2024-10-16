import os
import asyncio
import json

from dotenv import load_dotenv

from typing import List
from pydantic import BaseModel, Field, PrivateAttr

from pinecone import Pinecone as PineconeClient

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

app = FastAPI()

# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "medicalbot"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings,
    text_key="summary"
)

class EnhancedRetriever(BaseRetriever):
    top_k: int = Field(default=5, description="Number of top results to return")
    _vectorstore: PineconeVectorStore = PrivateAttr()

    def __init__(self, vectorstore: PineconeVectorStore, top_k: int = 5):
        super().__init__()
        self._vectorstore = vectorstore
        self.top_k = top_k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self._vectorstore.similarity_search(query, k=self.top_k*2)
        
        processed_results = []
        for doc in results:
            try:                
                if isinstance(doc, Document):
                    metadata = doc.metadata
                else:
                    print(f"Unexpected document type: {type(doc)}")
                    continue
                
                if isinstance(metadata, dict):
                    keywords = metadata.get('Keywords', '')
                elif isinstance(metadata, str):
                    print(f"Metadata is a string: {metadata}")
                    keywords = ''
                else:
                    print(f"Unexpected metadata type: {type(metadata)}")
                    keywords = ''
                
                if isinstance(keywords, str):
                    keywords = keywords.lower().split(', ')
                else:
                    keywords = []
                
                query_words = query.lower().split()
                keyword_score = sum(1 for word in query_words if word in keywords)
                
                combined_score = metadata.get('score', 0) if isinstance(metadata, dict) else 0
                combined_score += (keyword_score * 0.1)
                
                processed_results.append({
                    'document': doc,
                    'combined_score': combined_score
                })
            except Exception as e:
                print(f"Error processing document: {e}")
                print(f"Document causing error: {doc}")
                continue
        
        top_results = sorted(processed_results, key=lambda x: x['combined_score'], reverse=True)[:self.top_k]
        return [item['document'] for item in top_results]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

# Initialize EnhancedRetriever
retriever = EnhancedRetriever(vectorstore, top_k=5)

# Define the prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical assistant bot. Answer the following question based on the given medical transcriptions. If you don't have enough information to answer accurately, say so.
    Remember you are medical chatbot designed for natural human interaction and answer their questions.
    1) To determine their diagnosis.
    2) Help them understand next steps.
    3) Answer any and all questions they ask within your knowledge.
    4) Suggest which healthcare professional they need to consult.
    5) If asked also recommend medications using the medical transcriptions.
    
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
    
    {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Initialize language model
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# Create document chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class ChatRequest(BaseModel):
    message: str

class Source(BaseModel):
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

class ChatProcessingError(Exception):
    def __init__(self, message: str):
        self.message = message

@app.exception_handler(ChatProcessingError)
async def chat_processing_exception_handler(request: Request, exc: ChatProcessingError):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred while processing the chat: {exc.message}"}
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = rag_chain.invoke({
            "input": request.message,
            "chat_history": memory.chat_memory.messages
        })
                
        if isinstance(response, str):
            bot_response = response
            sources = []
        elif isinstance(response, dict):
            bot_response = response.get('answer', 'No answer provided')
            source_documents = response.get('context', [])
            sources = [Source(content=doc.page_content) for doc in source_documents if hasattr(doc, 'page_content')]
        else:
            raise ValueError(f"Unexpected response type from RAG Chain: {type(response)}")

        memory.chat_memory.add_user_message(request.message)
        memory.chat_memory.add_ai_message(bot_response)

        return ChatResponse(answer=bot_response, sources=sources)
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the chat: {str(e)}")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: rag_chain.invoke({
                    "input": data,
                    "chat_history": memory.chat_memory.messages
                })
            )
            bot_response = response['answer']
            
            # Update memory
            memory.chat_memory.add_user_message(data)
            memory.chat_memory.add_ai_message(bot_response)
            
            await manager.send_personal_message(f"Bot: {bot_response}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)