from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key
#persist_dir = "./chroma_db"

class LegalQueryView(APIView):
    def post(self, request):
        question = request.data.get("question")
        session_id = request.data.get("session_id")  # Required for document context

        if not question or not session_id:
            return Response(
                {"error": "Both 'question' and 'session_id' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Load the Chroma DB for that session
        persist_dir = f"vectorstore/chroma_store/{session_id}"
        if not os.path.exists(persist_dir):
            return Response(
                {"error": f"No vector store found for session ID: {session_id}"},
                status=status.HTTP_404_NOT_FOUND
            )

        # Setup Chroma + Retriever
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Setup LLM + RetrievalQA chain
        llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        try:
            answer = qa_chain.run(question)
            return Response({"answer": answer})
        except Exception as e:
            return Response({"error": str(e)}, status=500)
