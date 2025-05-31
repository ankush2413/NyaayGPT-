from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import DocumentUploadSerializer
from .document_utils import save_uploaded_file, process_and_embed,process_and_embed_without_batch
from uuid import uuid4

class DocumentUploadView(APIView):
    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data['file']
            session_id = serializer.validated_data.get('session_id') or str(uuid4())

            file_path = save_uploaded_file(file, session_id)
            chunks = process_and_embed(file_path, session_id)

            return Response({
                "message": "✅ Document processed and embedded",
                "session_id": session_id,
                "chunks_stored": chunks
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
