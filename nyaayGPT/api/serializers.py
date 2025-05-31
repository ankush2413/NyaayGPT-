from rest_framework import serializers

class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    session_id = serializers.CharField(required=False)