from django.urls import path
from .views import DocumentUploadView

urlpatterns = [
    path('upload-doc/', DocumentUploadView.as_view(), name='upload-doc'),
]