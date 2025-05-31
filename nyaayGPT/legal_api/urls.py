from django.urls import path
from .views import LegalQueryView

urlpatterns = [
    path('legal-query/', LegalQueryView.as_view(),name='leagal-query'),
]
