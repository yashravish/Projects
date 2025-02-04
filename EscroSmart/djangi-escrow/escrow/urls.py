from django.urls import path
from .views import create_escrow

urlpatterns = [
    path('create/', create_escrow, name='create_escrow'),
]
