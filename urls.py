from django.urls import path
from .views import  chat, index_backend_document,login_view

urlpatterns = [
    path("index/", index_backend_document, name="index_backend_document"),
    path("chat/", chat, name="chat"),
    path('login/', login_view, name='login'),
]