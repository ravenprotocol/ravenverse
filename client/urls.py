from django.urls import path

from client import views


urlpatterns = [
    path('home/', views.home, name="home"),
]