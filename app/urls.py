from django.conf.urls import url
from django.urls import path

from app import views


urlpatterns = [
    path('home/', views.home, name="home"),
    path('compute/', views.Compute.as_view(), name='Compute'),
    path('result/<int:op_id>', views.Result.as_view(), name='Result'),
]
