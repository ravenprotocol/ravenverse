from django.conf.urls import url
from django.urls import path

from app import views


urlpatterns = [
    path('home/', views.home, name="home"),
    path('compute/', views.Compute.as_view(), name='Compute'),
    path('result/<int:op_id>', views.Result.as_view(), name='Result'),
    path('compute_logistic_regression/', views.ComputeLogisticRegression.as_view(), name='ComputeLogisticRegression'),
    path('status_logistic_regression/<int:id>/', views.StatusLogisticRegression.as_view(), name='StatusLogisticRegression'),
    path('predict_logistic_regression/<int:id>/', views.PredictLogisticRegression.as_view(), name='PredictLogisticRegression'),
]
