from django.urls import path
from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path('sort/', views.sort, name="sort"),
    path('search/', views.search, name="search"),
    path('processing/', views.processing, name='processing'),
    path('search_visualize', views.search_visualize, name="search_visualize"),
    path('datastructures/', views.datastructures, name="datastructures" ),
    path('array_visualize', views.array_visualizer, name="array_visualize"),
    path("stack_visualize/", views.stack_visualizer, name="stack_visualize"),
    path('trees_visualize',views.tree_visualize, name="trees_visualize")
]

