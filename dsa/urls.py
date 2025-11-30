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
    path("tree/", views.trees_visualize, name="trees_visualize"),
    path("api/tree/", views.get_tree),
    path("api/tree/insert/", views.insert_node),
    path("api/tree/delete/", views.delete_node),
    path("api/tree/clear/", views.clear_tree),
    path('graph/', views.graph_visualize, name="graph_visualize"),
    path('api/graph/', views.get_graph),
    path('api/graph/add/', views.add_edge),
    path('api/graph/remove/', views.remove_edge),
    path('api/graph/clear/', views.clear_graph),
    path('linked_list/', views.linked_list_visualize, name='linked_list_visualization'),
    path('queue/', views.queue_visualize, name='queue_visualize'),
    path('trie/', views.trie_visualize, name='trie_visualize')

]

