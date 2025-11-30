from django.shortcuts import render, redirect
from .forms import SortingForm
import json
from django.shortcuts import render
from django.http import JsonResponse
from .forms import SearchForm
def home(request):
    return render(request, 'home')
def sort(request):
    form = SortingForm()
    return render(request, 'sort', {'form': form})


def search(request):
    if request.method == "POST":
        try:
            # Get form data
            array_str = request.POST.get("array", "")
            target_str = request.POST.get("target", "")
            algorithm = request.POST.get("algorithm", "linear")

            # Convert array and target to integers
            if array_str:
                array = list(map(int, array_str.split(",")))
            else:
                raise ValueError("Array cannot be empty.")

            if target_str:
                target = int(target_str)
            else:
                raise ValueError("Target cannot be empty.")

            # Store data in the session
            request.session["array"] = array
            request.session["target"] = target
            request.session["algorithm"] = algorithm

            # Redirect to visualization page after successful form submission
            return redirect('search_visualize')  # Make sure 'search_visualize' is the correct URL name

        except ValueError as e:
            return JsonResponse({"error": str(e)}, status=400)

    return render(request, "search")

# View for visualizing search steps
def search_visualize(request):
    array = request.session.get("array", [])
    target = request.session.get("target", None)
    algorithm = request.session.get("algorithm", "linear")

    # Sample data for steps (you can replace this with actual dynamic steps)
    if algorithm == "linear":
        steps = linear_search(array, target)  # Get steps from your function
    elif algorithm == "binary":
        steps = binary_search(array, target) 
    
    # Serialize the steps data to a JSON-safe string before passing it to the template
    steps_json = json.dumps(steps)

    return render(request, "search_visualization", {
        "array": array,
        "steps": steps_json,
        "target": target,
        "algorithm": algorithm
    })

def processing(request):
    if request.method == 'POST':
        form = SortingForm(request.POST)
        if form.is_valid():
            x = request.POST['numbers']
            y = request.POST['algo']  # Algorithm choice from the form
            z = x.split()  
            numbers = list(map(int, z))  # Convert input to list of integers

            # Determine which sorting algorithm to use
            if y == "bubbleSort":
                steps = bubbleSort(numbers)
            elif y == "insertionSort":
                steps = insertionSort(numbers)
            elif y == "selectionSort":
                steps = selectionSort(numbers)
            elif y == "mergeSort":
                steps = mergeSort(numbers)
            else:
                steps = []

            return render(request, 'processing', {'steps': json.dumps(steps)})

    return render(request, 'home', {'form': form})
def datastructures(request):
    return render(request, 'datastructures')
def array_visualizer(request):
    array = []
    
    if request.method == "POST":
        array_str = request.POST.get("array", "")
        if array_str:
            try:
                array = list(map(int, array_str.split(",")))
            except ValueError:
                array = []  # Reset if input is invalid

    return render(request, "array_visualize", {"array": array})
def stack_visualizer(request):
    return render(request, "stack_visualize")
# Bubble Sort
def bubbleSort(arr):
    steps = []
    n = len(arr)
    steps.append(arr.copy())  # Initial state

    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            # Log the comparison
            temp = arr.copy()
            temp[j] = [arr[j], "compare"]
            temp[j + 1] = [arr[j + 1], "compare"]
            steps.append(temp)  

            if arr[j] > arr[j + 1]:  
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

                # Log the swap
                temp = arr.copy()
                temp[j] = [arr[j], "swap"]
                temp[j + 1] = [arr[j + 1], "swap"]
                steps.append(temp)

        if not swapped:  # Stop early if sorted
            break

    return steps

# Insertion Sort
def insertionSort(arr):
    steps = []
    n = len(arr)
    steps.append(arr.copy())  # Initial state

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        # Log comparison before shifting
        temp = arr.copy()
        temp[j] = [arr[j], "compare"]
        temp[i] = [arr[i], "compare"]
        steps.append(temp)

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]

            # Log shifting step
            temp = arr.copy()
            temp[j + 1] = [arr[j + 1], "swap"]  
            steps.append(temp)

            j -= 1

        arr[j + 1] = key

        # Log insertion step
        temp = arr.copy()
        temp[j + 1] = [arr[j + 1], "swap"]
        steps.append(temp)

    steps.append(arr.copy())  # Capture final sorted state
    return steps

# Selection Sort
def selectionSort(arr):
    steps = []
    n = len(arr)
    steps.append(arr.copy())  # Initial state

    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            # Log the comparison
            temp = arr.copy()
            temp[j] = [arr[j], "compare"]
            temp[min_idx] = [arr[min_idx], "compare"]
            steps.append(temp)

            if arr[j] < arr[min_idx]:
                min_idx = j

        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]

            # Log the swap step
            temp = arr.copy()
            temp[i] = [arr[i], "swap"]
            temp[min_idx] = [arr[min_idx], "swap"]
            steps.append(temp)

    steps.append(arr.copy())  # Capture final sorted state
    return steps

# Merge Sort
def mergeSort(arr):
    steps = []  # To store each step

    def merge(arr, l, m, r):
        left = arr[l:m + 1]
        right = arr[m + 1:r + 1]

        # Show merging process
        temp_arr = arr.copy()
        temp_arr[l:r + 1] = ["_"] * (r - l + 1)  # Mark merging area with "_"
        steps.append(temp_arr.copy())

        i = j = 0
        k = l

        while i < len(left) and j < len(right):
            # Log comparison
            temp_arr = arr.copy()
            temp_arr[k] = [arr[k], "compare"]
            steps.append(temp_arr.copy())

            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
            steps.append(arr.copy())

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
            steps.append(arr.copy())

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
            steps.append(arr.copy())

    def merge_sort_helper(arr, l, r):
        if l < r:
            m = (l + r) // 2

            # Show dividing process
            temp_arr = arr.copy()
            temp_arr[l:r + 1] = ["|"] * (r - l + 1)  # Mark division with "|"
            steps.append(temp_arr.copy())

            merge_sort_helper(arr, l, m)
            merge_sort_helper(arr, m + 1, r)
            merge(arr, l, m, r)

    arr_copy = arr.copy()
    steps.append(arr_copy.copy())  # Capture initial state
    merge_sort_helper(arr_copy, 0, len(arr_copy) - 1)

    return steps
def linear_search(array, target):
    steps=[]
    for i in range(len(array)):
        steps.append({"index": i, "value": array[i], "found": array[i]==target})
        if array[i]==target:
            break
    return steps
def binary_search(array, target):
    steps=[]
    array.sort()
    left, right=0, len(array)-1
    while left<=right:
        mid=(left+right)//2
        steps.append({"left": left, "right": right, "mid": mid, "value": array[mid], "found": array[mid]==target})
        if array[mid]==target:
            break
        elif array[mid]< target:
            left=mid+1
        else:
            right=mid-1
    return steps
def stack_visualizer(request):
    # Get the stack from the session (initialize as an empty list if not present)
    stack = request.session.get('stack', [])

    if request.method == "POST":
        action = request.POST.get('action')
        value = request.POST.get('value')

        # Push action
        if action == 'push':
            if value:  # Only push if value is provided
                stack.append(int(value))  # Push value onto the stack
                request.session['stack'] = stack  # Store updated stack in session

        # Pop action
        elif action == 'pop':
            if stack:  # Ensure stack is not empty
                stack.pop()  # Pop the top element from the stack
                request.session['stack'] = stack  # Store updated stack in session

        # Peek action
        elif action == 'peek':
            top = stack[-1] if stack else None  # Get the top of the stack for peek operation
            return render(request, "stack_visualize", {
                'stack': stack,
                'top': top
            })

        # Clear action
        elif action == 'clear':
            stack = []  # Clear the stack
            request.session['stack'] = stack  # Store empty stack in session

        # After the operation, redirect to the same page
        return redirect('stack_visualize')  # Redirect back to the same page to refresh stack

    # Render the page with the current stack
    return render(request, "stack_visualize", {'stack': stack})

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
from django.views.decorators.csrf import csrf_exempt
class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)

    def delete(self, value):
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node, value):
        if not node:
            return None
        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            min_larger_node = self._find_min(node.right)
            node.value = min_larger_node.value
            node.right = self._delete_recursive(node.right, min_larger_node.value)
        return node

    def _find_min(self, node):
        while node.left:
            node = node.left
        return node

    def to_dict(self, node):
        if not node:
            return None
        children = []
        if node.left:
            children.append(self.to_dict(node.left))
        if node.right:
            children.append(self.to_dict(node.right))
        return {
            "value": node.value,
            "children": children
        }

# create a single global tree
tree = BinaryTree()

# default visualization page
def trees_visualize(request):
    return render(request, "trees_visualize")

def get_tree(request):
    return JsonResponse(tree.to_dict(tree.root), safe=False)

@csrf_exempt
def insert_node(request):
    data = json.loads(request.body)
    value = data.get("value")
    if value is not None:
        tree.insert(value)
    return JsonResponse(tree.to_dict(tree.root), safe=False)

@csrf_exempt
def delete_node(request):
    data = json.loads(request.body)
    value = data.get("value")
    if value is not None:
        tree.delete(value)
    return JsonResponse(tree.to_dict(tree.root), safe=False)

@csrf_exempt
def clear_tree(request):
    global tree
    tree = BinaryTree()
    return JsonResponse(tree.to_dict(tree.root), safe=False)
# --- Graph Representation ---
class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v):
        self.adj_list.setdefault(u, []).append(v)
        self.adj_list.setdefault(v, []).append(u)  # For undirected graph

    def remove_edge(self, u, v):
        if u in self.adj_list and v in self.adj_list[u]:
            self.adj_list[u].remove(v)
        if v in self.adj_list and u in self.adj_list[v]:
            self.adj_list[v].remove(u)

    def clear(self):
        self.adj_list = {}

    def to_d3_format(self):
        nodes = [{"id": str(k)} for k in self.adj_list]
        links = []
        seen = set()
        for u in self.adj_list:
            for v in self.adj_list[u]:
                if (v, u) not in seen:
                    links.append({"source": str(u), "target": str(v)})
                    seen.add((u, v))
        return {"nodes": nodes, "links": links}

graph = Graph()

from django.views.decorators.csrf import csrf_exempt

def graph_visualize(request):
    return render(request, "graph_visualise")

def get_graph(request):
    return JsonResponse(graph.to_d3_format(), safe=False)

@csrf_exempt
def add_edge(request):
    data = json.loads(request.body)
    u = data.get("u")
    v = data.get("v")
    if u is not None and v is not None:
        graph.add_edge(u, v)
    return JsonResponse(graph.to_d3_format(), safe=False)

@csrf_exempt
def remove_edge(request):
    data = json.loads(request.body)
    u = data.get("u")
    v = data.get("v")
    if u is not None and v is not None:
        graph.remove_edge(u, v)
    return JsonResponse(graph.to_d3_format(), safe=False)

@csrf_exempt
def clear_graph(request):
    graph.clear()
    return JsonResponse(graph.to_d3_format(), safe=False)
from django.shortcuts import render, redirect

# In-memory linked list for session-based demo


import json
from django.shortcuts import render

linked_list = []

def linked_list_visualize(request):
    global linked_list
    message = ""
    middle_index = -1  # used for middle operations
    head_index = 0 if linked_list else -1

    if request.method == 'POST':
        operation = request.POST.get('operation')
        value = request.POST.get('value', '').strip()
        pos_input = request.POST.get('position', '').strip()

        # Insert front
        if operation == 'insert_front' and value:
            linked_list.insert(0, value)
            middle_index = -1

        # Insert end
        elif operation == 'insert_end' and value:
            linked_list.append(value)
            middle_index = -1

        # Insert at middle
        elif operation == 'insert_middle' and value and pos_input.isdigit():
            pos = int(pos_input)
            if pos < 0: pos = 0
            elif pos > len(linked_list): pos = len(linked_list)
            linked_list.insert(pos, value)
            middle_index = pos

        # Delete front
        elif operation == 'delete_front':
            if linked_list:
                linked_list.pop(0)
            else:
                message = "List is already empty!"
            middle_index = -1

        # Delete end
        elif operation == 'delete_end':
            if linked_list:
                linked_list.pop()
            else:
                message = "List is already empty!"
            middle_index = -1

        # Delete at middle
        elif operation == 'delete_middle' and pos_input.isdigit():
            pos = int(pos_input)
            if 0 <= pos < len(linked_list):
                linked_list.pop(pos)
                middle_index = pos
            else:
                message = "Invalid position!"
                middle_index = -1

        # Clear
        elif operation == 'clear':
            linked_list.clear()
            middle_index = -1

    return render(request, 'linked_list_visualization', {
        'linked_list': linked_list,
        'head_index': head_index,
        'middle_index': middle_index,
        'message': message
    })
# views.py
import json
from django.shortcuts import render

# Example trie structure
trie = {}  # Global trie dictionary

def insert_word(trie, word):
    node = trie
    for char in word:
        if char not in node:
            node[char] = {}
        node = node[char]
    node["$"] = True  # End of word

def search_path(trie, word):
    """Return list of characters traversed while searching."""
    path = []
    node = trie
    for char in word:
        if char in node:
            path.append(char)
            node = node[char]
        else:
            break
    return path

def trie_visualize(request):
    global trie
    animation_path = []
    message = ""

    if request.method == "POST":
        word = request.POST.get("word", "").strip()
        operation = request.POST.get("operation")

        if word:
            if operation == "insert":
                insert_word(trie, word)
                message = f"Inserted '{word}'"
            elif operation == "search":
                animation_path = search_path(trie, word)
                message = f"Search path for '{word}'"

    context = {
        "trie": json.dumps(trie),  # ✅ Ensure valid JS object
        "animation_path": json.dumps(animation_path),  # ✅ Ensure valid JS array
        "message": message,
    }

    return render(request, "trie_visualize", context)
def queue_visualize(request):
    return render(request, 'queue_visualizer')
