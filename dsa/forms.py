from django import forms
ALGORITHM_CHOICES=[
    ('bubbleSort','bubbleSort'),
    ('insertionSort','insertionSort'),
    ('selectionSort', 'selectionSort'),
    ('mergeSort', 'mergeSort')
]
class SearchForm(forms.Form):
    array = forms.CharField(label="Array (comma-separated)", widget=forms.TextInput(attrs={"placeholder": "3,1,4,1,5"}))
    target = forms.IntegerField(label="Target Value", widget=forms.NumberInput(attrs={"placeholder": "Enter target"}))
    algorithm = forms.ChoiceField(
        label="Algorithm",
        choices=[("linear", "Linear Search"), ("binary", "Binary Search")],
        widget=forms.RadioSelect  # Display options as radio buttons
    )
class SortingForm(forms.Form):
    numbers=forms.CharField(label='numbers', max_length=100)
    algo=forms.CharField(label='Select algorithm', widget=forms.Select(choices=ALGORITHM_CHOICES))