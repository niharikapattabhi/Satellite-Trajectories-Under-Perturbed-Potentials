import pyflowchart

@pyflowchart.flowchart
def add(a, b):
    c = a + b
    return c

a = 10
b = 20
result = add(a, b)

print(result)