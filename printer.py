import numpy as np 

def print_under_other(text: str, values: list):
    print(text)
    print(np.round(values, 2))
    
def print_in_line(text: str, values: list):
    print(str(text)+str(values))
def print_header(text:str):
    x = text.center(100, "*")
    print(x)
