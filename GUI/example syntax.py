# Python Syntax Cheatsheet for VS Code

# This file provides a comprehensive overview of common Python syntax and their functionalities.
# It's designed to be easily copied and used in VS Code for quick reference and practice.

# --- 1. Basic Syntax ---

# Comments: Used to explain code. Single-line comments start with #.
# Multi-line comments can be enclosed in triple quotes (single or double).
"""
This is a multi-line comment.
It explains a block of code or provides general information.
"""

# Variables: Used to store data. No explicit type declaration is needed.
# Naming rules: Must start with a letter or underscore, cannot start with a digit,
# can only contain alphanumeric characters and underscores, are case-sensitive.
my_integer = 10
my_float = 20.5
my_string = "Hello, Python!"
my_boolean = True

# Printing output: The print() function displays values to the console.
print("--- 1. Basic Syntax ---")
print("My integer:", my_integer)
print("My float:", my_float)
print("My string:", my_string)
print("My boolean:", my_boolean)
print("Printing multiple outputs on the same line:", "A", "B", "C", end=" ")
print("D") # Continues on the same line because of end=" " above
print("This is a new line.")

# Input from user: The input() function reads a line of text from the user.
# It always returns a string, so type conversion is often necessary.
# user_name = input("Enter your name: ")
# print("Hello,", user_name)
# user_age_str = input("Enter your age: ")
# try:
#     user_age_int = int(user_age_str)
#     print("You are", user_age_int, "years old.")
# except ValueError:
#     print("Invalid age entered. Please enter a number.")

# Type conversion functions:
# int(): Converts to an integer.
# float(): Converts to a floating-point number.
# str(): Converts to a string.
# print(int("123"))
# print(float("12.34"))
# print(str(500))

# --- 2. Operators ---

# Arithmetic Operators: Perform mathematical calculations.
print("\n--- 2. Operators (Arithmetic) ---")
a = 10
b = 3
print("Addition (a + b):", a + b)           # 13
print("Subtraction (a - b):", a - b)        # 7
print("Multiplication (a * b):", a * b)     # 30
print("Division (a / b):", a / b)           # 3.333... (float division)
print("Floor Division (a // b):", a // b)   # 3 (integer division, rounds down)
print("Modulus (a % b):", a % b)           # 1 (remainder)
print("Exponentiation (a ** b):", a ** b)   # 1000 (10 to the power of 3)

# Assignment Operators (Augmented Assignment): Shorthand for arithmetic operations.
print("\n--- 2. Operators (Assignment) ---")
x = 5
print("Initial x:", x)
x += 2  # Equivalent to x = x + 2
print("x after x += 2:", x) # 7
x -= 3  # Equivalent to x = x - 3
print("x after x -= 3:", x) # 4
x *= 2  # Equivalent to x = x * 2
print("x after x *= 2:", x) # 8
x /= 4  # Equivalent to x = x / 4
print("x after x %= 1:", x) # 0.0

# Comparison Operators: Compare two values and return a Boolean (True/False).
print("\n--- 2. Operators (Comparison) ---")
val1 = 15
val2 = 25
print("val1 == val2:", val1 == val2) # False (Equal to)
print("val1 != val2:", val1 != val2) # True (Not equal to)
print("val1 < val2:", val1 < val2)   # True (Less than)
print("val1 > val2:", val1 > val2)   # False (Greater than)
print("val1 <= val2:", val1 <= val2) # True (Less than or equal to)
print("val1 >= val2:", val1 >= val2) # False (Greater than or equal to)

# Logical Operators: Combine conditional statements.
print("\n--- 2. Operators (Logical) ---")
condition1 = (val1 > 10)  # True
condition2 = (val2 < 20)  # False
print("condition1 and condition2:", condition1 and condition2) # False
print("condition1 or condition2:", condition1 or condition2)   # True
print("not condition1:", not condition1)                      # False

# The 'in' keyword: Checks for presence of an element in a sequence.
my_list = [10, 20, 30, 40]
print("30 in my_list:", 30 in my_list)   # True
print("50 in my_list:", 50 in my_list)   # False
my_string_check = "Hello World"
print("'World' in my_string_check:", "World" in my_string_check) # True

# Identity Operators: 'is' and 'is not' compare the identity of two objects.
# They check if two variables refer to the exact same object in memory.
print("\n--- 2. Operators (Identity) ---")
x = [1, 2, 3]
y = [1, 2, 3]
z = x
print("x is y:", x is y)   # False (different objects, even if content is same)
print("x is z:", x is z)   # True (z refers to the same object as x)
print("x is not y:", x is not y) # True

# Membership Operators: 'in' and 'not in' check for membership in a sequence.
print("\n--- 2. Operators (Membership) ---")
my_sequence = [10, 20, 30]
print("10 in my_sequence:", 10 in my_sequence) # True
print("40 not in my_sequence:", 40 not in my_sequence) # True

# --- 3. Control Flow ---

# Conditional Statements (if-elif-else): Execute code based on conditions.
print("\n--- 3. Control Flow (if-elif-else) ---")
score = 75
if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: F")

# For Loop: Iterates over a sequence (list, tuple, string, range, etc.).
print("\n--- 3. Control Flow (for loop) ---")
# Using range():
print("For loop with range(5):")
for i in range(5): # Iterates from 0 up to (but not including) 5
    print(i, end=" ")
print() # New line

print("For loop with range(1, 6, 2): (start, stop, step)")
for i in range(1, 6, 2): # Iterates from 1, then 1+2=3, then 3+2=5
    print(i, end=" ")
print()

# Iterating over a list:
fruits = ["apple", "banana", "cherry"]
print("Iterating over a list:")
for fruit in fruits:
    print(fruit)

# Iterating over a string:
word = "Python"
print("Iterating over a string:")
for char in word:
    print(char, end="-")
print()

# enumerate(): Gets both index and value when iterating.
print("Iterating with enumerate():")
for index, fruit in enumerate(fruits):
    print(f"Index {index}: {fruit}")

# While Loop: Continues as long as a condition is True.
print("\n--- 3. Control Flow (while loop) ---")
count = 0
while count < 3:
    print("Count is:", count)
    count += 1

# break statement: Exits the loop immediately.
print("\n--- 3. Control Flow (break) ---")
for i in range(10):
    if i == 5:
        print("Breaking loop at i = 5")
        break
    print(i, end=" ")
print()

# continue statement: Skips the rest of the current loop iteration and moves to the next.
print("\n--- 3. Control Flow (continue) ---")
for i in range(5):
    if i == 2:
        print("Skipping 2")
        continue
    print(i, end=" ")
print()

# pass statement: A null operation; nothing happens when it executes.
# Useful as a placeholder when syntax requires a statement but you want no action.
print("\n--- 3. Control Flow (pass) ---")
def my_empty_function():
    pass # This function does nothing yet

if True:
    pass # This if block does nothing yet

print("Pass statement example completed.")

# --- 4. Data Structures ---

# Lists: Ordered, mutable (changeable) collections of items. Enclosed in square brackets [].
print("\n--- 4. Data Structures (Lists) ---")
my_list = [10, 20, "hello", 30.5, True]
print("Original list:", my_list)

# Accessing elements (indexing):
print("First element (index 0):", my_list[0])   # 10
print("Last element (index -1):", my_list[-1]) # True
# Multi-dimensional list (list of lists):
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("Element at matrix[1][2]:", matrix[1][2]) # 6

# Modifying elements:
my_list[0] = 100
print("List after modifying index 0:", my_list)

# Slicing lists: [start:end:step]
print("Slice [1:4]:", my_list[1:4])       # ['hello', 30.5] (index 4 is excluded)
print("Slice [:3]:", my_list[:3])        # [100, 20, 'hello'] (from start to index 3 excluded)
print("Slice [2:]:", my_list[2:])        # ['hello', 30.5, True] (from index 2 to end)
print("Slice [::2]:", my_list[::2])      # [100, 'hello', True] (every second element)
print("Reverse slice [::-1]:", my_list[::-1]) # [True, 30.5, 'hello', 20, 100]

# List methods:
my_numbers = [1, 2, 3]
my_numbers.append(4)
print("After append(4):", my_numbers)    # Adds element to the end
my_numbers.extend([5, 6])
print("After extend([5, 6]):", my_numbers) # Adds elements of another list
my_numbers.insert(0, 0)
print("After insert(0, 0):", my_numbers)  # Inserts element at specific index
popped_item = my_numbers.pop()
print("After pop():", my_numbers, "Popped item:", popped_item) # Removes and returns last item
popped_at_index = my_numbers.pop(1)
print("After pop(1):", my_numbers, "Popped item at index 1:", popped_at_index) # Removes and returns item at index
my_numbers.remove(3) # Removes the first occurrence of a value
print("After remove(3):", my_numbers)
my_numbers.sort() # Sorts the list in-place
print("After sort():", my_numbers)
my_numbers.reverse() # Reverses the list in-place
print("After reverse():", my_numbers)
print("Index of 2:", my_numbers.index(2)) # Returns the index of the first occurrence of a value
print("Count of 2:", my_numbers.count(2)) # Returns the number of occurrences of a value
my_copy = my_numbers.copy() # Returns a shallow copy of the list
print("Copy of list:", my_copy)
my_numbers.clear() # Removes all items from the list
print("After clear():", my_numbers)


# Tuples: Ordered, immutable (unchangeable) collections of items. Enclosed in parentheses ().
print("\n--- 4. Data Structures (Tuples) ---")
my_tuple = (1, "two", 3.0, False)
print("Original tuple:", my_tuple)
print("First element:", my_tuple[0])
print("Slice [1:3]:", my_tuple[1:3])
# Tuples are immutable: my_tuple[0] = 100 # This would cause an error

# Sets: Unordered, mutable collections of unique items. Enclosed in curly braces {}.
# Duplicate values are automatically removed.
print("\n--- 4. Data Structures (Sets) ---")
my_set = {1, 2, 3, 2, 1}
print("Original set (duplicates removed):", my_set)
my_set.add(4)
print("After adding 4:", my_set)
my_set.remove(2)
print("After removing 2:", my_set)
set1 = {1, 2, 3}
set2 = {3, 4, 5}
print("Union (set1 | set2):", set1 | set2)       # {1, 2, 3, 4, 5}
print("Intersection (set1 & set2):", set1 & set2) # {3}
print("Difference (set1 - set2):", set1 - set2)   # {1, 2}

# Dictionaries: Unordered, mutable collections of key-value pairs. Enclosed in curly braces {}.
# Keys must be unique and immutable (e.g., strings, numbers, tuples).
print("\n--- 4. Data Structures (Dictionaries) ---")
my_dict = {"name": "Alice", "age": 30, "city": "New York"}
print("Original dictionary:", my_dict)

# Accessing values:
print("Name:", my_dict["name"])
print("Age:", my_dict.get("age")) # .get() is safer, returns None if key not found

# Adding/modifying entries:
my_dict["email"] = "alice@example.com"
print("After adding email:", my_dict)
my_dict["age"] = 31
print("After modifying age:", my_dict)

# Iterating over dictionaries:
print("Iterating over keys:")
for key in my_dict:
    print(key, end=" ")
print()

print("Iterating over values:")
for value in my_dict.values():
    print(value, end=" ")
print()

print("Iterating over key-value pairs:")
for key, value in my_dict.items():
    print(f"{key}: {value}")

# Removing entries:
removed_city = my_dict.pop("city")
print("After popping city:", my_dict, "Removed city:", removed_city)
del my_dict["name"]
print("After deleting name:", my_dict)

# Strings: Ordered, immutable (cannot be changed after creation) sequence of characters.
print("\n--- 4. Data Structures (Strings) ---")
my_string = "Python Programming"
print("Original string:", my_string)

# Accessing elements (indexing):
print("First character:", my_string[0])  # P
print("Last character:", my_string[-1]) # g

# Slicing strings: Same as lists.
print("Slice [0:6]:", my_string[0:6])   # Python
print("Slice [7:]:", my_string[7:])    # Programming

# String formatting:
name = "Bob"
age = 25
# f-strings (Formatted string literals - Python 3.6+): Recommended for readability.
print(f"Hello, {name}! You are {age} years old.")
# .format() method:
print("Hello, {}! You are {} years old.".format(name, age))
print("Hello, {0}! You are {1} years old.".format(name, age)) # Positional arguments
print("Hello, {n}! You are {a} years old.".format(n=name, a=age)) # Keyword arguments
# Old-style %-formatting (less common now):
print("Hello, %s! You are %d years old." % (name, age))


# String methods:
print("Length of string:", len(my_string))
print("String in uppercase:", my_string.upper())
print("String in lowercase:", my_string.lower())
print("Index of 'o':", my_string.find('o')) # Returns the lowest index where substring is found.
print("Replaced 'Programming' with 'Learning':", my_string.replace("Programming", "Learning"))
stripped_string = "   hello world   ".strip()
print("Stripped string (remove leading/trailing whitespace):", f"'{stripped_string}'")
split_string = my_string.split(" ") # Splits string by a delimiter (space by default)
print("Split string by space:", split_string)
joined_string = "-".join(["apple", "banana", "cherry"]) # Joins elements of an iterable with the string
print("Joined string with '-':", joined_string)

# Escape sequences for strings:
print("New line with \\n: Hello\nWorld")
print("Tab with \\t: Hello\tWorld")
print("Single quote with \\': It's a 'quote'.")
print("Double quote with \\\": He said \"Hello\".")
print("Backslash with \\\\: C:\\Users\\User")

# Multi-line strings:
multi_line_string = """
This is a string
that spans
multiple lines.
"""
print(multi_line_string)

# --- 5. Functions ---

# Defining a function: Use the 'def' keyword.
print("\n--- 5. Functions ---")
def greet(name): # 'name' is a parameter
    """
    This function greets the person passed in as argument.
    """
    return f"Hello, {name}!" # Return value

# Calling a function:
message = greet("Alice")
print(message)

# Function without arguments or return value:
def say_hi():
    print("Hi there!")

say_hi()

# Functions with default arguments:
def describe_pet(animal="dog", name="Buddy"):
    print(f"I have a {animal} named {name}.")

describe_pet()
describe_pet(animal="cat")
describe_pet(name="Max", animal="parrot")

# Arbitrary Arguments (*args): Allows a function to accept a variable number of positional arguments.
def sum_all_numbers(*args):
    total = 0
    for num in args:
        total += num
    return total

print("Sum of numbers:", sum_all_numbers(1, 2, 3, 4, 5))

# Arbitrary Keyword Arguments (**kwargs): Allows a function to accept a variable number of keyword arguments (as a dictionary).
def print_user_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

print_user_info(first_name="John", last_name="Doe", age=40)

# Lambda Functions (Anonymous Functions): Small, single-expression functions.
# Often used for short operations or as arguments to higher-order functions.
add_two = lambda x: x + 2
print("Lambda add_two(5):", add_two(5))

# --- 6. Modules ---

# Modules are files containing Python definitions and statements.
# They help organize code and enable reuse.

print("\n--- 6. Modules ---")

# Importing an entire module:
import math
print("Using math.sqrt(16):", math.sqrt(16))
print("Using math.pi:", math.pi)

# Importing specific functions from a module:
from math import factorial, ceil
print("Using factorial(5):", factorial(5))
print("Using ceil(4.2):", ceil(4.2))

# Importing with an alias:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Successfully imported matplotlib.pyplot as plt, numpy as np, and pandas as pd.")

# Example usage (simplified, for demonstration of import):
# Numpy for numerical operations:
arr = np.array([1, 2, 3, 4, 5])
print("Numpy array:", arr)
print("Mean of array:", np.mean(arr))

# Pandas for data manipulation:
data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)
print("Pandas DataFrame:\n", df)

# Matplotlib for plotting (basic example, won't show a plot in console):
# plt.plot([0, 1, 2], [0, 1, 4])
# plt.title("Sample Plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show() # This would open a plot window

# --- 7. Graphical User Interface (GUI) - Tkinter ---

# Tkinter is Python's standard GUI (Graphical User Interface) library.
# It provides a fast and easy way to create desktop applications.

print("\n--- 7. Graphical User Interface (GUI) - Tkinter ---")

# Import the tkinter module
import tkinter as tk
from tkinter import ttk # Import ttk for themed widgets

# --- Tkinter Example 1: Basic Window and Label ---
# This part is commented out because running Tkinter GUI directly in a console
# or a non-graphical environment might cause issues or hang the script.
# To run this, uncomment the code and execute it in a Python environment
# that supports GUI (e.g., a local VS Code environment with Python installed).

# def run_basic_gui_example():
#     root = tk.Tk()
#     root.title("Basic Tkinter Window")
#     root.geometry("300x150") # Set window size

#     # Create a Label widget
#     label = tk.Label(root, text="Hello from Tkinter!")
#     label.pack(pady=20) # Use pack() to place the label in the window

#     # Run the Tkinter event loop
#     # This keeps the window open and responsive to user interactions
#     root.mainloop()

# print("Uncomment 'run_basic_gui_example()' and run this file locally to see the GUI.")
# # run_basic_gui_example() # Uncomment to run this example

# --- Tkinter Example 2: Using pack() for Widget Organization ---
# pack(): Organises widgets in blocks before placing them in the parent widget.
# padx and pady: Specify padding (horizontal and vertical margin) around the widget in pixels.
# side: Specifies alignment (LEFT, RIGHT, TOP, BOTTOM).

# def run_pack_example():
#     root = tk.Tk()
#     root.title("Tkinter Pack Example")
#     root.geometry("400x200")

#     # Buttons packed with different sides
#     button_top = ttk.Button(root, text="Top Button")
#     button_top.pack(side=tk.TOP, pady=5)

#     button_left = ttk.Button(root, text="Left Button")
#     button_left.pack(side=tk.LEFT, padx=10, pady=5)

#     button_right = ttk.Button(root, text="Right Button")
#     button_right.pack(side=tk.RIGHT, padx=10, pady=5)

#     button_bottom = ttk.Button(root, text="Bottom Button")
#     button_bottom.pack(side=tk.BOTTOM, pady=5)

#     root.mainloop()

# print("Uncomment 'run_pack_example()' and run this file locally to see the Pack GUI example.")
# # run_pack_example() # Uncomment to run this example

# --- Tkinter Example 3: Using grid() for Widget Organization ---
# grid(): More flexible and powerful way to organise widgets.
# Define the position using row=xx, column=xx in a grid-like table format.
# padx and pady: Same as pack(), for internal padding.
# columnspan/rowspan: Makes a widget span multiple columns/rows.

# def run_grid_example():
#     root = tk.Tk()
#     root.title("Tkinter Grid Example")
#     root.geometry("300x200")

    #     # Configure grid columns and rows to expand
#     root.columnconfigure(0, weight=1)
#     root.columnconfigure(1, weight=1)
#     root.rowconfigure(0, weight=1)
#     root.rowconfigure(1, weight=1)

#     # Labels and Entry widgets arranged in a grid
#     label_username = ttk.Label(root, text="Username:")
#     label_username.grid(row=0, column=0, sticky="e", padx=5, pady=5) # sticky="e" for east/right alignment

#     entry_username = ttk.Entry(root)
#     entry_username.grid(row=0, column=1, sticky="ew", padx=5, pady=5) # sticky="ew" for expand width

#     label_password = ttk.Label(root, text="Password:")
#     label_password.grid(row=1, column=0, sticky="e", padx=5, pady=5)

#     entry_password = ttk.Entry(root, show="*") # show="*" masks password
#     entry_password.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

#     button_login = ttk.Button(root, text="Login")
#     button_login.grid(row=2, column=0, columnspan=2, pady=10) # columnspan makes it span 2 columns

#     root.mainloop()

# print("Uncomment 'run_grid_example()' and run this file locally to see the Grid GUI example.")
# # run_grid_example() # Uncomment to run this example

# --- 8. Error Handling (try-except-finally) ---
# Handles runtime errors (exceptions) gracefully to prevent program crashes.

print("\n--- 8. Error Handling (try-except-finally) ---")

def divide_numbers(numerator, denominator):
    try:
        result = numerator / denominator
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Invalid input types. Please provide numbers.")
        return None
    else: # Executed if no exceptions occurred in the try block
        print(f"Division successful. Result: {result}")
        return result
    finally: # Always executed, regardless of whether an exception occurred
        print("Execution of division attempt finished.")

divide_numbers(10, 2)
divide_numbers(10, 0)
divide_numbers(10, "a")

# Raising exceptions:
# You can explicitly raise an exception if a certain condition is not met.
def check_positive(number):
    if number < 0:
        raise ValueError("Number must be positive!")
    return number

try:
    print("Checked positive:", check_positive(5))
    print("Checked positive:", check_positive(-1)) # This will raise an exception
except ValueError as e:
    print(f"Caught exception: {e}")

# --- 9. File Input/Output (I/O) ---
# Reading from and writing to files.

print("\n--- 9. File Input/Output (I/O) ---")

file_name = "example.txt"

# Writing to a file: 'w' mode (write, overwrites if file exists)
try:
    with open(file_name, 'w') as file:
        file.write("Hello, Python file I/O!\n")
        file.write("This is the second line.\n")
    print(f"Content written to '{file_name}' successfully.")
except IOError as e:
    print(f"Error writing to file: {e}")

# Appending to a file: 'a' mode (append, adds to end of file)
try:
    with open(file_name, 'a') as file:
        file.write("This line was appended.\n")
    print(f"Content appended to '{file_name}' successfully.")
except IOError as e:
    print(f"Error appending to file: {e}")

# Reading from a file: 'r' mode (read)
# file.read(): Reads the entire contents of the file as a single string.
# file.readline(): Reads the next single line from the file.
# file.readlines(): Reads all lines into a list of strings.
try:
    with open(file_name, 'r') as file:
        content = file.read()
        print(f"\n--- Content of '{file_name}' (read()): ---\n{content}")
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
except IOError as e:
    print(f"Error reading file: {e}")

try:
    with open(file_name, 'r') as file:
        print(f"\n--- Content of '{file_name}' (readline() loop): ---")
        line1 = file.readline()
        print(f"First line: {line1.strip()}") # .strip() removes leading/trailing whitespace including newline
        line2 = file.readline()
        print(f"Second line: {line2.strip()}")
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
except IOError as e:
    print(f"Error reading file: {e}")

try:
    with open(file_name, 'r') as file:
        lines = file.readlines()
        print(f"\n--- Content of '{file_name}' (readlines()): ---")
        for i, line in enumerate(lines):
            print(f"Line {i+1}: {line.strip()}")
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
except IOError as e:
    print(f"Error reading file: {e}")

# Binary mode: 'rb' and 'wb' for non-text files (images, etc.)
# with open('image.jpg', 'rb') as f:
#     binary_data = f.read()

# --- 10. List/Dictionary Comprehensions & Generator Expressions ---
# Concise ways to create lists, dictionaries, or generators.

print("\n--- 10. Comprehensions & Generators ---")

# List Comprehension: [expression for item in iterable if condition]
squares = [x**2 for x in range(10) if x % 2 == 0]
print("Squares of even numbers:", squares)

# Dictionary Comprehension: {key_expression: value_expression for item in iterable if condition}
square_dict = {x: x**2 for x in range(5)}
print("Dictionary of squares:", square_dict)

# Generator Expression (uses parentheses instead of square brackets):
# Creates an iterator, generating values on the fly, saving memory.
gen_squares = (x**2 for x in range(10) if x % 2 == 0)
print("Generator expression (an iterator):", gen_squares)
print("Next square from generator:", next(gen_squares))
print("Remaining squares from generator:", list(gen_squares)) # Convert to list to see all generated values

# --- 11. Context Managers (with statement) ---
# Ensures resources are properly acquired and released (e.g., files, locks).

print("\n--- 11. Context Managers (with statement) ---")

# Already seen with file I/O:
# with open("another_example.txt", "w") as f:
#     f.write("This was written using a context manager.")
# print("Context manager ensures file is closed automatically.")

# --- 12. Decorators (brief mention) ---
# A way to modify or enhance functions/methods without changing their source code.
# Syntax: @decorator_name above a function definition.
print("\n--- 12. Decorators (Conceptual) ---")
# @my_decorator
# def my_function():
#     pass
print("Decorators are used to wrap functions, adding functionality.")
print("E.g., @property for getters/setters, @classmethod, @staticmethod.")

# --- 13. Modules from s5_Python for Statistics.pdf (Metrics) ---
# These are functions from the sklearn.metrics module for model evaluation.
# To use these, you would typically have actual and predicted data.
print("\n--- 13. Statistical Metrics (from s5_Python for Statistics.pdf) ---")
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math

# Example data (hypothetical, for demonstration)
actual_values = [10, 12, 15, 18, 20]
predicted_values = [11, 11.5, 14, 19, 21]

try:
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = math.sqrt(mse)
    r2 = r2_score(actual_values, predicted_values)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Coefficient of Determination (R-squared): {r2:.2f}")
except Exception as e:
    print(f"Could not calculate metrics. Ensure sklearn is installed and data is valid. Error: {e}")
    print("Install scikit-learn: pip install scikit-learn")


print("\n--- End of Python Syntax Cheatsheet ---")
