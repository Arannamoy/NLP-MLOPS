### P-1: Using filter() and lambda, extract even number from a list.
Example: 
Input: 5 6 7 8 9 10
output: [6 8 10]

`Solution:`
```py
inputs=list(input().split())
inputs=map(lambda x:int(x),inputs)
output=list(filter(lambda x:x%2==0,inputs))
print(output)
```

### P-2: Write a function safe_divide(a,b) that performs a/b and handles division by zero using try–except. Print "Division by zero" if b = 0.

Input Format:
Two integers a b.
Output Format:
Result or error message.

Example:
Input: 10 0  
Output: Division by zero

`Solution:`
```py
def safe_divide(a,b):
    try:
        print(a/b)
    except Exception as e:
        print(e)

inputs=input().split()
inputs=list(map(lambda x:int(x),inputs))
safe_divide(inputs[0],inputs[1])
```

### P-3: Create a base class Shape with a method area() that returns 0. Create subclasses:Rectangle(w,h) → area = w × h, Circle(r) → area = π × r². Read shape type and parameters, then print area (rounded to 2 decimals).

Input Format 1:
Rectangle 5 10
Output Format 1:
Area: 50.00

Input Format 2:
Circle 7
Output Format 2:
Area: 153.94

`Solution:`
```py
from math import pi
class Shape:
    def area(self):
        return 0

class Rectangle(Shape):
    def __init__(self,w,h):
        self.w=w
        self.h=h
    def area(self):
        return self.w*self.h

class Circle(Shape):
    def __init__(self,r):
        self.r=r
    def area(self):
        return pi*self.r**2

def area(shape_type):
    print("Area: %.2f"%shape_type.area())

inputs=list(input().split())
if inputs[0]=="Rectangle":
    area(Rectangle(float(inputs[1]),float(inputs[2])))
    
elif inputs[0]=="Circle":
    area(Circle(float(inputs[1])))

```

### P-4: Find and print the longest word in a given sentence (if tie, print first one).

Input Format:
One line string
Output Format:
Longest word

Example:
Input: Python programming is enjoyable
Output: programming

`Solution`
```py
inputs=input()
inputs=inputs.split()
inputs.sort(key=len,reverse=True)
print(inputs[0])
```
### P-5: A number is called strong if the sum of the factorial of its digits equals the number itself. Example: 145 = 1! + 4! + 5!

Input Format:
Single integer
Output Format:
"Strong Number" or "Not Strong Number"

Example:
Input: 145
Output: Strong Number

`Solution`

```py
from math import factorial
inputs=input()
sum=0
for i in list(inputs):
    sum+=factorial(int(i))

if sum==int(inputs):
    print("Strong Number")
else:
    print("Not Strong Number")

```