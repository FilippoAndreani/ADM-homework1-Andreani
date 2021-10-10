##EX.1

##INTRODUCTION

#Say "Hello, World!" With Python
print("Hello, World!")


#Python If-Else
import math
import os
import random
import re
import sys


if _name_ == '_main_':
    n= int(input().strip())
    if n%2!=0:
        print('Weird')
    else:
        if 2<= n <=5:
            print('Not Weird')
        elif 6<=n<=20:
            print('Weird')
        elif n>=20:
            print('Not Weird')
    

#Arithmetic Operators
if _name_ == '_main_':
    a = int(input())
    b = int(input())
    
print(a + b)
print(a - b)
print(a * b)

#Python: Division
if _name_ == '_main_':
    a = int(input())
    b = int(input())
    
print(a//b)
print(a/b)


#Loops
if _name_ == '_main_':
    n = int(input())
    for i in range(n):
        print(i**2)

#Write a function
def is_leap(year):
    leap = False
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 ==0:
                leap=True
            else:
                leap=False
        else:
            leap=True
    else:
        leap=False
        
    return leap

year = int(input())
print(is_leap(year))

#Print Function
if _name_ == '_main_':
    n = int(input())
    for i in range(n):
        print(i+1, end='')

        
##DATA TYPES

#List Comprehensions
if _name_ == '_main_':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    newlist=[]
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                newlist.append([i,j,k])

    print([m for m in (newlist)if sum(m)!=n])

#Find the Runner-Up Score!
if _name_ == '_main_':
    n = int(input())
    arr = list(map(int, input().split()))
    new_arr=[]
    for i in arr:
        if i not in new_arr:
            new_arr.append(i)
        
    print(sorted(new_arr)[-2])
    
#Nested Lists
if _name_ == '_main_':
    names=[]
    scores=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        names.append(name)
        scores.append(score)
    result_list=[]
    for i in range(len(names)):
        result_list.append([names[i], scores[i]])
    new_scores=[]
    for j in scores:
        if j not in new_scores:
            new_scores.append(j)
    secondscore=sorted(new_scores)[1]

    for k in sorted(result_list, key= lambda x: x[0]):
        if(k[1]==secondscore):
            print(k[0])
    
#Finding the percentage
if _name_ == '_main_':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    average=sum(student_marks[query_name])/len(scores)
    print(format(average, '.2f'))
    
#Tuples   #I used pypy looking for some comeand in the discussion
if _name_ == '_main_':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    tupla1=tuple(integer_list)
    print(hash(tupla1))
    
if _name_ == '_main_':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    tupla1=tuple(integer_list)
    print(hash(tupla1))


    
##STRINGS

#sWAP cASE

def swap_case(s):
    swap=s.swapcase()
    return swap

if _name_ == '_main_':
    s = input()
    result = swap_case(s)
    print(result)
    
#String Split and Join

def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return line
    

if _name_ == '_main_':
    line = input()
    result = split_and_join(line)
    print(result)
    
#What's Your Name?
def print_full_name(first, last):
    print('Hello {} {}! You just delved into python.'. format(first, last))

if _name_ == '_main_':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)
    
#Mutations
def mutate_string(string, position, character):
    x=list(string)
    x[position]=character
    s=''.join(x)
    return s

if _name_ == '_main_':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

#Find a string
def count_substring(string, sub_string):
    ntimes=0
    for i in range(len(string)):
        if string[i:].startswith(sub_string): 
            ntimes+=1
    return ntimes

if _name_ == '_main_':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)
    
#String Validators
if _name_ == '_main_':
    s = input()
    print (any (x.isalnum() for x in s))
    print (any (x.isalpha() for x in s))
    print (any (x.isdigit() for x in s))
    print (any (x.islower() for x in s))
    print (any (x.isupper() for x in s))
    
#Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

    
#text wrap
import textwrap

def wrap(string, max_width):
    x=textwrap.wrap(string, max_width)
    x = textwrap.fill(string, max_width)
    return x

if _name_ == '_main_':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
    
#Capitalize!
def solve(s):
    x=s.split()
    for i in x:
        s=s.replace(i,i.capitalize())
    return s    

if _name_ == '_main_':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()
    
#Merge the Tools!
def merge_the_tools(string, k):
    # your code goes here
            lista = []
            nuova_lista = []
            for i in range(0, len(string), k):
                    lista.append(string[i: (i + k)])
            for j in lista:
                    for l in j:
                            if l not in nuova_lista:
                                    nuova_lista.append(l)
                    print("".join(nuova_lista))
                    nuova_lista = []

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)
    
    
##SETS

#Introduction to Sets
def average(array):
    # your code goes here
    return sum(set(array))/len(set(array))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
    
#No Idea!
n, m = input().split()

ar = input().split()

A = set(input().split())
B = set(input().split())
h=0
for i in ar:
    
    if i in A:
        h += 1
    if i in B:
        h -=1
print(h)

#Symmetric Difference
M = int(input())
A = set(map(int, input().split()))
N = int(input())
B = set(map(int, input().split()))
for i in (sorted(A.union(B)-A.intersection(B))):
    print(i)
    
#Set .add()
N=input()
s=set()
for i in range(int(N)):
    s.add(input())
print(len(s))

#Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
com= int(input())

for x in range(com):
    comandi=input().split()
    if comandi[0]=='remove':
        s.remove(int(comandi[1]))
    elif comandi[0]== 'discard':
        s.discard(int(comandi[1]))
    elif comandi[0]=='pop':
        s.pop()
        
print(sum(s))

#Set .union() Operation
n=int(input())
s=set(input().split())
m=int(input())
p=set(input().split())

print(len(s.union(p)))

#Set .intersection() Operation
n=int(input())
s=set(input().split())
m=int(input())
p=set(input().split())

print(len(s.intersection(p)))

#Set .difference() Operation
n=int(input())
s=set(input().split())
m=int(input())
p=set(input().split())

print(len(s.difference(p)))

#Set .symmetric_difference() Operation

n=int(input())
s=set(input().split())
m=int(input())
p=set(input().split())

print(len(s.union(p)- s.intersection(p)))

#The Captain's Room

k = int(input())
arr=list(map(int, input().split()))

set1=set(arr)
print(((sum(set1)*k)-(sum(arr)))//(k-1))

#Check Subset
for i in range (int(input())):
    x = input()
    A= set(input().split())
    y= input()
    B=set(input().split())
    print(B.intersection(A) == A)
    
#Check Strict Superset

A=set(input().split())
strsupset=0
n = int(input())
for i in range (n):
        B = set(input().split())
        if A.issuperset(B) :
                strsupset += 1
print(strsupset == n)


##COLLECTIONS

#collections.Counter()

X=int(input())
size= list(map(int,input().split()))
N=int(input())
res = 0
for i in range(N):
    l = list(map(int,input().split()))
    t = l[0]
    p = l[1]
    if t in size:
        res += p
        size.remove(t)
print(res)

#Collections.namedtuple()

from collections import namedtuple

N=int(input())
col=list(input().split())


res = 0
for i in range(N):
    d = input().split()
    res += int(list(d)[col.index('MARKS')])
print(res/N)


#Collections.OrderedDict()

from collections import OrderedDict

N=int(input())
x=OrderedDict()
for i in range(N):
    itemN=int(input())
x={}

for i in range(N):
    l=input()
    if l in x:
        x[l] += 1
    else:
        x[l] = 1
    
print(len(x))
for i in x:
    print(x[i], end=' ') = int(item[-1])
    item_name = " ".join(item[:-1])
    x[item_name] = x.get(item_name, 0) + int(net_price)  
    
for i,n in x.items():
    print(i, n)
    
#Word Order

N=int(input())
x={}

for i in range(N):
    l=input()
    if l in x:
        x[l] += 1
    else:
        x[l] = 1
    
print(len(x))
for i in x:
    print(x[i], end=' ')
    
#Collections.deque()

from collections import deque
N=int(input())
d=deque()
for i in range(N):
    x=input().split()
    if x[0]=='append':
        d.append(x[1])
    elif x[0]=='pop':
        d.pop()
    elif x[0]=='appendleft':
        d.appendleft(x[1])
    elif x[0]=='popleft':
        d.popleft()
print(*d)       
        
    
#Company Logo

import math
import os
import random
import re
import sys

from collections import Counter

s = Counter(input()).items()

x=sorted(s, key=lambda c: (-c[1], c[0]))[:3]
for car, n in x:
    print(car, n)

#Piling Up!

T=int(input())
for i in range(T):
    x=input()
    y=input().split()
    lista = [int(i) for i in y]
    l = len(lista)
    i = 0
    while i < l - 1 and lista[i] >= lista[i+1]:
        i += 1
    while i < l - 1 and lista[i] <= lista[i+1]:
        i += 1
    print("Yes" if i == l - 1 else "No")
    
##DATE AND TIME

#Calendar Module
import calendar
MM, DD, YYYY= map(int,input().split())
day=calendar.weekday(YYYY, MM, DD)
if day == 0:
    print('MONDAY')
elif day == 1:
    print('TUESDAY')   
elif day == 2:
    print('WEDNESDAY')
elif day == 3:
    print('THURSDAY')
elif day == 4:
    print('FRIDAY')
elif day == 5:
    print('SATURDAY')
else:
    print('SUNDAY')
    
    
##EXCEPTIONS

#Exceptions

N=int(input())

for i in range(N):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

##BUILT-INS

#Input()

x,k = map(int,input().split())
p=eval(input())
if(p==k):
    print (True)
else:
    print (False)
    
#Python Evaluation
var=input()
eval(var)

#ginortS

S=input()
l = []
u = []
o = []
e = []
for i in S:
    if i.islower():
        l.append(i)
    elif i.isupper():
        u.append(i)
    elif i.isdigit():
        if int(i)%2 == 0:
            e.append(i)
        elif int(i)%2 != 0:
            o.append(i)

a =sorted(l)+sorted(u)+sorted(o)+sorted(e)
for i in a:
    print(i,end = "")

##PHYTON FUNCTIONALS

#Map and Lambda Function

cube = lambda x:x**3 # complete the lambda function 

def fibonacci(n):
    a = 0
    b = 1
    for i in range(n):
        yield a
        a,b = b,a+b

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))
    

##REGEX AND PARSING     #Here, sometimes, i used the discussion to see how to define the pattern

#Detect Floating Point Number
T=int(input())
for i in range(T):   
    try:
        n=input()
        int(n.split('.')[1])    
        if float(n):
            print('True')
    except:        
        print('False')

#Re.split()

regex_pattern = r"[.,]"	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

#Group(), Groups() & Groupdict()

S=input()
import re
m = re.search(r'([a-z0-9])\1', S)
print(m.group(1) if m else -1)

#Re.start() & Re.end()

import re
S=input()
k=input()
m = re.search(k, S)
pattern = re.compile(k)
if not m: print("(-1, -1)")
while m:
    print("({0}, {1})".format(m.start(),m.end()-1))
    m = pattern.search(S,m.start()+1)
    
    
#Regex Substitution

N=int(input())
for i in range(N):
    x = input()
    
    while ' && ' in x or ' || ' in x:
        x = x.replace(" && ", " and ").replace(" || ", " or ")
    print(x)

#Validating Roman Numerals

regex_pattern = r"(MMM|MM|M)?(CM|DCCC|DCC|DC|D|CD|CCC|CC|C)?(XC|LXXX|LXX|LX|L|XL|XXX|XX|X)?(IX|VIII|VI|V|IV|III|II|I)?$"	# Do not delete 'r'.

import re
print(str(bool(re.match(regex_pattern, input()))))

#Validating phone numbers

import re
N=int(input())

for i in range(N):
    x=input()
    if re.match(r'[789]\d{9}',x) and len(x)==10:   
        print('YES') 
    else:  
        print('NO')  

#Validating and Parsing Email Addresses

import re
n = int(input())
for i in range(n):
    x,y = input().split(' ')
    mat = re.match(r"<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>", y)
    if mat:
        print(x,y)

#Hex Color Code
import re
N=int(input())

m=r'(#[0-9a-fA-F]{3,6})[^\n ]'
for i in range(N):
    x=input()
    for c in re.findall(m,x):
        print(c)

#Validating Credit Card Numbers
import re
N=int(input().strip())
card = re.compile(
    r"^"
    r"(?!.*(\d)(-?\1){3})"
    r"[456]"
    r"\d{3}"
    r"(?:-?\d{4}){3}"
    r"$")
for i in range(N):
    s=input().strip()
    print("Valid" if card.search(s) else "Invalid")
    
#Validating Postal Codes

regex_integer_in_range = r"^[1-9][0-9]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=.\1)"	# Do not delete 'r'.


import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

##XML

#XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    score=0
    for i in node.iter():
        score += len(i.items())
    return score

    
if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

##CLOSURES AND DECORATORS

#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        f(map(lambda x: "+91 " + x[-10:-5] + " " + x[-5:], l))
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 
    
    
#Decorators 2 - Name Directory
import operator

def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')
    
    
##NUMPY

#Arrays
mport numpy

def arrays(arr):
    a=numpy.array(arr[::-1], float)
    return a
    # complete this function
    # use numpy.array

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#Shape and Reshape

import numpy

arr=numpy.array(list(map(int,input().split())))
arr.shape=(3,3)
print(arr)

#Transpose and Flatten

import numpy

N,M=map(int, input().split())
arr=numpy.array([input().strip().split() for i in range(N)],int)
print(arr.transpose())    
print(arr.flatten())

#Concatenate

import numpy
N,M,P = map(int,input().split())
arrA = numpy.array([input().split() for i in range(N)],int)
arrB = numpy.array([input().split() for j in range(M)],int)
print(numpy.concatenate((arrA, arrB)))

#Zeros and Ones

import numpy
num = list(map(int, input().split()))
print (numpy.zeros(num, dtype = numpy.int))
print (numpy.ones(num, dtype = numpy.int))

#Array Mathematics

import numpy

N,M = map(int,input().split())

A=numpy.zeros((N,M),int)
B=numpy.zeros((N,M),int)
for i in range(N):
  A[i]=numpy.array(input().split(),int)
for j in range(N):
  B[j]=numpy.array(input().split(),int)  

print(A+B)
print(A-B)
print(A*B)
print(numpy.array(A//B,int))
print(A%B)
print(A**B)


#Floor, Ceil and Rint

import numpy

numpy.set_printoptions(legacy = '1.13')

arr = numpy.array(input().split(),float)

print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))

#Sum and Prod
import numpy

N,M = map(int,input().split())
arr = numpy.array([input().split() for i in range(N)], int)
x=(numpy.sum(arr,axis=0))
print(numpy.prod(x))

#Min and Max

import numpy

N,M = map(int,input().split())
arr = numpy.array([input().split() for i in range(N)], int)
x=(numpy.min(arr,axis=1))
print(numpy.max(x))

#Mean, Var, and Std
import numpy

N,M = map(int,input().split())
arr = numpy.array([input().split() for i in range(N)], int)
print(numpy.mean(arr, axis = 1)) 
print(numpy.var(arr, axis = 0))
print(round(numpy.std(arr, axis = None),11))

#Dot and Cross

import numpy

N=int(input())
A = numpy.array([input().split() for i in range(N)],int)
B = numpy.array([input().split() for j in range(N)],int)
x = numpy.dot(A,B)
print (x)

#Inner and Outer
import numpy

A = numpy.array(input().split(), int)
B = numpy.array(input().split(), int)
print(numpy.inner(A,B))
print(numpy.outer(A,B))


#Polynomials

import numpy

P = list(map(float,input().split()));
x = input();
print(numpy.polyval(P,float(x)));

#Linear Algebra

import numpy

N=int(input())
A=numpy.array([input().split() for i in range(N)],float)
print(round(numpy.linalg.det(A),2))


###EX.2

##Birthday Cake Candles


import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    x=max(candles)
    y= candles.count(x)
    return y
    # Write your code here

    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


##Number Line Jumps


import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    # Write your code here
    if (v1 > v2) and (x2 - x1) % (v2 - v1) == 0:
        return 'YES'
    else:
        return 'NO'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    # Write your code here
    shared=5
    cumulative=0
    for i in range(0,n):
        liked=shared//2
        shared=liked*3
        cumulative=cumulative+liked
    return cumulative
    


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
    
#Recursive Digit Sum
import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    # Write your code here
    x=str(sum([int(i) for i in n]))
    if len(n)==1:
        if k==1:
            return n
        else:
            return superDigit(n*k,1)
    else:
        return superDigit(x,k)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()
