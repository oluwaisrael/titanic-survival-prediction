import numpy as np

# Your first NumPy array
scores = np.array([85, 90, 78, 92, 88])

print(scores)
print(type(scores))
print(scores + 5)
print(scores * 2)
print(np.mean(scores))
print(np.median(scores))
# 2D array - like a table of data
data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

print(data)
# Grabbing specific values
print(data[0])     
print(data[1][2])   
print(data[2, 1])   

students = np.array([[2, 3,60],
                     [5, 3, 88],
                     [9, 3, 92]])

print(students.shape)
print(np.mean(students))
print(students[1])