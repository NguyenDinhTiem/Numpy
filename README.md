# Numpy in Python

>- Numpy là một **thư viện** của python. Đây là thư viện **cốt lõi** cho scientific computing. Bao gồm các tác vụ xử lý như: mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation, v.v...
>- Sử dụng Numpy, các toán tử toán học và logic có thể thực hiện được. 
![](https://i.imgur.com/79EqLFQ.png)


# Cài đặt
> ###      pip install numpy


```python
!pip install numpy
```

# Khai báo trong mã nguồn
> ### import numpy as np


```python
import numpy as np
```

# Tạo Numpy từ mảng một chiều
> ### Từ List
>> Cú pháp: arr_np = numpy.array(list)


```python
# code examples
# tạo ndarray từ list

import numpy as np

# tạo list
l = list(range(1, 4))

# tạo ndarray
data = np.array(l)

print(data)
print(data[0])
print(data[1])
```

# Các thuộc tính thông dụng của Numpy
> ### dtype: loại dữ liệu mong muốn
> ### shape: trả về một tuple với index có một số phần tử tương ứng
> ### ndim: chỉ định kích thước tối thiểu của mảng kết quả


```python
# code example với shape là 1D
# tạo ndarray từ list

import numpy as np

# tạo list
list1D = [1, 2, 3]

# tạo ndarray
data = np.array(list1D)

print(data)
print(data.shape)
print(data.ndim)
print(data.dtype)
```

--> Mảng một chiều (ndim=1) có 3 phần tử, có kiểu dữ liệu là int64


```python
# code example với shape là 2D
# tạo ndarray từ list

import numpy as np

# tạo list
list2D = [[1., 2.], [3., 4.], [5., 6.]]

# tạo ndarray
data = np.array(list2D)

print(data)
print(data.shape)
print(data.ndim)
print(data.dtype)
```

--> Mảng hai chiều (ndim=2): có 3 dòng và 2 cột, có kiểu dữ liệu là float64


```python
# code example với shape là 3D
# tạo ndarray từ list

import numpy as np

# tạo list
list3D = [ [[1, 2], [3, 4], [5, 6]],
           [[1, 6], [2, 2], [3, 4]],
           [[7, 7], [8, 2], [9, 5]] ]

# tạo ndarray
data = np.array(list3D, dtype=np.int32)

print(data)
print(data.shape)
print(data.ndim)
print(data.dtype)
```

--> Mảng ba chiều (ndim=3): chứa 3 ma trận mà một ma trận có 3 dòng và 2 cột, có kiểu dữ liệu là int32

# Các thao tác trên Numpy array



> ### Cập nhật phần tử


```python
# code example
# thay đổi giá trị phần tử

import numpy as np

# tạo list
l = list(range(1, 4))

# tạo ndarray
data = np.array(l)
print(data)

# Cập nhật phần tử trong mảng
data[0] = 8
print(data)
```

> ### Thêm phần tử


```python
# Thêm vào cuối
data = np.arange(10)
data = np.append(data, [25, 30])
print(data)
```

> ### Xóa phần tử



```python
# Xóa phần tử tại vị trí index
print(data)
data = np.delete(data, 10)
print(data)
```

> ### Số phần tử trong Numpy array


```python
print(data.shape[0])
```

> ### Tạo mảng có giá trị mặc định 0, 1


```python
# code examples
# dùng hàm zeros() function
# với tất cả phần tử là 0

arr = np.zeros((2, 3))
print(arr)
```


```python
# code examples
# dùng hàm ones() function
# với tất cả phần tử là 1

arr = np.zeros((2, 3))
print(arr)
```


```python
# code examples
# dùng hàm full() function
# với tất cả phần tử là hằng số fill_value

arr = np.full((2, 3), 9)
print(arr)
```

> ### Thay đổi kích cỡ Numpy array


```python
# code examples
# thay đổi kích cỡ mảng dùng hàm reshape()
import numpy as np

# tạo list
l = [[1, 2, 3],
     [4, 5, 6]]

# tạo ndarray
data = np.array(l)
print("data\n", data)
print("data shape\n", data.shape)
print("\n")
# thay đổi kích cỡ
data_rs = np.reshape(data, (3,2))
print("data_rs\n", data_rs)
print("data_rs shape\n", data_rs.shape)
```

> ### Các hàm liên quan đến dữ liệu


```python
### hàm where() ###
import numpy as np

# tạo array
arr = np.arange(5)
print(arr)

#condition - lấy những phần tử < 3
condition = arr < 3

# if (condition) là true, trả về arr, ngược lại trả về arr*2
out = np.where(condition, arr, arr*2)

print(condition)
print(out)
```


```python
### hàm flatten() ###
import numpy as np

# tạo array
arr = np.array([[1, 2], [3, 4]])

# làm phẳng thành mảng
out = arr.flatten()

print("Before:")
print(arr)

print("After flatten:")
print(out)
```

> # Chỉ mục trong Numpy Array

>> ## Kỹ thuật dùng Slicing
>>> ### Cú pháp: array[for_axis_0, for_axis_1, ...]
>>> ### ":" lấy tất cả các dòng và cột
>>> ### "a:b" lấy tất các phần tử từ vị trí a đến vị trí b


```python
# code examples
import numpy as np

# khởi tạo numpy array a_arr
a_arr = np.array([[1, 2, 3], [5, 6, 7]])

# Sử dụng kỹ thuật slicing để tạo mảng b_arr
# lấy tất cả các dòng và cột 1,2
b_arr = a_arr[:, 1:3]

print("a_arr:")
print(a_arr)
print("\n")
print("b_arr:")
print(b_arr)
```


```python
### truy xuất lấy theo dòng ###
import numpy as np

# tạo một numpy array có shape(3,3) với giá trị
# [ [1, 2, 3],
#   [5, 6, 7],
#   [9, 10, 11] ]

arr = np.array([[1, 2, 3],
                [5, 6, 7],
                [9, 10, 11]])

# trường hợp 1: truy xuất theo số chiều giảm
row_m1 = arr[1, :]

# trường hợp 2: truy xuất theo chiều được giữ nguyên
row_m2 = arr[1:2, :]

print(row_m1, row_m1.shape)
print(row_m2, row_m2.shape)
```

<li> trường hợp 1: output là mảng 1 chiều có 3 phần tử <br>
<li> trường hợp 2: output là ma trận có 1 dòng và 3 cột


```python
### truy xuất lấy theo cột ###
import numpy as np

# tạo một numpy array có shape(3,3) với giá trị
# [ [1, 2, 3],
#   [5, 6, 7],
#   [9, 10, 11] ]

arr = np.array([[1, 2, 3],
                [5, 6, 7],
                [9, 10, 11]])

# trường hợp 1: truy xuất theo số chiều giảm
row_m1 = arr[:, 1]

# trường hợp 2: truy xuất theo chiều được giữ nguyên
row_m2 = arr[:, 1:2]

print(row_m1, row_m1.shape)
print(row_m2, row_m2.shape)
```

<li> trường hợp 1: output là mảng 1 chiều có 3 phần tử <br>
<li> trường hợp 2: output là ma trận có 1 cột và 3 dòng


```python
### Sử dụng Lists như Index ###
import numpy as np

# tạo arr
arr = np.array([[1, 2],
                [3, 4],
                [5, 6]])

# truy cập giá trị (0,0), (1,2), (2,0)
output = arr[[0, 1, 2], [0, 1, 0]]

print(output)
```


```python
### Sử dụng Boolean như Index ###
import numpy as np

# tạo arr
arr = np.array([[1, 2],
                [3, 4],
                [5, 6]])

print(arr)

# tìm các phần tử lớn hơn 2
bool_idx = (arr > 2)

print(bool_idx)
```

> # Toán tử thông dụng trong Numpy Array

>> ### Toán tử  "$+$"


```python
# code examples
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

print("data x \n", x)
print("data x \n", y)

# Tổng của 2 mảng
print("method 1 \n", x + y)
print("method 2 \n", np.add(x, y))
```

>> ### Toán tử  "$-$"


```python
# code examples
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

print("data x \n", x)
print("data x \n", y)

# Hiệu của 2 mảng
print("method 1 \n", x - y)
print("method 2 \n", np.subtract(x, y))
```

>> ### Toán tử  "$*$"


```python
# code examples
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

print("data x \n", x)
print("data x \n", y)

# Tích của 2 mảng
print("method 1 \n", x * y)
print("method 2 \n", np.multiply(x, y))
```

>> ### Toán tử  "$/$"


```python
# code examples
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

print("data x \n", x)
print("data x \n", y)

# Thương của 2 mảng
print("method 1 \n", y / x)
print("method 2 \n", y // x)
print("method 3 \n", np.divide(y, x))
```

>> ### Căn bậc hai


```python
# code examples
import numpy as np

data = np.array([1, 2, 3, 4])

print("data \n", data)

# Căn bậc 2 của từng phần tử trong data
print("sqrt: \n", np.sqrt(data))
```

>> ### Inner product
>>> #### Hàm này trả về kết quả tích vô hướng của các véc-tơ, scalar.


```python
# codes examples
import numpy as np 

print("Inner product:")
print(np.inner(np.array([1,2,3]),np.array([0,1,0])))
```

Inner product = 1$*$0 $+$ 2$*$1 $+$ 3$*$0


```python
# Codes examples
# Multi-dimensional array example 
import numpy as np 
a = np.array([[1,2], [3,4]]) 

print("Array a:")
print(a) 

print("\n")
b = np.array([[11, 12], [13, 14]]) 
print("Array b:")
print(b) 

print("\n")

print("Inner product:")
print(np.inner(a,b))
```

Inner product = <br>
[[1$*$11 $+$ 2$*$12, 1$*$13 $+$ 2$*$14]<br> 
[3$*$11 $+$ 4$*$12, 3$*$13 $+$ 4$*$14]]


```python
# Codes examples
import numpy as np

v = np.array([1, 2])
w = np.array([2, 3])

# Inner product giữa u và v
print("method 1:", v.dot(w))
print("method 2:", np.dot(v,w))
```


```python
# Codes examples
# Inner product giữa Matrix với Véc-tơ
import numpy as np

X = np.array([[1, 2],
              [3, 4]])

v = np.array([1, 2])

print("matrix X \n", X)
print("matrix Y \n", v)

# phép nhân giữa ma trận với vector
print("method 1: X.dot(v) \n", X.dot(v))
print("method 2: v.dot(X) \n", v.dot(X))
```


```python
# Codes examples
# Inner product giữa Matrix với Matrix
import numpy as np

X = np.array([[1, 2],
              [3, 4]])

Y = np.array([[1, 2],
             [2, 1]])

print("matrix X \n", X)
print("matrix Y \n", v)

# phép nhân giữa ma trận với vector
print("method 1: X.dot(Y) \n", X.dot(Y))
print("method 2: Y.dot(X) \n", Y.dot(X))
```

>> ### Tranpose
>>> ### Thực hiện việc chuyển đổi dòng thành cột, và cột thành dòng trong ma trận



```python
# code examples
# Transpose matrix
import numpy as np

X = np.array([[1, 2],
              [3, 4]])

print("Before Transpose:")
print(X)

# Transpose
print("After Transpose:")
print(X.T)
```

# Cơ chế Broadcasting
>> ### Là việc thực hiện các phép toán số học giữa các numpy array có kích thước khác nhau. Trong đó, thường là các trường hợp một mảng có kích thước nhỏ và mảng có kích thước lớn hơn, nhưng muốn thực hiện các phép toán giữa chúng. Lúc này, mảng nhỏ hơn sẽ tự nhân bản sao cho có cùng shape với mảng lớn hơn. Khi đó, các phép toán được thực thi bình thường.


```python
# code examples
# Vector và một scalar
import numpy as np

# create data array
data = np.array([1, 2, 3])
print("Cho 1 mảng:", data)

# scalar
factor = 2
print("giá trị scalar:", factor)

# broadcasting
result_mul = data * factor
result_minus = data - factor

print("Kết quả tích giữa vec-tor và scalar:", result_mul)
print("Kết quả hiệu giữa vec-tor và scalar:", result_minus)
```


```python
# code examples
# Matric và một vector
import numpy as np

# create data array
matrix = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [10, 11, 12]])
print("Cho 1 matrix:\n", matrix)

# vector
vector = np.array([1, 0, 1])
print("vector:", vector)

# broadcasting
Y = matrix + vector

print("Kết quả cộng giữa matrix và vector:\n", Y)
```

Bài viết dựa trên tài liệu khóa học AI Foundation Course in AI VietNam
