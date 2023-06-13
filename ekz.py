#Сортировки
#Быстрая:
def quicksort(nums, fst, lst):
    i, j = fst, lst
    pivot = nums[randint(fst, lst)]
    while i <= j:
        while nums[i] < pivot:
            i += 1
        while nums[j] > pivot:
            j -= 1
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
            i, j = i + 1, j - 1
    if fst < j:
        quicksort(nums, fst, j)
    if i < lst:
        quicksort(nums, i, lst)
    return nums
# без рекурсии
def quick_sort(nums_2, fst, lst):
    if len(nums_2) <= 1:
        return nums_2
    stack = [(0, len(nums_2) - 1)]
    # пока стек не пуст
    while stack:
        fst, lst = stack.pop()
        i, j = fst, lst
        pivot = nums_2[randint(fst, lst)]
        while i <= j:
            while nums_2[i] < pivot:
                i += 1
            while nums_2[j] > pivot:
                j -= 1
            if i <= j:
                nums_2[i], nums_2[j] = nums_2[j], nums_2[i]
                i, j = i + 1, j - 1
        if fst < j:
            stack.append((fst, j))
        if i < lst:
            stack.append((i, lst))
    return nums_2
#Пузырьковая(Обменом)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
#Вставками
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
#Выбором
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr
#Подсчётом
def counting_sort(arr):
    n = len(arr)
    min_val = arr[0]
    max_val = arr[0]
    for i in range(1, n):
        if arr[i] < min_val:
            min_val = arr[i]
        if arr[i] > max_val:
            max_val = arr[i]
    m = max_val - min_val + 1

    x = [0] * m
    for i in arr:
        x[i - min_val] += 1

    j = 0
    for i in range(m):
        while x[i] != 0:
            arr[j] = i + min_val
            j += 1
            x[i] -= 1
    return arr
#Слиянием
def merge_sort(arr):
    if len(arr) > 1:
        left_arr = arr[:len(arr)//2]
        right_arr = arr[len(arr)//2:]

        merge_sort(left_arr)
        merge_sort(right_arr)

        i = 0 # left_arr index
        j = 0 # right_arr index
        k = 0 # merge array index
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] < right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            k += 1
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1

        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1
    return arr
#Поиск
#Бинарный
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
def binary_search_recursion(nums_2, low, high, n2):
    if low <= high:
        mid = (low + high) // 2
        if nums_2[mid] == n2:
            return mid
        elif nums_2[mid] > n2:
            return binary_search_recursion(nums_2, low, mid - 1, n2)
        else:
            return binary_search_recursion(nums_2, mid + 1, high, n2)
def binary_search(nums, n):
    low, high = 0, len(nums) - 1
    while low <= high:
        # Определение середины
        mid = (low + high) // 2
        if nums[mid] < n:
            low = mid + 1
        elif nums[mid] > n:
            high = mid - 1
        else:
            return mid
#Линейный
def linear_search(nums, n):
    for i in range(len(nums)):
        if nums[i] == n:
            return i
    return None
#Рекурсия
#Евклид
def euclidean(a, b):
    while b != 0:
        remainder = a % b
        a = b
        b = remainder
    return a
def euclidean_recursive(a, b):
    if b == 0:
        return a
    else:
        return euclidean_recursive(b, a % b)
a = int(input("Введите число a: "))
b = int(input("Введите чилсло b: "))
print(f"НОД чисел {a} и {b} = {euclidean_recursive(a, b)}")
#Факториал:
def fact(n):
    if n == 1:
        return 1
    return fact(n - 1) * n
print(fact(35))
#	Перевод в двоичную:
def perevod(n):
        if n == 0:
            return '0'
        binary = ''
        while n > 0:
            binary = str(n % 2) + binary
            n //= 2
        return binary
def perevod_recursion(n):
    if n == 0:
        return '0'
    elif n == 1:
        return '1'
    else:
        return perevod_recursion(n // 2) + str(n % 2)
#Палиндром:
def palindrom(s):
    if len(s) <= 1:
        return True
    if s[0] != s[-1]:
        return False
    else:
        return palindrom(s[1:-1])
#Фибоначчи:
def fib(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)
def fib_cash(n):
    if n not in cash:
        cash[n] = fib_cash(n-1) + fib_cash(n - 2)
    return cash[n]
cash = {0: 1, 1: 1}
print(fib_cash(900))
#Разбиение целого числа на слагаемое
def find(m, n = 1):
    if m == 0:
        return 1
    if n > m:
        return 0
    return find(m-n, n) + find(m, n+1)
m = int(input("Введите натуральное число: "))
print(find(m))
#Количество простых чисел в числе
def isprim(n):
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n and n % d != 0:
        d += 2
    return d * d > n

def pi(x):
    if x < 2:
        return 0
    else:
        return pi(x - 1) + int(isprim(x))
x = int(input(""))
print("кол-во простых чисел",pi(x))
#Проверка числа на простоту
def prostoe(n,j):
    if(n<2):
        return False
    if(j==n):
        return True
    if(n % j == 0):
        return False
    return prostoe(n,j+1)
n=int(input())
print(prostoe(n,2))
#Списки
#Append:
class Node:
    def __init__(self):
        self.head: Node
        self.tail: Node

def append(self, value: int):
    node = Node(value)
    if not self.head:
        self.head = self.tail = node
        return
    self.tail.next = node
    self.tail = node
#Push:
class Node:
    def __init__(self):
        self.head: Node
        self.tail: Node

def push(self, value: int):
    if not self.head:
        self.head = self.tail = Node(value)
        return
    node = Node(value)
    node.next = self.head
    self.head = node
#Del_Node:
class Node:
    def __init__(self):
        self.head: Node
        self.tail: Node

def delete_node(self, node: Node, pred: Node):
    if node is not None and pred is not None:
        if node.right is not None and node.left is not None:
            node.data = self._pop_max(node.left, node)
        else:
            if node.left is not None:
                if pred.left is node:
                    pred.left = node.left

                else:
                    pred.right = node.left

            else:
                if pred.left is node:
                    pred.left = node.right
                else:
                    pred.right = node.right

            del node
#pop:
def pop(self):
    if not self.head:
        return None
    value = self.head.data
    if self.head == self.tail:
        self.head = self.tail = None
        return value
    self.head = self.head.next
    return value
#class Node:
    def __init__(self, data: str):
        self.data: str = data
        self.next: Node | None = None
        self.prev: Node | None = None

    def __repr__(self):
        return f'Data = {self.data}\n' \
                f'Next = {None if not self.next else self.next.data} \n'

class LinkedList:

    def __init__(self):
        self.head: Node | None = None
        self.tail: Node | None = None
def main():

    #очередь
    list = LinkedList()
    val = int(input())
    while val != 0:
        list.append(val)
        val = int(input())

    list.print()
    list.del_all_list()

    random_list = LinkedList()
    for random_value in range(10):
        random_list.append(r.randint(-10, 10))
    random_list.print()
    print('очередь')

    #стек
    S = LinkedList()
    val = int(input())
    while val != 0:
        S.push(val)
        val = int(input())
    print('стек')
    S.print()
#Динамика
#Рюкзачок без восстановления
def knapsack(weights, prices, x):
    n = len(weights)
    dp = [[0] * (x + 1) for i in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(x, weights[i - 1] - 1, -1):
            if weights[i - 1] <= j:
                dp[i][j] = dp[i - 1][j - weights[i - 1]] + prices[i - 1]
                if dp[i][j] < dp[i - 1][j]:
                    dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j]
    return dp[n][x]


weights = [1, 10, 1, 1, 2, 2, 1, 2, 1, 1, 2, 5, 2]
prices = [10, 10, 1, 10, 7, 2, 8, 8, 9, 7, 1, 2, 1]
x = int(input("Введите вместимость: "))
print("Максимальная стоимость рюкзака:", knapsack(weights, prices, x))
#Рюкзачок с восстановлением
def knapsack(items, capacity):
    table = [[0 for j in range(capacity + 1)] for i in range(len(items) + 1)]
    selections = [[[] for j in range(capacity + 1)] for i in range(len(items) + 1)]
    for i in range(1, len(items) + 1):
        for j in range(1, capacity + 1):
            weight = items[i - 1][0]
            value = items[i - 1][1]
            if weight > j:
                table[i][j] = table[i - 1][j]
                selections[i][j] = selections[i - 1][j]
            else:
                without_item = table[i - 1][j]
                with_item = table[i - 1][j - weight] + value
                if with_item > without_item:
                 table[i][j] = with_item
                 selections[i][j] = selections[i - 1][j - weight]+ [i]
                else:
                    table[i][j] = without_item
                    selections[i][j] = selections[i - 1][j]
    return table[-1][-1], selections[-1][-1]

capacity = 20
items = [(1, 10), (10, 10), (1, 1), (1, 10), (2, 7), (2, 2), (1, 8), (2, 8), (1, 9), (1, 7), (2, 1), (5, 2), (2, 1)]
result, selections = knapsack(items, capacity)
print("Максимальная стоимость:", result)
print("Выбранные предметы:", selections)
#Король
def find_optimal_path(Price):
    N = len(Price)
    optimal_cost = [[0 for j in range(N)] for i in range(N)]
    optimal_cost[0][0] = Price[0][0]
    for i in range(1, N):
        optimal_cost[i][0] = Price[i][0] + optimal_cost[i - 1][0]
        optimal_cost[0][i] = Price[0][i] + optimal_cost[0][i - 1]
    for i in range(1, N):
        for j in range(1, N):
            optimal_cost[i][j] = Price[i][j] + min(optimal_cost[i - 1][j], optimal_cost[i][j - 1], optimal_cost[i - 1][j - 1])
    path = []
    i, j = N - 1, N - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if optimal_cost[i - 1][j] < optimal_cost[i][j - 1] and optimal_cost[i - 1][j] < optimal_cost[i - 1][j - 1]:
                i -= 1
            elif optimal_cost[i][j - 1] < optimal_cost[i - 1][j] and optimal_cost[i][j - 1] < optimal_cost[i - 1][j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
    path.append((0, 0))
    path = path[::-1]
    return optimal_cost[N - 1][N - 1], path
Price = [[6, 1, 4, 5, 7],
        [0, 4, 3, 8, 0],
        [1, 0, 4, 1, 8],
        [8, 5, 3, 0, 4],
        [4, 5, 1, 2, 3]]
optimal_cost, optimal_path = find_optimal_path(Price)
print("Минимальная стоимость пути:", optimal_cost)
print("Оптимальный путь:", optimal_path)
#Кузнечик
N = 10 + 1
A = [0] * N
# k = +1, +2, +3
A[0] = 1
A[1] = 1
A[2] = 2
for i in range(3, N):
    A[i] = A[i - 1] + A[i - 2] + A[i - 3]
print(A)
print('до 10 столбика ', A[-1], 'кол-во маршрутов')
k = 3  # int(input())
B = [0] * N
B[0] = 1
for i in range(1, k + 1):
    B[i] = 0
    for j in range(i - 1, -1, -1):
        B[i] += B[j]
for i in range(k + 1, N):
    B[i] = 0
    for j in range(1, k + 1):
        B[i] += B[i - j]
print(B)
C = [0] * N
C[0] = 1
for i in range(1, N):
    C[i] = 0
    for j in range(i - 1, max(-1, i - k - 1), -1):
        C[i] += C[j]
print(C)
Frog = [1] * N
# Frog[5] = 0
Frog[7] = 0
D = [0] * N
D[0] = 1
for i in range(1, N):
    D[i] = 0
    for j in range(i - 1, max(-1, i - k - 1), -1):
        D[i] += D[j]
    D[i] *= Frog[i]
print(D)
F = [0] * N
F[0] = 1
for i in range(1, N):
    F[i] = 0
    if Frog[i]:
        for j in range(i - 1, max(-1, i - k - 1), -1):
            F[i] += F[j]

print(F)
#ПСП
def check_brackets():
    s = input('Введите комбинацию скобок: ')
    stack = []
    final = True
    for i in s:
        if i in '({[':
            stack.append(i)
        elif i in ')}]':
            if not stack:
                final = False
                break
            open_b = stack.pop()
            if open_b == '(' and i == ')':
                continue
            if open_b == '{' and i == '}':
                continue
            if open_b == '[' and i == ']':
                continue
            final = False
            break
    if final and len(stack) == 0:
        print('YES')
    else:
        print('NO')

check_brackets()
#Лесенка
def find_min(n):
    cost = []
    for i in range(n):
        print("Введите стоимость", i + 1, "ступеньки: ")
        cost.append(int(input()))

    dp = [0] * n
    dp[0] = cost[0]
    dp[1] = cost[1]
    for i in range(2, n):
        dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
    return dp[-1]
print("Наименьшая стоимость прохода по лесенке: ", find_min(n=int(input("Введите количество ступенек: "))))
