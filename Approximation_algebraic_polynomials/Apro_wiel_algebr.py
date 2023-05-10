import math
import numpy as np
import pandas as pd
from tabulate import tabulate

def f(x):
    """Funkcja f(x), którą chcemy aproksymować."""
    return np.power(math.e, -np.sin(3*x))

def s_k(x_p,k):
    """Funkcja pomocnicza, obliczająca sumę potęg elementów z tablicy x_p do k-tej potęgi."""
    result = np.power(x_p,k)
    suma = sum(result)
    return suma

def t_k(x_p,y_p,k):
    """Funkcja pomocnicza, obliczająca sumę iloczynów elementów z tablic x_p i y_p, podniesionych do k-tej potęgi."""
    suma = 0
    result  = np.power(x_p,k)
    for i in range(len(result)):
        suma += y_p[i]*result[i]
    return suma

def Aproksymacja(x_p,y_p,x,m):
    """Funkcja aproksymująca wielomianem o stopniu m."""
    s_matrix = np.zeros(shape=(m+1, m+1))
    # Tworzenie macierzy s_matrix o wymiarach (m+1) x (m+1) i wypełnienie jej sumami potęg x_p
    for i in range(m+1):
        for j in range(m+1):
            s_matrix[i][j] = s_k(x_p,i+j)

    t_matrix = np.zeros(shape=(m+1))
    # Tworzenie tablicy t_matrix o długości m+1 i wypełnienie jej sumami iloczynów x_p i y_p podniesionych do potęg od 0 do m
    for i in range(m+1):
        t_matrix[i] = t_k(x_p,y_p,i)

    # Rozwiązanie macierzowego równania otrzymanego z wykorzystaniem metody najmniejszych kwadratów
    result = np.linalg.linalg.solve(s_matrix, t_matrix)

    y_out = []
    # Dla każdego elementu x w tablicy x oblicz wartość aproksymacji wielomianowej z wykorzystaniem otrzymanych współczynników
    for x_out in x:
        k_cur=0
        y_cur=0
        # Oblicz wartość wielomianu w punkcie x_out
        for a in result:
            y_cur+=a*(x_out**k_cur)
            k_cur+=1
        # Dodaj wartość wielomianu do tablicy y_out
        y_out.append(y_cur)
    return y_out

def calc_error(F, f_p, x_p):
    """Funkcja obliczająca błędy aproksymacji."""
    F_real = F(x_p)
    diffs = np.abs(F_real - f_p)
    return {
        'max': max(diffs),
        'sq': np.sqrt(sum(x ** 2 for x in diffs))
    }


def calculate_errors(n_min, n_max, m_min, m_max):
    ns = range(n_min, n_max + 1, 5)  # Number of nodes
    ms = range(m_min, m_max + 1)  # Degree of the polynomial
    ws = [[(1,) * n for _ in range(len(list(ms)))] for n in ns]  # Give equal weights to all points

    # Stworzenie pustej tabeli z wartościami błedu aproksymacji
    results_table = [[None for _ in range(len(ms) + 1)] for _ in range(len(ns) + 1)]

    # Uzupełnienie pierwszego wiersza tabeli wartościami m
    results_table[0][1:] = ms

    # Stworzenie pustej tabeli z bledu przybliżenia wartościami aproksymacji
    error_table = [[None for _ in range(len(ms) + 1)] for _ in range(len(ns) + 1)]

    # Uzupełnienie pierwszego wiersza tabeli wartościami m
    error_table[0][1:] = ms

    # Uzupełnienie pierwszej kolumny tabeli wartościami n
    for i, n in enumerate(ns):
        results_table[i + 1][0] = n
        error_table[i+1][0] = n
    # Uzupełnienie tabeli wartościami aproksymacji
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            x = np.linspace(-math.pi, 2.0 * math.pi, n, endpoint=True)
            y = f(x)
            x_p = np.linspace(-math.pi, 2.0 * math.pi, 1000, endpoint=True)
            y_p = Aproksymacja(x, y, x_p, m)
            error = calc_error(f, y_p, x_p)
            results_table[i + 1][j + 1] = error['max']
            error_table[i + 1][j + 1] = error['sq']

    # stworzenie DataFrame z wynikami aproksymacji
    df = pd.DataFrame(results_table, columns=['n/m'] + [f'm={m}' for m in ms])
    df_err = pd.DataFrame(error_table, columns=['n/m'] + [f'm={m}' for m in ms])

    # zapisanie DataFrame do pliku Excel
    df.to_excel('wyniki_aproksymacji.xlsx', index=False)
    df_err.to_excel('wyniki_error_aproksymacji.xlsx', index=False)

