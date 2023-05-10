import math
import numpy as np
import pandas as pd


def f(x):
    """Funkcja f(x), którą chcemy aproksymować."""
    return np.power(math.e, -np.sin(3*x))

def x_mutation(x):
    return (2.0 / 3.0) * x - (1.0 / 3.0) * math.pi

def Aproksymacja_Trygonometryczna(x, y, m, x_apr):


    def a_mut(k: int) -> float:
        return 2 / len(x) * sum(y[i] * math.cos(k * x[i]) for i in range(len(x)))

    def b_mut(k: int) -> float:
        return 2 / len(x) * sum(y[i] * math.sin(k * x[i]) for i in range(len(x)))

    x = list(map(x_mutation, x))
    a_k = list(map(a_mut, range(m + 1)))
    b_k = list(map(b_mut, range(m + 1)))

    def f_p(x):
        x_a = x_mutation(x)
        return .5 * a_k[0] + sum(a_k[k] * math.cos(k * x_a) + b_k[k] * math.sin(k * x_a) for k in range(1, m))

    y_pr = []
    for x_i in x_apr:
        y_pr.append(f_p(x_i))
    return y_pr

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
            if m < math.floor((n-1)/2):
                x = np.linspace(-math.pi, 2.0 * math.pi, n, endpoint=True)
                y = f(x)
                x_p = np.linspace(-math.pi, 2.0 * math.pi, 1000, endpoint=True)
                y_p = Aproksymacja_Trygonometryczna(x, y, m,x_p)
                error = calc_error(f, y_p, x_p)
                results_table[i + 1][j + 1] = error['max']
                error_table[i + 1][j + 1] = error['sq']

    # stworzenie DataFrame z wynikami aproksymacji
    df = pd.DataFrame(results_table, columns=['n/m'] + [f'm={m}' for m in ms])
    df_err = pd.DataFrame(error_table, columns=['n/m'] + [f'm={m}' for m in ms])

    # zapisanie DataFrame do pliku Excel
    df.to_excel('wyniki_aprokso.xlsx', index=False)
    df_err.to_excel('wyniki_error_aprokso.xlsx', index=False)

