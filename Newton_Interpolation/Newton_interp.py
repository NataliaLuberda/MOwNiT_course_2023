import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# definiujemy funkcję f(x)


def f(x):
    return np.power(math.e, -np.sin(3*x))

def print_function():
    # generujemy wartości x w przedziale <-π,2π>
    x = np.linspace(-np.pi, 2 * np.pi, 1000,endpoint= True)

    # obliczamy wartości funkcji f(x) dla każdego x
    y = f(x)

    # tworzymy wykres funkcji f(x)
    plt.plot(x, y)

    # ustawiamy ograniczenia osi, aby były widoczne kółka
    plt.xlim(-np.pi - 0.5, 2 * np.pi + 0.5)
    plt.ylim(np.min(y) - 0.5, np.max(y) + 0.5)

    # dodajemy zamknięte kółka na końcach przedziału
    plt.scatter([-np.pi, 2 * np.pi], [f(-np.pi), f(2 * np.pi)], marker='o', color='k')

    # dodajemy tytuł wykresu i etykiety osi
    plt.xlabel('x')
    plt.ylabel('y')

    # wyświetlamy wykres
    plt.show()

#obliczamy wartości funkcji interpolującej w węzłach xi z użyciem interpolacji Lagrange'a
#węzły równomiernie rozłożone
def equally_spaced(n):
    # generujemy równomiernie rozłożone węzły interpolacji
    a = -math.pi
    b = 2.0 * math.pi
    x = linspace_with_ends(a,b,n)

    # obliczamy wartości funkcji f(x) w węzłach interpolacji
    y = f(x)

    # obliczamy wartości funkcji interpolującej w punktach x_interp
    x_interp = linspace_with_ends(a,b,500)
    y_interp = newton_poly(y,x,x_interp)

    # tworzymy wykres funkcji f(x) i funkcji interpolującej
    plt.plot(x_interp, f(x_interp), label='f(x)')
    plt.plot(x_interp, y_interp, label='f_interp(x) for ' + str(n) + " points")

    # ustawiamy ograniczenia osi, aby były widoczne kółka
    # plt.xlim(a - 0.5, b + 0.5)
    # plt.ylim(np.min(f(x_interp)) - 0.5, np.max(f(x_interp)) + 0.5)

    # dodajemy zamknięte kółka na końcach przedziału
    plt.scatter(x, y, marker='o', color='k')
    # dodajemy tytuł wykresu i etykiety osi
    plt.title('Interpolacja Newton\'a dla równomiernie rozłożonych węzłów')
    plt.xlabel('x')
    plt.ylabel('y')

    # dodajemy legendę
    plt.legend()

    # wyświetlamy wykres
    plt.show()

    # obliczamy błędy interpolacji dla każdego punktu x_interp
    errors = np.abs(f(x_interp) - y_interp)
    errors_max = max(errors)
    absolute_error = error_function(x_interp, y_interp)
    return errors_max, absolute_error


def chebyshev_inter(n):
    # generujemy węzły interpolacyjne zgodnie z zerami wielomianu Czebyszewa
    a = -np.pi
    b = 2.0 * np.pi
    x = chebyshev_nodes(a, b, n)

    # obliczamy wartości funkcji f(x) w węzłach interpolacyjnych
    y = f(x)

    # interpolujemy funkcję f(x) z użyciem interpolacji Lagrange'a
    x_interp = linspace_with_ends(a,b,500)
    y_interp = newton_poly(y,x, x_interp)

    # tworzymy wykres funkcji f(x) oraz jej interpolacji
    plt.plot(x_interp, f(x_interp), label='f(x)')
    plt.plot(x_interp, y_interp, label='f_interp(x) for ' + str(n) + " points")

    # dodajemy zamknięte kółka na końcach przedziału
    plt.scatter(x, y, marker='o', color='k')

    # dodajemy tytuł wykresu i etykiety osi
    plt.title('Interpolacja Newton\'a dla węzłów rozłożonych \n zgodnie z zerami wielomianu Czebyszewa')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    plt.show()

    # obliczamy błędy interpolacji dla każdego punktu x_interp
    errors = np.abs(f(x_interp) - y_interp)
    errors_max = max(errors)
    absolute_error = error_function(x_interp,y_interp)
    return errors_max,absolute_error


def newton_poly(y,x, x_interp):
    n = len(x)
    m = len(x_interp)
    p = np.zeros(m)
    div_diff = [y]

    for j in range(1, n):
        diff = []
        for i in range(n - j):
            resum = (div_diff[j - 1][i + 1] - div_diff[j - 1][i]) / (x[i + j] - x[i])
            diff.append(resum)
        div_diff.append(diff)
    for k in range(m):
        result = y[0]
        for j in range(1, n):
            term = div_diff[j][0]
            for i in range(j):
                term *= (x_interp[k] - x[i])
            result += term
        p[k] = result
    return p

def linspace_with_ends(start, stop, n):
    return np.linspace(start, stop, n, endpoint=True)

def chebyshev_nodes(a, b, n):
    k = np.arange(1, n+1)
    x_k = (0.5*(a+b) + 0.5*(b-a)*np.cos((2.0*k-1)*np.pi/(2.0*n)))
    return x_k

def error_function(x_interp,y_intep):
    error = 0
    n = len(x_interp)
    for i in range(n):
        error += np.power(f(x_interp[i]-y_intep[i]),2)
    error = math.sqrt(error)
    error = error/n
    return error

def Newton_interp():
    k = [7, 8, 9, 10, 11, 12, 15, 20,30,40,50]
    error_max = []
    nearly = []
    for i in k:
        tmp = chebyshev_inter(i)
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Liczba węzłów': k, 'max(|f(x_interep)-y_interep|)': error_max, 'wzór IV': nearly})

    # zapisujemy DataFrame do pliku CSV z nazwą zawierającą wartość n
    df.to_excel(f'new_interp_Newton_n_Czebyszew.xlsx')

    error_max = []
    nearly = []
    for i in k:
        tmp = equally_spaced(i)
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Liczba węzłów': k, 'max(|f(x_interep)-y_interep|)': error_max, 'wzór IV': nearly})

    # zapisujemy DataFrame do pliku CSV z nazwą zawierającą wartość n
    df.to_excel(f'new_interp_Newton_n_EqualySpaced.xlsx')