import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x):
    return np.power(math.e, -np.sin(3*x))

def f_prim(x):
    return -3.0*np.cos(3.0*x)*np.power(math.e, -np.sin(3*x))


def Hermite_function(x,y,x_point):
    n = len(x)
    Macierz_wart = [[0 for col in range(2*n+1)] for row in range(2*n+1)] #tworzę macierz wypełnioną zerami

    for i in range(0, 2 * n, 2) :# w metodzie interpolacji Hermite'a stosuje podwójne węzły
        Macierz_wart[i][0] = x[i // 2]
        Macierz_wart[i + 1][0] = x[i // 2]
        Macierz_wart[i][1] = y[i // 2]
        Macierz_wart[i + 1][1] = y[i // 2]

    for i in range(2, 2 * n + 1):
        for j in range(1 + (i - 2), 2 * n):
            if i == 2 and j % 2 == 1:
                Macierz_wart[j][i] = f_prim(x[j // 2])
            else:
                Macierz_wart[j][i] = (Macierz_wart[j][i - 1] - Macierz_wart[j - 1][i - 1]) / (Macierz_wart[j][0] - Macierz_wart[(j - 1) - (i - 2)][0])

    def function_def(x_point):
        result = 0
        for i in range(0, 2 * n):
            curr = 1.0
            j = 0
            while j < i:
                curr *= (x_point - x[j // 2])
                if j + 1 != i:
                    curr *= (x_point - x[j // 2])
                    j += 1
                j += 1
            result += curr * Macierz_wart[i][i + 1]
        return result


    return function_def(x_point)


def chebyshev_nodes(a, b, n):
    k = np.arange(1, n+1)
    x_k = (0.5*(a+b) + 0.5*(b-a)*np.cos((2.0*k-1)*np.pi/(2.0*n)))
    return x_k



def linspace_with_ends(start, stop, n):
    return np.linspace(start, stop, n, endpoint=True)


def chebyshev_inter(n):
    # generujemy węzły interpolacyjne zgodnie z zerami wielomianu Czebyszewa
    a = -np.pi
    b = 2.0 * np.pi
    x = chebyshev_nodes(a, b, n)

    # obliczamy wartości funkcji f(x) w węzłach interpolacyjnych
    y = f(x)

    # interpolujemy funkcję f(x) z użyciem interpolacji Hermite'a
    x_interp = chebyshev_nodes(a,b,500)
    y_interp = Hermite_function(x,y, x_interp)

    # tworzymy wykres funkcji f(x) oraz jej interpolacji
    plt.plot(x_interp, f(x_interp), label='f(x)')
    plt.plot(x_interp, y_interp, label='f_interp(x) for ' + str(n) + " points")

    # dodajemy zamknięte kółka na końcach przedziału
    plt.scatter(x, y, marker='o', color='k')

    # dodajemy tytuł wykresu i etykiety osi
    plt.title('Interpolacja Hermite\'a dla węzłów rozłożonych \n zgodnie z zerami wielomianu Czebyszewa')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    plt.show()

    # obliczamy błędy interpolacji dla każdego punktu x_interp
    errors = np.abs(f(x_interp) - y_interp)
    errors_max = max(errors)
    absolute_error = error_function(x_interp,y_interp)
    return errors_max,absolute_error

def error_function(x_interp,y_intep):
    error = 0
    n = len(x_interp)
    for i in range(n):
        error += np.power(f(x_interp[i])-y_intep[i],2)
    error = math.sqrt(error)
    error = error/n
    return error

#obliczamy wartości funkcji interpolującej w węzłach xi z użyciem interpolacji Hermite'a
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
    y_interp = Hermite_function(x,y,x_interp)

    # tworzymy wykres funkcji f(x) i funkcji interpolującej
    plt.plot(x_interp, f(x_interp), label='f(x)')
    plt.plot(x_interp, y_interp, label='f_interp(x) for ' + str(n) + " points")

    # ustawiamy ograniczenia osi, aby były widoczne kółka
    # plt.xlim(a - 0.5, b + 0.5)
    # plt.ylim(np.min(f(x_interp)) - 0.5, np.max(f(x_interp)) + 0.5)

    # dodajemy zamknięte kółka na końcach przedziału
    plt.scatter(x, y, marker='o', color='k')
    # dodajemy tytuł wykresu i etykiety osi
    plt.title('Interpolacja Hermite\'a dla równomiernie rozłożonych węzłów')
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


if __name__ == '__main__':
    k = [5,7, 9, 11, 13, 15,50,80]
    error_max = []
    nearly = []
    for i in k:
        tmp = chebyshev_inter(i)
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Liczba węzłów': k, 'max(|f(x_interep)-y_interep|)': error_max, 'wzór IV': nearly})

    # zapisujemy DataFrame do pliku CSV z nazwą zawierającą wartość n
    df.to_excel(f'new_interp_hermite_n_Czebyszew.xlsx')

    error_max = []
    nearly = []
    for i in k:
        tmp = equally_spaced(i)
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Liczba węzłów': k, 'max(|f(x_interep)-y_interep|)': error_max, 'wzór IV': nearly})

    # zapisujemy DataFrame do pliku CSV z nazwą zawierającą wartość n
    df.to_excel(f'new_interp_hermite_n_EqualySpaced.xlsx')



