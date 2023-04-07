import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.power(math.e, -np.sin(3*x))

def f_prim(x):
    return -3.0*np.cos(3.0*x)*np.power(math.e, -np.sin(3*x))

def h(x,i):
    return x[i + 1] - x[i]

#oblicza pierwszą różnicę dzieloną dla funkcji y na punktach i i i+1
def delta_1(x,y, i):
    return (y[i + 1] - y[i]) / h(x,i)

# oblicza drugą różnicę dzieloną dla funkcji y
def delta_2(x,y,i):
    return (delta_1(x,y,i + 1) - delta_1(x,y,i)) / (x[i + 1] - x[i - 1])

#oblicza trzecią różnicę dzieloną dla funkcji y
def delta_3(x,y, i):
    return (delta_2(x,y,i + 1) - delta_2(x,y,i)) / (x[i + 2] - x[i - 1])

#zwraca indeks i takie, że X[i] <= x < X[i+1]
def find_interval(x,X):
    n = len(X)
    l = 0
    r = n - 1
    while l <= r:
        mid = (l + r) // 2
        if x >= X[mid]:
            l = mid + 1
        else:
            r = mid - 1
    return l - 1

def Spline3(x_interpol,x, y, type):
    n = len(x)
    h_matrix = np.zeros(shape=(n, n))
    m_matrix = np.zeros(shape=(n, 1))

    for i in range(1, n - 1):
        h_matrix[i][i - 1] = h(x,i - 1)
        h_matrix[i][i] = 2 * (h(x,i - 1) + h(x,i))
        h_matrix[i][i + 1] = h(x,i)

        m_matrix[i] = delta_1(x,y,i) - delta_1(x,y,i - 1)

    if type == "cubic_spline":
        h_matrix[0][0] = -h(x,0)
        h_matrix[0][1] = h(x,0)
        h_matrix[n - 1][n - 2] = h(x,n - 2)
        h_matrix[n - 1][n - 1] = -h(x,n - 2)

        m_matrix[0] = np.power(h(x,0), 2) * delta_3(x,y,0)
        m_matrix[n - 1] = -np.power(h(x,n - 2), 2) * delta_3(x,y,n - 4)
        solver = np.linalg.solve(h_matrix, m_matrix)

    elif type == "natural_spline":
        h_matrix = h_matrix[1:-1, 1:-1]
        m_matrix = m_matrix[1:-1]
        solver = [0, *np.linalg.solve(h_matrix, m_matrix), 0]

    result = []
    for x_p in x_interpol:
        #współczynników funkcji interpolowanej w spline'ach.
        d = min(find_interval(x_p,x),n-2)
        c = (y[d + 1] - y[d]) / h(x,d) - h(x,d) * (solver[d + 1] + 2 * solver[d])
        b = 3 * solver[d]
        a = (solver[d + 1] - solver[d]) / h(x,d)
        result.append(y[d] + c * (x_p - x[d]) + b * (x_p - x[d])**2.0 + a * (x_p - x[d])**3.0)

    return result

def linspace_with_ends(start, stop, n):
    return np.linspace(start, stop, n)

def equally_spaced_3(n,type):
    # generujemy równomiernie rozłożone węzły interpolacji
    a = -math.pi
    b = 2.0 * math.pi
    x = linspace_with_ends(a,b,n)

    # obliczamy wartości funkcji f(x) w węzłach interpolacji
    y = f(x)
    # obliczamy wartości funkcji interpolującej w punktach x_interp
    x_interp = linspace_with_ends(a,b,500)
    y_interp = Spline3(x_interp, x , y ,type)

    # tworzymy wykres funkcji f(x) i funkcji interpolującej
    plt.plot(x_interp, f(x_interp), label='f(x)')
    plt.plot(x_interp, y_interp, label='f_interp(x) for ' + str(n) + " points")

    # ustawiamy ograniczenia osi, aby były widoczne kółka
    # plt.xlim(a - 0.5, b + 0.5)
    # plt.ylim(np.min(f(x_interp)) - 0.5, np.max(f(x_interp)) + 0.5)

    # dodajemy zamknięte kółka na końcach przedziału
    plt.scatter(x, y, marker='o', color='k')
    # dodajemy tytuł wykresu i etykiety osi
    plt.title('Interpolacja funkcją sklejaną III stopnia \ndla równomiernie rozłożonych węzłów')
    plt.xlabel('x')
    plt.ylabel('y')

    # dodajemy legendę
    plt.legend()

    # wyświetlamy wykres
    plt.show()
    print(y_interp)
    errors = np.abs(f(x_interp) - y_interp)
    errors_max = np.max(errors)
    absolute_error = error_function(x_interp, y_interp)
    return errors_max, absolute_error


def error_function(x_interp,y_intep):
    error = 0
    n = len(x_interp)
    for i in range(n):
        error += np.power(f(x_interp[i])-y_intep[i],2)
    error = math.sqrt(error)
    error = error/n
    return error
