import math
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.power(math.e, -np.sin(3 * x))

#zwraca tablicę równoodległych punktów pomiędzy wartościami start i stop, zawierającą n punktów
def linspace_with_ends(start, stop, n):
    return np.linspace(start, stop, n)

#definiuje interpolację kwadratową dla danych wejściowych x_wejsciowe i y_wejsciowe oraz określonego typu warunków brzegowych
def interpolacja_kwadratowa(x_wejsciowe, y_wejsciowe, typ_warunkow):
    n = len(y_wejsciowe)  # Liczba węzłów
    warunek_brzegowy = typ_warunkow
    funkcje_sklejane = []

    #funkcja zwraca wartość gamma dla i-tego węzła, która jest potrzebna do obliczenia współczynników funkcji sklejanych
    def gamma(i):
        return (y_wejsciowe[i] - y_wejsciowe[i - 1]) / (x_wejsciowe[i] - x_wejsciowe[i - 1])

    #Ta funkcja oblicza współczynniki dla funkcji sklejanych. Współczynniki te są przechowywane w listach a, b i c.
    def oblicz_wspolczynniki(a, b, c):
        for i in range(1, n):
            b.append(2 * gamma(i) - b[-1])

        for i in range(n - 1):
            a.append((b[i + 1] - b[i]) / (2 * (x_wejsciowe[i + 1] - x_wejsciowe[i])))

        return a, b, c

    #a funkcja oblicza funkcje sklejane i przechowuje je w liście funkcje_sklejane.
    def oblicz_funkcje(a, b, c):
        def s(i):
            a_i = a[i]
            b_i = b[i]
            c_i = c[i]
            return lambda x: a_i * (x - x_wejsciowe[i]) ** 2.0 + b_i * (x - x_wejsciowe[i]) + c_i

        for i in range(n - 1):
            funkcje_sklejane.append(s(i))

    if warunek_brzegowy == "natural":
        a = []
        b = [0]
        c = y_wejsciowe
        a, b, c = oblicz_wspolczynniki(a, b, c)
        oblicz_funkcje(a, b, c)
    elif warunek_brzegowy == "clamped":
        a = []
        b = [(y_wejsciowe[1] - y_wejsciowe[0]) / (x_wejsciowe[1] - x_wejsciowe[0])]
        c = y_wejsciowe
        oblicz_wspolczynniki(a, b, c)
        oblicz_funkcje(a, b, c)

    return funkcje_sklejane


def funkcja_interpolacji_kwadratowej(x_wejsciowe, y_wejsciowe, typ_warunkow, x_wyjsciowe):
    wynik = []
    n = len(y_wejsciowe)

    def szukaj_indeksu_przedzialu(x_p):
        l = 0
        r = n - 1

        while l <= r:
            m = (l + r) // 2
            if x_p >= x_wejsciowe[m]:
                l = m + 1
            else:
                r = m - 1

        return l - 1

    funkcje_sklejane = interpolacja_kwadratowa(x_wejsciowe, y_wejsciowe, typ_warunkow)

    for j in x_wyjsciowe:
        i = max(0, min(szukaj_indeksu_przedzialu(j), n - 2))
        wynik.append(funkcje_sklejane[i](j))

    return wynik


def equally_spaced_2(n, type):
    # generujemy równomiernie rozłożone węzły interpolacji
    a = -math.pi
    b = 2.0 * math.pi
    x = linspace_with_ends(a, b, n)

    # obliczamy wartości funkcji f(x) w węzłach interpolacji
    y = f(x)

    # obliczamy wartości funkcji interpolującej w punktach x_interp
    x_interp = linspace_with_ends(a, b, 500)
    y_interp = funkcja_interpolacji_kwadratowej(x, y, type, x_interp)

    # tworzymy wykres funkcji f(x) i funkcji interpolującej
    plt.plot(x_interp, f(x_interp), label='f(x)')
    plt.plot(x_interp, y_interp, label='f_interp(x) for ' + str(n) + " points")

    # ustawiamy ograniczenia osi, aby były widoczne kółka
    # plt.xlim(a - 0.5, b + 0.5)
    # plt.ylim(np.min(f(x_interp)) - 0.5, np.max(f(x_interp)) + 0.5)

    # dodajemy zamknięte kółka na końcach przedziału
    plt.scatter(x, y, marker='o', color='k')
    # dodajemy tytuł wykresu i etykiety osi
    plt.title('Interpolacja funkją sklejaną II stopnia \n dla równomiernie rozłożonych węzłów i warunkami typu: ' +type )
    plt.xlabel('x')
    plt.ylabel('y')

    # dodajemy legendę
    plt.legend()

    # wyświetlamy wykres
    plt.show()
    errors = np.abs(f(x_interp) - y_interp)
    errors_max = np.max(errors)
    absolute_error = error_function(x_interp, y_interp)
    return errors_max, absolute_error



def error_function(x_interp,y_intep):
    error = 0
    n = len(x_interp)
    for i in range(n):
        error += np.power(f(x_interp[i]-y_intep[i]),2)
    error = math.sqrt(error)
    error = error/n
    return error
