# This is a sample Python script.
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from openpyxl.workbook import Workbook

from Aproksymacja_Trygonometryczna import Aproksymacja_Trygonometryczna, f, calculate_errors


def main(n):
    # Generujemy n równoodległych punktów z przedziału [-pi, 2pi], włączając oba końce.
    x = np.linspace(-math.pi, 2.0 * math.pi, n, endpoint=True)#tu można zmienić liczbę węzłów na inną
    # Obliczamy wartości funkcji dla wygenerowanych punktów.
    y = f(x)
    # Generujemy 1000 równoodległych punktów z przedziału [-pi, 2pi], włączając oba końce.
    x_p = np.linspace(-math.pi, 2.0 * math.pi,1000, endpoint=True)
    # Aproksymujemy funkcję na podstawie wygenerowanych punktów x i y, z użyciem funkcji Aproksymacja().
    # Stopień aproksymacji wynosi 5.
    y_p = Aproksymacja_Trygonometryczna(x, y,12,x_p)#tu zamiast 12 możemy zmienić stopień

    # Rysujemy wykres oryginalnej funkcji.
    plt.plot(x_p, f(x_p), label='Wyjściowa funkcja')
    # Rysujemy wykres funkcji aproksymującej.
    plt.plot(x_p, y_p, label='Funkcja aproksymująca')
    # Dodajemy zamknięte kółka na końcach przedziału.
    plt.scatter(x, y, marker='o', color='k')
    # Dodajemy tytuł wykresu i etykiety osi.
    plt.title("Funkcja aproksymcji stopnia 12 przy " + str(n) + " węzłach")
    plt.xlabel('x')
    plt.ylabel('y')
    # Obliczamy różnicę między wartościami aproksymowanej funkcji i oryginalnej funkcji.
    diff = [abs(y_p[i] - f(x_p[i])) for i in range(len(y_p))]

    # Dodajemy legendę.
    plt.legend()

    # Wyświetlamy wykres.
    plt.show()

    # Zwracamy wartości: maksymalny błąd i pierwiastek błędu kwadratowego.
    return (max(diff), math.sqrt(sum(x ** 2 for x in diff)))

    # Jeśli ten plik jest uruchamiany jako skrypt, a nie moduł,
    # to wykonaj następujące instrukcje.
if __name__ == '__main__':
    # Określamy liczbę węzłów n, dla których będziemy generować wykresy.
    k = [10,20,25,30]
    nearly = []
    error_max = []
    # Dla każdej wartości m generujemy wykres i obliczamy wartości błędów.
    for m in k:
        tmp = main(m)
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Stopień wielomianu': k, 'Wzór X': error_max, 'Wzór XI': nearly})
    # zapisujemy DataFrame do pliku CSV z wielomianem stopnia m = 12
    df.to_excel(f'new_data_aproks_12_m.xlsx')