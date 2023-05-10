import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from openpyxl.workbook import Workbook
from Apro_wiel_algebr import f, Aproksymacja, calculate_errors

# Funkcja wykreslana
def main(n):
    # Generujemy n punktów równoodległych w przedziale od -pi do 2pi
    x = np.linspace(-math.pi, 2.0 * math.pi, 80, endpoint=True) # tu możemy zmienić ilość punktów
    # Wartości funkcji w wygenerowanych punktach
    y = f(x)
    # Generujemy 1000 punktów równoodległych w tym samym przedziale
    x_p = np.linspace(-math.pi, 2.0 * math.pi, 1000, endpoint=True)
    # Dokonujemy aproksymacji wartości funkcji w punktach x_p
    y_p = Aproksymacja(x, y, x_p, n)

    # Rysujemy wykresy funkcji oryginalnej i aproksymującej
    plt.plot(x_p, f(x_p), label='Wyjściowa funkcja')
    plt.plot(x_p, y_p, label='Funkcja aproksymująca')
    # Dodajemy punkty (x, y) na wykres
    plt.scatter(x, y, marker='o', color='k')
    # Dodajemy tytuł wykresu i etykiety osi
    plt.title("Funkcja aproksymcji stopnia " + str(n) + " przy 40 węzłach")
    plt.xlabel('x')
    plt.ylabel('y')
    # Obliczamy różnicę między wartościami funkcji aproksymującej i oryginalnej
    diff = [abs(y_p[i] - f(x_p[i])) for i in range(len(y_p))]

    # Dodajemy legendę do wykresu
    plt.legend()

    # Wyświetlamy wykres
    plt.show()

    # Zwracamy maksymalną wartość różnicy oraz pierwiastek sumy kwadratów różnic
    return (max(diff), math.sqrt(sum(x ** 2 for x in diff)))

if __name__ == '__main__':
    # Testujemy funkcję dla różnych stopni wielomianu
    k = [4, 6, 8, 10, 12] # tu możemy zmienić stopnie wielomianów
    nearly = []
    error_max = []
    for m in k:
        tmp = main(m)
        error_max.append(tmp[0])
        nearly.append(tmp[1])

    # Tworzymy DataFrame z wynikami
    df = pd.DataFrame({'Stopień wielomianu': k, 'Wzór XI': error_max, 'Wzór XII': nearly})
    eror = calculate_errors(10, 100, 3, 20)

    # Zapisujemy DataFrame do pliku Excel z nazwą zawierającą wartość n
    df.to_excel(f'nowe_dane_aproksymacji_80.xlsx')
