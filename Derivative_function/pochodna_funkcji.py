import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill


def f_precision(x):#funkcja poczatkowa
    return np.sin(x) + np.cos(3 * x)



def f_der(x):#wartość pochodnej
    return math.cos(x) - 3 * math.sin(3 * x)



def approx_derivative(x, h, precision):#przybliżona wartość pochodnej
    return precision((f(x + h) - f(x)) / h)


def result_function_h_plot():#funkcja bez rzutowania na precyzję
    n = [i for i in range(0, 40)]
    x = 1
    results = []
    for i in range(0, n[-1] + 1):
        h = 2.0 ^ (-i)
        results.append(f_prim(x, h))

    plt.plot(n, results)
    plt.xlabel("N")
    plt.ylabel("Result f'(x)")

    plt.show()
    return


def result_function_h_float64():#funkcja obliczająca wartość h i przybliżenie pochodnej funkcji dla  x= 1 w zmienncych typu double
    n = [i for i in range(0, 41)]
    x = np.float64(1)
    results = []
    for i in range(0, n[-1] + 1):
        h = np.float64(2.0 ** (-i))
        approx_derivative = np.float64((f(x + h) - f(x)) / h)
        results.append(approx_derivative)
        print(f"h = {h:.40f}, approx_der = {approx_derivative:.15f}")

    plt.scatter(n, results)
    plt.xlabel("N")
    plt.ylabel("Result f'(x)")

    plt.show()
    return


def result_function_h_float32():#funkcja obliczająca wartość h i przybliżenie pochodnej funkcji dla  x= 1 w zmienncych typu float
    n = [i for i in range(0, 41)]
    x = np.float32(1)
    h_value = []
    results = []
    for i in range(0, n[-1] + 1):
        h = np.float32(2.0 ** (-i))
        h_value.append(h)
        approx_derivative = np.float32((f_precision(x + h) - f_precision(x)) / h)
        results.append(approx_derivative)

    plt.scatter(h_value, results)
    plt.xscale('log')
    plt.xlabel('h')
    plt.ylabel('Approximate derivative')
    plt.show()

    # Utworzenie tabeli pandas z wynikami
    df = pd.DataFrame({'n': n, 'h': h_value, 'result': results})

    # Dodanie kolorów do tabeli
    cmap = 'viridis'
    df_styled = df.style.background_gradient(cmap=cmap)

    # Zapisanie tabeli do pliku csv
    df.to_csv('results.csv', index=False, float_format='%.10f')
    df_styled.to_excel('results_styled.xlsx', index=False, float_format='%.10f')

    return


def result_function_h_double():#funkcja obliczająca wartość h i przybliżenie pochodnej funkcji dla  x= 1 w zmienncych typu double
    n = [i for i in range(0, 41)]
    x = np.double(1)
    h_value = []
    results = []
    for i in range(0, n[-1] + 1):
        h = np.double((2.0 ** (-i)))
        h_value.append(h)
        approx_derivative = np.double((f_precision(x + h) - f_precision(x)) / h)
        results.append(approx_derivative)

    plt.scatter(h_value, results)
    plt.xlabel('h')
    plt.xscale('log')
    plt.ylabel('Approximate derivative')
    plt.show()

    # Utworzenie tabeli pandas z wynikami
    df = pd.DataFrame({'n': n, 'h': h_value, 'result': results})

    # Dodanie kolorów do tabeli
    cmap = 'viridis'
    df_styled = df.style.background_gradient(cmap=cmap)

    # Zapisanie tabeli do pliku csv
    df.to_csv('results_h_double.csv', index=False, float_format='%.10f')
    df_styled.to_excel('results_styled_h_double.xlsx', index=False, float_format='%.10f')

    return

def result_function_h_longdouble():#funkcja obliczająca wartość h i przybliżenie pochodnej funkcji dla  x= 1 w zmienncych typu long double
    n = [i for i in range(0, 41)]
    x = np.longdouble(1)
    h_value = []
    results = []
    for i in range(0, n[-1] + 1):
        h = np.longdouble((2.0 ** (-i)))
        h_value.append(h)
        approx_derivative = np.longdouble((f_precision(x + h) - f_precision(x)) / h)
        results.append(approx_derivative)

    plt.scatter(h_value, results)
    plt.xlabel('h')
    plt.xscale('log')
    plt.ylabel('Approximate derivative')
    plt.show()

    # Utworzenie tabeli pandas z wynikami
    df = pd.DataFrame({'n': n, 'h': h_value, 'result': results})

    # Dodanie kolorów do tabeli
    cmap = 'viridis'
    df_styled = df.style.background_gradient(cmap=cmap)

    # Zapisanie tabeli do pliku csv
    df.to_csv('results_h_longdouble.csv', index=False, float_format='%.10f')
    df_styled.to_excel('results_styled_h_longdouble.xlsx', index=False, float_format='%.10f')

    return



def error_in_result():#funkcja obliczająca wartość h i przybliżenie pochodnej funkcji w punkcie x = 1
                    # oraz różnice  w zmienncych różnego typu dla prawdziwej wartości pochodnej
    def compute_results(precision_type):
        if precision_type == 'float':
            precision = np.float32
            file_name = 'results_float.csv'
        elif precision_type == 'double':
            precision = np.double
            file_name = 'results_double.csv'
        elif precision_type == 'longdouble':
            precision = np.longdouble
            file_name = 'results_longdouble.csv'
        else:
            raise ValueError("Invalid precision type. Choose from 'float', 'double', or 'longdouble'.")


        approx_deriv = approx_derivative(x, h, precision)
        true_deriv = precision(np.cos(x) - 3 * np.sin(3 * x))

        results = pd.DataFrame({
            'h': h.astype(precision),
            'approx_derivative': approx_deriv,
            'true_derivative': true_deriv,
            'abs_error': np.abs(approx_deriv - true_deriv).astype(precision),
            'rel_error': np.abs((approx_deriv - true_deriv) / true_deriv)
        })

        fig, ax = plt.subplots()

        ax.loglog(h,np.abs(approx_deriv - true_deriv).astype(precision) , '.', label =  file_name)
        ax.set_xlabel('h')
        ax.set_ylabel('Absolute Error')
        ax.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim([1e-10, 1])
        plt.show()

        results.to_csv(file_name, index=False,float_format='%.10f')

        return results

    # Compute results for different precision types
    results_float = compute_results('float')
    results_double = compute_results('double')
    results_longdouble = compute_results('longdouble')

    # Plot absolute error for each precision type
    fig, ax = plt.subplots()

    ax.loglog(results_float['h'], results_float['abs_error'], '.', label='float')
    ax.loglog(results_longdouble['h'], results_longdouble['abs_error'], '.', label='longdouble')
    ax.loglog(results_double['h'], results_double['abs_error'], '.', label='double')
    ax.set_xlabel('h')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-10, 1])
    plt.show()



def f(x):
    return np.sin(x) + np.cos(3 * x)


def result_float64():#wypisanie wartości dla double, funkcja pomocnicza
    x = np.float64(1)

    for n in range(41):
        h = np.float64(2 ** (-n))
        approx_derivative = np.float64((f(x + h) - f(x)) / h)
        print(f"n={n}, h_float64={h:.40f}, approx_derivative_float64={approx_derivative:.40f}")


def result_float32():#wypisanie wartości dla float, funkcja pomocnicza
    x = np.float32(1)

    for n in range(41):
        h = np.float32(2 ** (-n))
        approx_derivative = np.float32((f(x + h) - f(x)) / h)
        print(f"n={n}, h_float32={h:.40f}, approx_derivative_float32={approx_derivative:.40f}")


def calculate_difference_double():#funkcja obliczająca wartość h i przybliżenie pochodnej funkcji dla  x= 1 w zmienncych typu double
    h_value = []
    dif = []
    for n in range(41):
        h = np.float64(2.0 ** (-n))
        h_plus_one = np.float64(h + 1.0)
        h_value.append(h_plus_one)
        difference = np.float64(f_precision(h_plus_one + 1.0) - f_precision(1))
        dif.append(difference)


    df = pd.DataFrame({'h+1': h_value, 'f`(h+1) - f`(1)': dif})
    df.to_excel('results_double.xlsx', index=False, float_format='%.10f')

def calculate_difference_longdouble():
    h_value = []
    dif = []
    for n in range(41):
        h = np.longdouble(2.0 ** (-n))
        h_plus_one = np.longdouble(h + 1.0)
        h_value.append(h_plus_one)
        difference = np.longdouble(f_precision(h_plus_one + 1.0) - f_precision(1))
        dif.append(difference)


    df = pd.DataFrame({'h+1': h_value, 'f`(h+1) - f`(1)': dif})
    df.to_excel('results_longdouble.xlsx', index=False, float_format='%.10f')

def calculate_difference_float():
    h_value = []
    dif = []
    for n in range(41):
        h = np.float32(2.0 ** (-n))
        h_plus_one = np.float32(h + 1.0)
        h_value.append(h_plus_one)
        difference = np.float32(f_precision(h_plus_one + 1.0) - f_precision(1))
        dif.append(difference)


    df = pd.DataFrame({'h+1': h_value, 'f`(h+1) - f`(1)': dif})
    df.to_excel('results_float_h.xlsx', index=False, float_format='%.10f')

calculate_difference_float()
calculate_difference_double()
calculate_difference_longdouble()