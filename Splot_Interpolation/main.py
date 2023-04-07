import pandas as pd


from quadratic_spline import equally_spaced_2
from three_spline import equally_spaced_3
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    k = [5,7,9,10,15,20,30,50]
    error_max = []
    nearly = []
    for i in k:
        tmp = equally_spaced_3(i,"natural_spline")
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Liczba węzłów': k, 'max(|f(x_interep)-y_interep|)': error_max, 'wzór IV': nearly})

    # zapisujemy DataFrame do pliku CSV z nazwą zawierającą wartość n
    df.to_excel(f'new_interp_3splot_n_natural.xlsx')

    k = [5, 7, 9, 10, 15, 20, 30, 50]
    error_max = []
    nearly = []
    for i in k:
        tmp = equally_spaced_3(i, "cubic_spline")
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Liczba węzłów': k, 'max(|f(x_interep)-y_interep|)': error_max, 'wzór IV': nearly})

    # zapisujemy DataFrame do pliku CSV z nazwą zawierającą wartość n
    df.to_excel(f'new_interp_3splot_n_cubic.xlsx')

    k = [5, 7, 9, 10, 15, 20, 30, 50]
    error_max = []
    nearly = []
    for i in k:
        tmp = equally_spaced_2(i,'clamped')
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Liczba węzłów': k, 'max(|f(x_interep)-y_interep|)': error_max, 'wzór IV': nearly})

    # zapisujemy DataFrame do pliku CSV z nazwą zawierającą wartość n
    df.to_excel(f'new_interp_2splot_n_clamped.xlsx')

    k = [5, 7, 9, 10, 15, 20, 30, 50]
    error_max = []
    nearly = []
    for i in k:
        tmp = equally_spaced_2(i, 'natural')
        error_max.append(tmp[0])
        nearly.append(tmp[1])
    df = pd.DataFrame({'Liczba węzłów': k, 'max(|f(x_interep)-y_interep|)': error_max, 'wzór IV': nearly})

    # zapisujemy DataFrame do pliku CSV z nazwą zawierającą wartość n
    df.to_excel(f'new_interp_2splot_n_natural.xlsx')