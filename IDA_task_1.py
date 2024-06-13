import numpy as np
import matplotlib.pyplot as plt

def get_polinoma(a: np.array, t: range) -> np.array:
    N = len(t)    # t - array of argument value
    M = len(a)    # a - list, tuple or array of ideal model's existing parametres
    y = np.zeros(N, dtype=float)    # numpy array of N-zeros
    for value_n in range(N):
        y[value_n] = a[0]    # creates an 'y' array according of N-range values
    if M == 0:
        return y    # returns new 'y' np-array if a-seguence(np.array) equal zero
    for value_n in range(N):
        t_pow = 1    # t_pow - 
        for value_m in range(1, M):
            t_pow *= t[value_n]
            y[value_n] += a[value_m] * t_pow    # creates y-np.array according existing values of  t_pow & a-sequence(np.array)
    
    return y


if __name__ == '__main__':
    GN = 3
    SN = 3
    a = np.ones(3)
    a[0] = 10 + 2 * GN
    a[1] = 0.01 + 0.002 * SN
    a[2] = 0.005 + 0.0001 * SN
    print(f'\n>>> Теоритичні коефіцієнти ідеальної моделі:\n{a}')
    M = len(a)
    # print(M)
    N = 100
    t = range(1, N+1)    # include value 100
    y = get_polinoma(a, t)
    # print(y)
    # print(y.shape)
    noise_level = 1    # set noise level
    y_max = np.max(abs(y))    # max value of ideal feature
    # print(y_max)
    sigma = y_max * noise_level / M    #standard deviation for noise component
    print('----------------------')
    print(f'\n>>> noise level = {noise_level:0.5e}, y_max={y_max:0.5e}, sigma = {sigma:0.5e}')
    np.random.seed(7319014)
    noise = np.random.normal(0, sigma, size=N)
    y_exp = y + noise
    # print()
    # print(y_exp)
    a_appr = np.polyfit(t, y_exp, M-1)
    print(f'\n>>> МНК-оцінки коефіцієнтів:\n {a_appr}')
    p = np.poly1d(a_appr)    # creates a polynom view
    print(f'\n>>> Оцінка полінома матиме вигляд:\n{p}')
    a_appr = a_appr[::-1]
    print(f'\n>>> Отримані коєфіцієнти у порядку, де індекс коєфіцієнта дорівнює ступеню аргумента\n{a_appr}')

# significant decline in the coefficient estimates
    d_a = np.zeros(M)
    for value_m in range(M):
        d_a[value_m] = abs(a[value_m] - a_appr[value_m]) / abs(a[value_m])
    print(f'\n>>> Відності похибки коефіцієнтів\n{d_a}')
    
    # receive values of identified model
    y_appr = get_polinoma(a_appr, t)
    # print(y_appr)
    dy = y - y_appr
    mid_dy = np.mean(dy)
    max_dy = np.max(np.abs(dy))    #receive max value
    disp_dy = np.var(dy)    # dispersion of care
    stdev_dy = np.sqrt(disp_dy)
    print(f'\n>>> Характеристики відхілення \n    ідентифікованої функції від ідеальної:')
    print(f'    середнє відхилення = {mid_dy:0.5e}, максимальне відхилення = {max_dy:0.5e}')
    print(f'дисперсія відхилення =  {disp_dy:0.5e}, СКО (стандартне) відхилення = {stdev_dy:0.5e}')

    '''diagrams'''
    plt.figure(figsize=(9, 4))
    plt.plot(t, y, 'b-', label='Ідеальна модель')
    plt.plot(t, y_exp, 'r--', label='Зашумлені дані')
    plt.plot(t, y_appr, 'g-', label='Ідентифікована модель')
    plt.xlabel('Значення аргументу t')
    plt.ylabel('Значення функцій y(t)')
    plt.legend()
    plt.grid(True)
    plt.show()
