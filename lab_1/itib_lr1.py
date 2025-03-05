import matplotlib.pyplot as plt
from itertools import product, combinations
import math


def work_func(f):
    print("Таблица истинности булевой функции f = x1 v -(x2) v -(x3 v x4)")
    print("+----+----+----+----+----+")
    print("| x1 | x2 | x3 | x4 | F  |")
    print("+----+----+----+----+----+")
    for x1, x2, x3, x4 in product(*[range(2) for _ in range(4)]):
        res = int(((not x3 or x4) and (not x1)) or x2)
        print(f"| {x1}  | {x2}  | {x3}  | {x4}  | {res}  |")
        f.append(res)
    print("+----+----+----+----+----+")


def y1(net):
    return 1 if net >= 0 else 0


def y2(net):
    return 1 if (1 / (1 + math.exp(-net))) >= 0.5 else 0


def delta_y1(n, delta, x, _):
    return n * delta * x


def delta_y2(n, delta, x, net):
    temp = 1 / (1 + math.exp(-net))
    return n * delta * temp * (1 - temp) * x


def plot_error(error, name):
    plt.plot(error[1:], 'ro-', linewidth=2, markersize=5)
    plt.grid(True)
    plt.title(f"График суммарной ошибки {name}")
    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.show()


def hemming(f1, f2):
    distance = 0
    for i in range(len(f1)):
        distance += f1[i] ^ f2[i]
    return distance


def calculate_func_weight(x_array, y, weights):
    f_real = []
    for X in x_array:
        net = sum([x * w for x, w in zip(X, weights[1:])]) + weights[0]
        f_real.append(y(net))
    return f_real


def training(f, y, delta_w, n):
    x_array = list(product(*[range(2) for _ in range(4)]))
    weights = 5 * [0]
    error_steps = [1]
    epoch = 0
    print("Эпоха".rjust(5), "Значения".rjust(18), "Вектор весов".rjust(40), "Расстояние Хэмминга".rjust(20))

    while error_steps[epoch]:
        f_real = calculate_func_weight(x_array, y, weights)
        error_steps.append(hemming(f, f_real))
        y_string = ''.join([str(_) for _ in f_real])
        w_string = ', '.join([str("%.3f" % it) if it < 0 else str("%.4f" % it) for it in weights])
        print(str(epoch).rjust(5), str(y_string).rjust(18), str(w_string).rjust(40),
              str(error_steps[epoch+1]).rjust(3))
        epoch += 1
        if error_steps[epoch] == 0:
            break
        for X, f_x in zip(x_array, f):
            net = sum([x * w for x, w in zip(X, weights[1:])]) + weights[0]
            delta = f_x - y(net)
            weights[0] += delta_w(n, delta, 1, net)
            for i in range(1, len(weights)):
                weights[i] += delta_w(n, delta, X[i-1], net)

    name = "пороговой ФА" if y == y1 else "логистической ФА"
    plot_error(error_steps, name)


def learn_min(f, y, delta_w, n, epoch_max):
    x_array = list(product(*[range(2) for _ in range(4)]))

    for num in range(1, 5):
        for index_comb in list(combinations([_ for _ in range(16)], num)):
            weights = 5 * [0]
            error_steps = [1]
            epoch = 0
            f_temp = [f[_] for _ in index_comb]
            x_array_temp = [x_array[_] for _ in index_comb]
            print("\nЭпоха".rjust(5), "Значения".rjust(18), "Вектор весов".rjust(40), "Расстояние Хэмминга".rjust(20))

            while error_steps[epoch] and epoch <= epoch_max:
                f_real = calculate_func_weight(x_array, y, weights)
                error_steps.append(hemming(f, f_real))
                y_string = ''.join([str(_) for _ in f_real])
                w_string = ', '.join([str("%.3f" % it) if it < 0 else str("%.4f" % it) for it in weights])
                print(str(epoch).rjust(5), str(y_string).rjust(18), str(w_string).rjust(40),
                      str(error_steps[epoch+1]).rjust(3))
                epoch += 1
                if error_steps[epoch] == 0:
                    break
                for X, f_x in zip(x_array_temp, f_temp):
                    net = sum([x * w for x, w in zip(X, weights[1:])]) + weights[0]
                    delta = f_x - y(net)
                    weights[0] += delta_w(n, delta, 1, net)
                    for i in range(1, len(weights)):
                        weights[i] += delta_w(n, delta, X[i - 1], net)

            if error_steps[epoch] == 0:
                plot_error(error_steps, "логистической ФА на минимальной обучающей выборке")
                print("Выбранные наборы")
                print("+----+----+----+----+----+")
                print("| x1 | x2 | x3 | x4 | F  |")
                print("+----+----+----+----+----+")
                for X, f_x in zip(x_array_temp, f_temp):
                    print(f"| {X[0]}  | {X[1]}  | {X[2]}  | {X[3]}  | {f_x}  |")
                print("+----+----+----+----+----+")
                return


F = []
work_func(F)
print("\n\nПороговая ФА\n")
training(F, y1, delta_y1, 0.3)
print("\n\nЛогистическая ФА\n")
training(F, y2, delta_y2, 0.3)
print("\n\nЛогистическая ФА с минимально возможной обучающей выборкой")
learn_min(F, y2, delta_y2, 0.3, 25)