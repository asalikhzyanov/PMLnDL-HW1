import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin, cos, sqrt, atan2, radians, exp
import random


def read_csv(filepath):
    global coordinates
    df = pd.read_csv(filepath)
    df['population'] = df['population'].astype(int)
    sorted_cities = df.sort_values(by=['population'], ascending=False)[:30]
    coordinates = {}
    for index, row in sorted_cities.iterrows():
        key = row['address'].split('г ')[1]
        coordinates[key] = (row['geo_lat'], row['geo_lon'])


def compute_distance(source, destination):
    r = 6373.0
    source_latitude = radians(source[0])
    source_longitude = radians(source[1])
    destination_latitude = radians(destination[0])
    destination_longitude = radians(destination[1])
    d_latitude = destination_latitude - source_latitude
    d_longitude = destination_longitude - source_longitude

    a = sin(d_latitude / 2) ** 2 + cos(source_latitude) * cos(destination_latitude) * sin(d_longitude / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def get_energy(path):
    energy = 0
    for i in range(len(path)):
        if i == len(path) - 1:
            energy += compute_distance(coordinates[path[i]], coordinates[path[0]])
        else:
            energy += compute_distance(coordinates[path[i]], coordinates[path[i + 1]])
    return energy


def simulated_annealing(annealing_rate=.001, t_init=1000, t_thresh=1, max_iter=10000):
    global config

    def acceptance_probability(energy, new_energy, temp):
        if new_energy < energy:
            return 1
        return exp((energy - new_energy) / temp)

    best_ = list(coordinates.keys())
    current = best_.copy()
    t = t_init
    energies_ = []
    i = 0
    while t > t_thresh and i < max_iter:
        new = current.copy()
        pos1 = random.randint(0, len(coordinates) - 1)
        pos2 = random.randint(0, len(coordinates) - 1)
        tmp = new[pos1]
        new[pos1] = new[pos2]
        new[pos2] = tmp

        if acceptance_probability(get_energy(current), get_energy(new), t) > random.random():
            current = new

        if get_energy(current) < get_energy(best_):
            best_ = current
        energies_.append(get_energy(best_))
        if i % 25 == 0:
            config.append([t, energies_[-1], best_])
        t *= (1 - annealing_rate)
        i += 1
    config.append([t, energies_[-1], best_])
    return best_, energies_


def plot_(energies_):
    plt.xlabel('iterations')
    plt.ylabel('energy')
    plt.plot(energies_)
    plt.grid()
    plt.show()


def animate(i):
    ax.clear()
    cities = coordinates.keys()
    y = [coordinates[city][1] for city in cities]
    x = [coordinates[city][0] for city in cities]
    ax.scatter(x, y)
    for j, txt in enumerate(cities):
        ax.annotate(txt, (x[j], y[j]))
    cities = config[i][2]
    y = [coordinates[city][1] for city in cities]
    x = [coordinates[city][0] for city in cities]
    plt.title('T={}, energy={}'.format(round(config[i][0], 2), round(config[i][1], 2)))
    plt.plot(x, y)
    plt.axis('off')


coordinates = {}
config = []
read_csv('cities.csv')
best, energies = simulated_annealing()
plot_(energies)

print('Start creating gif...')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig.set_size_inches(10, 5)
ani = animation.FuncAnimation(fig, animate, frames=len(config), interval=200)
ani.save('SimulatedAnnealing.gif', writer='imagemagick')
print('Done!')
