import copy
import numpy as np
from matplotlib import pyplot as plt

import rosenbrock
import sphere
import rastrigin
import quartic

class Individuo:
    def __init__(self, solucion, velocidad):
        self._solucion = solucion
        self._velocidad = velocidad
        self._b = copy.deepcopy(solucion)
        self._b_fitness = np.inf

class PSO:
    def __init__(self,
    cantidad_individuos,
    dimensiones,
    ro, #Tamaño de vecindad
    phi1_max,
    phi2_max,
    v_max,
    problema,
    generaciones):
        self._cantidad_individuos = cantidad_individuos
        self._dimensiones = dimensiones
        self._ro = ro
        self._phi1_max = phi1_max
        self._phi2_max = phi2_max
        self._v_max = v_max
        self._problema = problema
        self._generaciones = generaciones
        self._individuos = []
        self._rango = self._problema.MAX_VALUE - self._problema.MIN_VALUE
        self._mejor = np.inf

    def crearIndividuos(self):
        for i in range(self._cantidad_individuos):
            solucion = np.random.random(size = self._dimensiones) * self._rango + self._problema.MIN_VALUE
            velocidad = np.random.random(size = self._dimensiones) * self._v_max * 2 + self._v_max
            individuo = Individuo(solucion, velocidad)
            individuo._b_fitness = self._problema.fitness(individuo._solucion)
            self._individuos.append(individuo)

    def mejorIndividuo(self):
        for i in self._individuos:
            fitness = self._problema.fitness(i._solucion)
            if fitness < self._mejor:
                self._mejor = fitness

    def run(self):
        mejoresHistoricos = []
        self.crearIndividuos()
        self.mejorIndividuo()
        generacion = 0
        while (generacion <= self._generaciones):
            for idx in range(len(self._individuos)):
                h = 0
                for i in range(-self._ro // 2, self._ro // 2 + 1):
                    if i == 0:
                        continue
                    elif i == -self._ro // 2:
                        h = copy.deepcopy(self._individuos[(idx + i) % len(self._individuos)])
                    elif self._problema.fitness(self._individuos[(idx + i) % len(self._individuos)]._solucion) < self._problema.fitness(h._solucion):
                        h = copy.deepcopy(self._individuos[(idx + i) % len(self._individuos)])
                phi1 = np.random.random(size = self._dimensiones) * self._phi1_max
                phi2 = np.random.random(size = self._dimensiones) * self._phi2_max
                self._individuos[idx]._velocidad = (self._individuos[idx]._velocidad +
                np.multiply(phi1, self._individuos[idx]._b - self._individuos[idx]._solucion) +
                np.multiply(phi2, h._solucion - self._individuos[idx]._solucion))
                for i in range(self._dimensiones):
                    if abs(self._individuos[idx]._velocidad[i]) > self._v_max:
                        self._individuos[idx]._velocidad[i] = self._v_max / (self._individuos[idx]._velocidad[i])
                self._individuos[idx]._solucion = self._individuos[idx]._solucion + self._individuos[idx]._velocidad
                fitness_individuo = self._problema.fitness(self._individuos[idx]._solucion)
                if fitness_individuo < self._individuos[idx]._b_fitness:
                    self._individuos[idx]._b = copy.deepcopy(self._individuos[idx]._solucion)
                    self._individuos[idx]._b_fitness = fitness_individuo
                    if fitness_individuo < self._mejor:
                        self._mejor = fitness_individuo
            if generacion % 100 == 0:
                print('Generación ', generacion, ':', self._mejor)
                mejoresHistoricos.append(self._mejor)
            generacion += 1
        return mejoresHistoricos



def main():
    titulo = ""

    print("1.- Sphere")
    print("2.- Rosenbrock")
    print("3.- Rastrigin")
    print("4.- Quartic")
    opc = input("Seleccione que funcion desea realizar: ")

    if opc == "1":
        titulo = "Sphere"
        print("Ejecutando funcion Sphere")
        func = sphere.Sphere()

    elif opc == "2":
        titulo = "Rosenbrock"
        print("Ejecutando funcion Rosenbrock")
        func = rosenbrock.Rosenbrock()

    elif opc == "3":
        titulo = "Rastrigin"
        print("Ejecutando funcion Rastrigin")
        func = rastrigin.Rastrigin()

    else:
        titulo = "Quartic"
        print("Ejecutando funcion Quartic")
        func = quartic.Quartic()

    dimension = 2
    promedios_matriz =[]
    x = np.arange(0, 2001, 100)

    while dimension<=8:
        mejores_matriz = []
        print(f"{dimension} dimensiones")
        for i in range(5):
            print(f"Ejecucion {i+1}")
            
            rango = func.MAX_VALUE - func.MIN_VALUE
            cantidad_individuos = 30
            ro = 8
            phi1_max = 1.7
            phi2_max = 2.0
            v_max = rango * 0.01
            generaciones = 2000
            pso = PSO(cantidad_individuos, dimension, ro, phi1_max, phi2_max, v_max, func, generaciones)
            mejor_array = pso.run()

            mejores_matriz.append([abs(ele) for ele in mejor_array])

        promedio_mejores = np.mean(mejores_matriz, axis = 0)
        promedios_matriz.append(promedio_mejores)
        plt.plot(x,promedio_mejores)
        plt.title(f"{dimension} dimensiones")
        plt.show()
        dimension*=2

    for array in promedios_matriz:
        plt.plot(x,array)

    plt.title(titulo)
    plt.legend(["2 Dimensiones", "4 Dimensiones", "8 Dimensiones"])
    plt.show()

if  __name__ == '__main__':
    main()
