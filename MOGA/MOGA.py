from cmath import inf
from hashlib import new
from turtle import color
from typing import List
import random
import matplotlib.pyplot as plt
import numpy as np
import math

class Gene(object):
    def __init__(self, value: List[float]) -> None:
        self.value = value
        self.eval = [float(inf) for _ in range(2)]
        self.distance = 0

class Generation(object):
    def __init__(self, popSize: int, dimention: int, pc: float, a: float, pm: float, b: int, maxGeneration: int, searchSpace: List[float]) -> None:
        self.popSize = popSize
        self.dimention = dimention
        self.pc = pc
        self.a = a
        self.pm = pm
        self.b = b
        self.currentGeneration = 1
        self.maxGeneration = maxGeneration
        self.searchSpace = searchSpace
        self.population = [ Gene([random.random()*(searchSpace[1] - searchSpace[0])+ searchSpace[0] for _ in range(dimention)]) for _ in range(popSize) ]
        self.minEval = [[0, float(inf)] for _ in range(2)]
        self.f = None

    def printer(self):
        print("=====Generation {0}=====\nEval 1: {1} at generation {2}\nEval 2: {3} at generation {4}\n".format(self.currentGeneration, self.minEval[0][1], self.minEval[0][0], self.minEval[1][1], self.minEval[1][0]))


    def schaffer(self):
        for gene in self.population:
            gene.eval[0] = gene.value[0] ** 2

            if gene.eval[0] < self.minEval[0][1]:
                self.minEval[0][0] = self.currentGeneration
                self.minEval[0][1] = gene.eval[0]

        for gene in self.population:
            gene.eval[1] = (gene.value[0]-2) ** 2

            if gene.eval[1] < self.minEval[1][1]:
                self.minEval[1][0] = self.currentGeneration
                self.minEval[1][1] = gene.eval[1]


    def ZDT1(self):
        for gene in self.population:
            gene.eval[0] = gene.value[0]

            if gene.eval[0] < self.minEval[0][1]:
                self.minEval[0][0] = self.currentGeneration
                self.minEval[0][1] = gene.eval[0]

        for gene in self.population:
            sum = 0
            for i in range(1, self.dimention):
                sum += gene.value[i]
            g = 1 + 9 * sum / (self.dimention - 1)
            gene.eval[1] = g * (1 - np.sqrt(gene.value[0] / g))

            if gene.eval[1] < self.minEval[1][1]:
                self.minEval[1][0] = self.currentGeneration
                self.minEval[1][1] = gene.eval[1]
    
    def ZDT6(self):
        for gene in self.population:
            gene.eval[0] = 1- np.exp(-4 * gene.value[0]) * (np.sin(6 * np.pi * gene.value[0]) ** 6)

            if gene.eval[0] < self.minEval[0][1]:
                self.minEval[0][0] = self.currentGeneration
                self.minEval[0][1] = gene.eval[0]

        for gene in self.population:
            sum = 0
            for i in range(1, self.dimention):
                sum += gene.value[i]
            g = 1 + 9 * ((sum/(self.dimention)) ** 0.25)
            gene.eval[1] = g * (1 - (gene.eval[0]/g) ** 2)

            if gene.eval[1] < self.minEval[1][1]:
                self.minEval[1][0] = self.currentGeneration
                self.minEval[1][1] = gene.eval[1]

        

    def checkDominant(self, i, j) -> int:
        d = 0
        for k in range(len(self.population[0].eval)):
            if self.population[i].eval[k] < self.population[j].eval[k]: d += 1
            elif self.population[i].eval[k] > self.population[j].eval[k]: d -= 1

        return d

    def fastNonDominatedSort(self):
        popSize = len(self.population)
        self.f = [[]]
        s = [[] for _ in range(popSize)]
        n = [0 for _ in range(popSize)]

        for p in range(popSize):
            for q in range(popSize):
                if p == q: continue
                d = self.checkDominant(p, q)
                if d > 0: s[p].append(q)
                elif d < 0: n[p] += 1

            if n[p] == 0:
                self.f[0].append(p)
        
        i = 0
        while(len(self.f[i]) != 0):
            Q = []
            for p in self.f[i]:
                for q in s[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        Q.append(q)
            
            i += 1
            self.f.append(Q)

    def crowdingDistanceSorting(self, f, oldGen) -> List[Gene]:
        if len(f) < 3: 
            sol = []
            for i in range(len(f)):
                sol.append(oldGen[f[i]])
            return sol

        fmax = [1, 10]
        fmin = [0, 0]
        sol = []
        for i in f:
            sol.append(oldGen[i])
        
        d = [0 for _ in range(len(sol))]
        I = [[i for i in range(len(sol))] for _ in range(2)]

        for i in range(len(sol[0].eval)):
            for j in range(len(sol) - 1):
                for k in range(j+1, len(sol)):
                    if sol[j].eval[i] > sol[k].eval[i]:
                        sol[j], sol[k] = sol[k], sol[j]
            
            sol[0].distance, sol[-1].distance = float(inf), float(inf)
            for l in range(1, len(sol)-1):
                sol[l].distance += (sol[l+1].eval[i] - sol[l-1].eval[i])/(fmax[i] - fmin[i])
            
        for i in range(len(sol) - 1):
            for j in range(i+1, len(sol)):
                if sol[i].distance < sol[j].distance:
                    sol[i], sol[j] = sol[j], sol[i]

        return sol
    
    def makeNewPopulation(self):
        oldGen = self.population
        self.population = []

        for i in range(len(self.f)):
            sol = self.crowdingDistanceSorting(self.f[i], oldGen)
            if len(self.population) == self.popSize: break
            elif self.popSize >= len(self.population) + len(sol):
                for gene in sol:
                    self.population.append(gene)
            else: 
                diff = len(self.population) + len(sol) - self.popSize
                while diff != 0:
                    sol.pop(-1)
                    diff -= 1
                for gene in sol:
                    self.population.append(gene)
                break
            
            if len(self.population) == self.popSize: break

        del(oldGen)
    
    # Crowded tournament selection
    def selection(self) -> None:
        for _ in range(int(self.popSize/5)):
            rand1 = random.randint(0, len(self.population) - 1)
            rand2 = random.randint(0, len(self.population) - 1)
            
            d = self.checkDominant(rand1, rand2)

            if d > 0: self.population.append(self.population[rand1])
            elif d < 0: self.population.append(self.population[rand2])
            else:
                rand = random.randint(0,1)
                if self.population[rand1].eval[rand] < self.population[rand2].eval[rand]:
                    self.population.append(self.population[rand1])
                else: 
                    self.population.append(self.population[rand2])


    # Whole arithmetic crossover
    def crossover(self) -> None:
        isCrossover = []
        for _ in range(len(self.population)):
            if random.random() <= self.pc: isCrossover.append(True)
            else: isCrossover.append(False)

        for i in range(len(self.population)):
            if isCrossover[i]:
                for j in range(i+1, len(self.population)):
                    if isCrossover[j]:
                        isCrossover[j] = False
                        child = []
                        for k in range(self.dimention):
                            child.append(self.population[i].value[k] * self.a + self.a * self.population[j].value[k])
                        
                        self.population.append(Gene(child))
                        # self.population.append(Gene(child))
                        del child

    # Non-uniform mutation
    def mutation(self) -> None:
        for j in range(len(self.population)):
            if random.random() < self.pm:
                newValue = [0 for _ in range(self.dimention)]
                if random.randint(0,1):
                    for i in range(self.dimention):
                        geneValue = self.population[j].value[i]
                        newValue[i] = geneValue - (geneValue - self.searchSpace[0]) * (1 - (random.random() ** ((1 - self.currentGeneration/self.maxGeneration) ** self.b)))
                        # newValue[i] = geneValue - (geneValue - self.searchSpace[0]) * (1 - (random.random() ** self.b))

                
                else:
                    for i in range(self.dimention):
                        geneValue = self.population[j].value[i]
                        newValue[i] = geneValue + (self.searchSpace[1] - geneValue) * (1 - (random.random() ** ((1 - self.currentGeneration/self.maxGeneration) ** self.b)))
                        # newValue[i] = geneValue + (self.searchSpace[1] - geneValue) * (1 - (random.random() ** self.b))


                self.population.append(Gene(newValue))


if __name__ == "__main__":
    # Set parameters
    popSize = 20
    pc = 0.5
    a = 0.5
    pm = 0.8
    b = 5
    maxGeneration = 2000


    wait = 0.000001
    plt.ion()
    fig, ax = plt.subplots()
    f1, f2 = [],[]
    ff1, ff2 = [], []
    fff1, fff2 = [], []

    plt.title("Dynamic Plot",fontsize=25)
    plt.xlabel("f1",fontsize=18)
    plt.ylabel("f2",fontsize=18)


    dimention = 30
    searchSpace = [0, 1]
    z = np.linspace(0,1)
    x = z
    y = 1 * (1 - np.sqrt(z))
    plt.xlim(searchSpace[0] - 0.25, searchSpace[1] + 0.25)
    plt.ylim(-0.5, 5)


    # dimention = 1
    # searchSpace = [-10 ** 3, 10 ** 3]
    # z = np.linspace(0, 2)
    # x = z ** 2
    # y = (z - 2) ** 2
    # plt.xlim(-0.2, 20)
    # plt.ylim(-0.5, 20)

    # dimention = 10
    # searchSpace = [0, 1]
    # x = 1 - np.exp(-4 * z) * (np.sin(6 * np.pi * z) ** 6)
    # y = 1 - x ** 2
    # plt.xlim(0.2, 1.25)
    # plt.ylim(-0.2, 5)
    g = Generation(popSize, dimention, pc, a, pm, b, maxGeneration, searchSpace)


    plt.title("Dynamic Plot",fontsize=25)
    plt.xlabel("f1",fontsize=18)
    plt.ylabel("f2",fontsize=18)

    for gene in g.population:
        f1.append(gene.eval[0])
        f2.append(gene.eval[1])

    sc = ax.scatter(f1, f2, color="black", s=20)
    ff = ax.scatter(ff1, ff2, color="red", s=20)
    fff = ax.scatter(fff1, fff2, color="blue", s=20)
    ax.plot(x, y, color="red")

    input()

    while(g.currentGeneration < maxGeneration+1):
        g.crossover()
        g.mutation()
        g.ZDT1()
        # g.ZDT6()
        # g.schaffer()
        g.fastNonDominatedSort()

        f1, f2 = [],[]
        ff1, ff2 = [],[]
        fff1, fff2 = [],[]

        plt.title("Generation {0}".format(g.currentGeneration),fontsize=25)
        for gene in g.population:
            f1.append(gene.eval[0])
            f2.append(gene.eval[1])

        for i in g.f[0]:
            ff1.append(g.population[i].eval[0])
            ff2.append(g.population[i].eval[1])

        for i in g.f[1]:
            fff1.append(g.population[i].eval[0])
            fff2.append(g.population[i].eval[1])

            

        sc.set_offsets(np.c_[f1,f2])
        ff.set_offsets(np.c_[ff1,ff2])
        fff.set_offsets(np.c_[fff1,fff2])
        fig.canvas.draw_idle()
        plt.pause(wait)

        g.makeNewPopulation()
        g.fastNonDominatedSort()
        f1, f2 = [],[]
        ff1, ff2 = [],[]
        fff1, fff2 = [],[]


        for gene in g.population:
            f1.append(gene.eval[0])
            f2.append(gene.eval[1])

        for i in g.f[0]:
            ff1.append(g.population[i].eval[0])
            ff2.append(g.population[i].eval[1])


        for i in g.f[1]:
            fff1.append(g.population[i].eval[0])
            fff2.append(g.population[i].eval[1])

        # for i in g.edge:
        #     e1.append(i.eval[0])
        #     e2.append(i.eval[1])
            

        sc.set_offsets(np.c_[f1,f2])
        ff.set_offsets(np.c_[ff1,ff2])
        fff.set_offsets(np.c_[fff1,fff2])
        fig.canvas.draw_idle()
        plt.pause(wait)

        g.currentGeneration += 1

    # print("======Solution Sets=====")
    # for i in range(len(g.population)):
    #     print("{0}: {1}".format(i+1, g.population[i].value))