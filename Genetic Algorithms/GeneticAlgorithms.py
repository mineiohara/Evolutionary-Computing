import random
from re import M
from typing import List

import openpyxl


class Gene(object):
    def __init__(self, realValuedGene: float) -> None:
        self.realValuedGene = realValuedGene
        self.eval = None


class Generation(object):
    # Initialize
    def __init__(self, popSize: int, dimention: int, pc: float, a: float, pm: float, b: int, maxGeneration: int, searchSpace: List[float]) -> None:
        self.popSize = popSize
        self.dimention = dimention
        self.pc = pc
        self.a = a
        self.pm = pm
        self.b = b
        self.currentGeneration = 0
        self.maxGeneration = maxGeneration
        self.searchSpace = searchSpace
        self.population = [ Gene(random.randrange(searchSpace[0], searchSpace[1])) for _ in range(popSize) ]
        self.minEval = [0, 100000000000000]
        self.maxTotalFitness = 0

    # Evaluation function
    def schewefel(self) -> None:
        sum = 0
        product = 1

        for gene in self.population:
            for i in range(self.dimention):
                # print(gene.realValuedGene)
                sum += gene.realValuedGene
                product *= gene.realValuedGene
            gene.eval = round(sum + product, 6)
            if abs(gene.eval) < abs(self.minEval[1]):
                self.minEval[0] = self.currentGeneration
                self.minEval[1] = gene.eval

        # print("Processing: {0} times, Min eval: {1} at generation {2}".format(self.currentGeneration, self.minEval[1], self.minEval[0]))

    # Print out the result
    def printer(self) -> None:
        for gene in self.population:
            print(gene.realValuedGene, end=" ")
        print()


    # Non-uniform mutation
    def mutation(self) -> None:
        for gene in self.population:
            if random.random() < self.pm:
                if random.randint(0,1): 
                    gene.realValuedGene -= (gene.realValuedGene - searchSpace[1]) * (1 - (random.random() ** ((1 - self.currentGeneration/self.maxGeneration) ** self.b)))
                    # print("LB")

                else:
                    gene.realValuedGene += (searchSpace[0] - gene.realValuedGene) * (1 - (random.random() ** ((1 - self.currentGeneration/self.maxGeneration) ** self.b)))
                    # print("UB")

    # Whole arithmetic crossover
    def crossover(self) -> None:
        isCrossover = []
        for _ in range(self.popSize):
            if random.random() <= self.pc: isCrossover.append(True)
            else: isCrossover.append(False)

        for i in range(self.popSize):
            if isCrossover[i]:
                for j in range(i+1, self.popSize):
                    if isCrossover[j]:
                        isCrossover[j] = False
                        # print("Parent1: {0}, Parent2: {1}".format(self.population[i].realValuedGene, self.population[j].realValuedGene))
                        child = self.population[i].realValuedGene * self.a + self.a * self.population[j].realValuedGene
                        self.population[i].realValuedGene, self.population[j].realValuedGene = child, child
                        # print("Child: {0}".format(child))

    # Tournament selection
    def selection(self) -> None:
        oldGen = self.population
        for i in range(self.popSize):
            rand1 = random.randint(0,self.popSize - 1)
            rand2 = random.randint(0,self.popSize - 1)

            if abs(oldGen[rand1].eval) < abs(oldGen[rand2].eval):
                self.population[i] = oldGen[rand1]
            else: 
                self.population[i] = oldGen[rand2]
        
        del oldGen



if __name__=='__main__':
    #Set parameters
    popSize = 20
    dimention = 30
    searchSpace = [-10, 10]

    pc = 0.25
    a = 0.5

    pm = 0.01
    maxGeneration = 200000
    b = 5

    g = Generation(popSize, dimention, pc, a, pm, b, maxGeneration, searchSpace)

    for m in range(100):
        g = Generation(popSize, dimention, pc, a, pm, b, maxGeneration, searchSpace)
        for i in range(g.maxGeneration):
            g.currentGeneration = i+1
            g.crossover()
            g.mutation()
            g.schewefel()
            g.selection()
        
        print("Optimum value: {0} at generation {1}".format(g.minEval[1], g.minEval[0]))
        wb = openpyxl.load_workbook('./data.xlsx')
        sheet = wb['Sheet1']
        sheet.cell(row=m+1, column=1).value = m+1
        sheet.cell(row=m+1, column=2).value = g.minEval[1]*100
        wb.save('./data.xlsx')
        wb.close()
        del g


