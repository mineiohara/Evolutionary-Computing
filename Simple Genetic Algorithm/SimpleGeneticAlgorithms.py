import math
import random
from typing import List


#A gene object in population
class Gene(object):
    def __init__(self, bitGene: List[str]) -> None:
        self.bitGene = bitGene
        self.eval = None
        

#Generation object
class Generation(object):
    # Initialize
    def __init__(self, popSize: int, pc: float, pm: float, decimalPoints: int, searchSpace: List[float]) -> None:
        self.popSize = popSize
        self.pc = pc
        self.pm = pm
        self.decimalPoints = decimalPoints
        self.searchSpace = searchSpace
        self.bitLen = [ math.ceil(math.log2((domain[1] - domain[0]) * (10**decimalPoints) + 1)) for domain in searchSpace ]
        self.population = [ Gene(bitGene = format(random.randint(0, (2**sum(self.bitLen))-1), '0'+ str(sum(self.bitLen)) +'b')) for _ in range(popSize) ]
        self.maxEval = [0, 0]
        self.maxTotalFitness = 0


    # Show best score
    def printResult(self) -> None:
        print('Max total fitness:', self.maxTotalFitness)
        print('Max eval:', self.maxEval[0], 'at', self.maxEval[1], 'loop')
        print()


    # Evaluation
    def evaluate(self, loop: int) -> None:

        # F(x, y) = 21.5 + x * sin(4pi*x) + y * sin(20pi * y)
        totalFitness = 0
        for gene in self.population:
            x = int(gene.bitGene[0:self.bitLen[0]], 2) * (searchSpace[0][1] - searchSpace[0][0]) / (2**self.bitLen[0] - 1) + searchSpace[0][0]
            y = int(gene.bitGene[self.bitLen[0]:], 2) * (searchSpace[1][1] - searchSpace[1][0]) / (2**self.bitLen[1] - 1) + searchSpace[1][0]
            gene.eval = round(21.5 + x * math.sin(4 * math.pi * x) + y * math.sin(20 * math.pi * y), self.decimalPoints)
            totalFitness += gene.eval
            if gene.eval > self.maxEval[0]: self.maxEval = [gene.eval, loop]
        
        if totalFitness > self.maxTotalFitness: self.maxTotalFitness = totalFitness

        # Select the gene of new generation
        p = 0.0
        q = []
        for gene in self.population:
            p += gene.eval/totalFitness
            q.append(round(p, self.decimalPoints))

        oldGen = self.population
        for gene in self.population:
            rand = random.random()
            for elem in q:
                if elem > rand:
                    gene.bitGene = oldGen[q.index(elem)].bitGene
                    break
        del oldGen


    # Crossover
    def crossover(self) -> None:
        isCrossover = []
        for gene in self.population:
            if random.random() <= self.pc: isCrossover.append(True)
            else: isCrossover.append(False)
        
        length = len(isCrossover)
        for i in range(length):
            if isCrossover[i]:
                isCrossover[i] + False
                for j in range(i, length):
                    if isCrossover[j]:
                        isCrossover[j] = False
                        rand = random.randint(0, self.popSize-1)
                        self.population[i].bitGene = self.population[j].bitGene[:rand] + self.population[i].bitGene[rand:]
                        self.population[j].bitGene = self.population[i].bitGene[:rand] + self.population[j].bitGene[rand:]


    # Mutation
    def mutation(self) -> None:
        for i in range(self.popSize):
            for j in range(sum(self.bitLen)):
                if random.random() < self.pm:
                    if self.population[i].bitGene[j] == '0': 
                        self.population[i].bitGene = self.population[i].bitGene[:j] + '1' + self.population[i].bitGene[j+1:]
                    else:
                        self.population[i].bitGene = self.population[i].bitGene[:j] + '0' + self.population[i].bitGene[j+1:]



if __name__ == '__main__':
    # Set parameters
    popSize = 20
    pc = 0.25
    pm = 0.01
    decimalPoints = 4
    searchSpace = [[-3.0, 12.1], [4.1, 5.8]]

    # Make a instance and initialize
    g = Generation(popSize, pc, pm, decimalPoints, searchSpace)
    g.evaluate(1)
    g.printResult()

    #Reproduction
    for i in range(100000):
        g.crossover()
        g.mutation()
        g.evaluate(i+2)
        print('Loop:', i+2)
        g.printResult()

    g.printResult()