import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np,random, operator, pandas as pd
import scipy as sc
from skimage import io
import math
from statistics import mode
from scipy import stats
from PIL import Image
from skimage.transform import rescale, resize
import random, operator, pandas as pd

def createChromosome(k,maxSol):
    solution = np.random.randint(0,maxSol,k)
    return solution

def initialPopulation(popSize, k,constant):
    population = []
    for i in range(0, popSize):
        population.append(createChromosome(k,constant))
    return population

def calculateError(chromosome,problem,value):
    size = len(chromosome)
    result = 0
    for i in range(0,size):
        result = result + (problem[i] * chromosome[i])
    result = abs(value - result)
    return result
    
def rankRoutes(population,problem,value):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = calculateError(population[i],problem,value)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    mutationIndex1 = random.randrange(0,len(patent1));
    mutationIndex2 = random.randrange(mutationIndex1,len(patent1));

    for i in range(0, mutationIndex1):
        child.append(parent1[i]);

    for i in range(mutationIndex1,mutationIndex2):
        child.append(parent2[i]);
        
    for i in range(mutationIndex2,len(patent2)):
        child.append(parent2[i]);
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children	

def mutate(individual, mutationRate):

    index = int(random.random() * len(individual))
    individual[index] = (individual[index] + 1)%2        
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def geneticAlgorithm(constant, popSize, k,eliteSize, mutationRate, generations,Problem,value):
    currentGen = initialPopulation(popSize, k,constant)
    
    for i in range(0, generations):
        popRanked = rankRoutes(currentGen,Problem,value)
        #print(popRanked[0])
        if popRanked[0][1] == 0:
            #print(currentGen[popRanked[0][0]])
            return currentGen[popRanked[0][0]]
        selectionResults = selection(popRanked, eliteSize)
        matingpool = matingPool(currentGen, selectionResults)
        children = breedPopulation(matingpool, eliteSize)
        children = mutatePopulation(children, mutationRate)
        
        currentGen = np.vstack((currentGen,children))
        popRanked = rankRoutes(currentGen,Problem,value)
        selectionResults = selection(popRanked, popSize)
        currentGen = matingPool(currentGen, selectionResults)
        
    bestRoute = currentGen[0]

    return bestRoute
    
        
coeffarr = [1,2,3];
cons = 10
#print(arr);
#myarr1 = createarray(aa,rows,cols)
size = len(coeffarr)
#mproblem = Equation(arr)
myarr = geneticAlgorithm(constant = cons,popSize = 100,k=size,eliteSize = 70,mutationRate = 0.1,
generations = 50,Problem = coeffarr,value = cons )
print(myarr)


