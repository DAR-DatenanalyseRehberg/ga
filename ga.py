import pandas as pd
from random import randint, shuffle, choices, random
from heapq import nlargest
from copy import deepcopy
df = pd.read_excel("DistanceStores22.xlsx")
df['DistanceKM'] = ((df['Distance(m)']/1000).astype(int)).astype(str)
keys = zip(df['StartLocation'],df['Destination'])
values = df['DistanceKM'].astype(int)
source_distances = dict(zip(keys, values))
distances = {}
for places in source_distances:
    distances[places] = source_distances[places]


class PotentialSolution:
    def randomInstance(cls):
        return PotentialSolution()

    def fitness(self):
        return 1

    def mutate(self):
        return self

    def crossover(self, other):
        return [self, other]

class AdvancedGuess:

    def __init__(self, population, expected, max_generations,
            crossover_chance, mutation_chance):
        self.population = population
        self.expected = expected
        self.max_generations = max_generations
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance

    def chooseParents(self):
        pool = choices(self.population, k = len(self.population) // 4)
        return nlargest(2, pool, key = lambda potentialSolution: potentialSolution.fitness())

    def propagate(self):
        newPopulation = []
        while len(newPopulation) < len(self.population):
            parents = self.chooseParents()
            if random() < self.crossover_chance:
                [child1, child2] = parents[0].crossover(parents[1])
                newPopulation.append(child1)
                newPopulation.append(child2)
            else:
                newPopulation.append(parents[0])
                newPopulation.append(parents[1])
        if len(newPopulation) > len(self.population):
            newPopulation.pop()
        for potentialSolution in newPopulation:
            if random() < self.mutation_chance:
                potentialSolution.mutate()
        self.population = newPopulation

    def find(self):
        optimal = deepcopy(
            max(self.population, key = lambda potentialSolution: potentialSolution.fitness())
        )
        for i in range(0, self.max_generations):
            if optimal.fitness() >= self.expected:
                return optimal
            self.propagate()
            current_best = deepcopy(
                max(self.population, key = lambda potentialSolution: potentialSolution.fitness())
            )
            if current_best.fitness() > optimal.fitness():
                optimal = current_best
            print(i, optimal)
        return optimal

class distanceShuffling(PotentialSolution):
    locations = df['StartLocation'].unique().tolist() #["StoreA", "StoreB", "StoreC","StoreD"]
    distance =distances # {('StoreA', 'StoreB'): 15, ('StoreB', 'StoreA'): 15, ('StoreC', 'StoreB'): 10,
    def __init__(self, places):
        self.places = places

    def getDistance(self):
        sum = 0
        for index, store in enumerate(self.places):
            if index < len(self.places) - 1:
                nextStore = self.places[index + 1]
            else:
                nextStore = self.places[0]
            sum += self.distance[(store, nextStore)] # C zu D 2 km, A zu D 10km
        return sum # total sum of distance per first combination:D,B,C,A -->38km; danach nächste Schleife, wieder beginnend Index0

    def fitness(self):
        return 1 / self.getDistance()

    @classmethod
    def randomCoordinates(cls):
        PlacesCopy = cls.locations[:]
        shuffle(PlacesCopy)
        return distanceShuffling(PlacesCopy)

    def mutate(self):
        rand_index_1 = randint(0, len(self.places) - 1)
        rand_index_2 = randint(0, len(self.places) - 1)
        if rand_index_1 != rand_index_2:
            self.places[rand_index_1], self.places[rand_index_2] = (
                self.places[rand_index_2], self.places[rand_index_1]
            )

    def crossover(self, other):
        child1 = deepcopy(self)
        child2 = deepcopy(other)
        rand_index = randint(0, len(child1.places) - 1)
        store = child1.places[rand_index]
        other_index = child2.places.index(store)
        if rand_index != other_index:
            child1.places[rand_index], child1.places[other_index] = (
                child1.places[other_index], child1.places[rand_index]
            )
            child2.places[rand_index], child2.places[other_index] = (
                child2.places[other_index], child2.places[rand_index]
            )
        return [child1, child2]

    def __str__(self):
        result = " - ".join(self.places)
        result += " - " + self.places[0]
        result += ": " + str(self.getDistance())
        return result

if __name__ == '__main__':
    population = []
    for i in range(0, 20):
        population.append(distanceShuffling.randomCoordinates())
    advancedGuessing = AdvancedGuess(population, 0.0000001, 20, 0.7, 0.5)
    optimal = advancedGuessing.find()
    print("Shortest distance found:", optimal)


