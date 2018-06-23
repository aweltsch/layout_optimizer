import sys
from pyomo.environ import *
from pyomo.opt import SolverFactory

letters = "qwertyuiopasdfghjklzxcvbnm,./;" # TODO read as input / from config
weights = [[3.5, 2.4, 2.0, 2.2, 3.5, 3.5, 2.2, 2.0, 2.4, 3.5],
           [1.5, 1.2, 1.0, 1.0, 2.9, 2.9, 1.0, 1.0, 1.2, 1.5],
           [3.5, 2.8, 2.5, 1.7, 2.6, 2.6, 1.7, 2.5, 2.8, 3.5]]
fingers = set([0, 1, 2, 3, 6, 7, 8, 9])
finger_index = [[0, 1, 2, 3, 3, 6, 6, 7, 8, 9],
           [0, 1, 2, 3, 3, 6, 6, 7, 8, 9],
           [0, 1, 2, 3, 3, 6, 6, 7, 8, 9]]

pinky_left = 0
pinky_right = 9
ring_left = 1
ring_right = 8
middle_left = 2
middle_right = 7

same_finger_penalties = {0: 2.5, 1: 2.5, 2: 3.5}
pinky_ring_penalties = {0: 0.5, 1: 1.0, 2: 1.5}
ring_middle_penalties = {0: 0.1, 1: 0.2, 2: 0.3}

# TODO: incorporate this
bigram_threshold = 1 # use only the major percentage of bigrams
def map_char(c):
    c = c.lower()
    return c

# TODO
def optimize_frequencies(frequencies):
    pass

def analyze_frequencies():
    _letters = set(letters)
    frequencies = {}
    bigram = ""
    total = 0
    bigram_total = 0
    for c in sys.stdin.read():
        c = map_char(c)
        if len(bigram) == 0:
            bigram = c
        elif len(bigram) == 1:
            bigram += c
        else:
            bigram = bigram[1:] + c

        if c in _letters:
            frequencies[c] = frequencies.get(c, 0) + 1
            total += 1
            if len(bigram) == 2 and bigram[0] in _letters:
                assert  bigram[1] in _letters
                frequencies[bigram] = frequencies.get(bigram, 0) + 1
                bigram_total += 1
    assert total > 0
    assert bigram_total > 0

    bigram_res = {}
    for i, l in enumerate(letters):
        if l not in frequencies:
            frequencies[l] = 0
        for k in letters[i:]:
            # bigram ab is the same as ba in terms of penalty
            bigram_res[l + k] = (frequencies.get(l + k, 0) + frequencies.get(k + l, 0)) / bigram_total

    # unite frequencies
    result = {}
    for l in letters:
        result[l] = frequencies.get(l, 0) / total
    for k, v in bigram_res.items():
        result[k] = v
    return result

def print_instance(letters, weights, frequencies):
    print("weights:")
    for l in weights:
        print(l)
    print("frequencies:")
    for l in frequencies:
        print("{} {}".format(l, frequencies[l]))

def _add_bigram_penalty():
    pass

def add_bigram_penalties():
    pass

def create_instance(frequencies):
    matrix_indices = [(i,j) for i in range(len(weights)) for j in range(len(weights[i]))]
    objective = [] # gradually append summands

    p = ConcreteModel()
    p.letters = Set(initialize=letters)
    p.matrix_indices = Set(initialize=matrix_indices)

    p.v = Var(p.letters, p.matrix_indices, domain=Binary)

    # FIXME maybe add sos-1 constraint
    def need_key(model, l):
        return sum(model.v[l,i] for i in model.matrix_indices) == 1
    p.need_key = Constraint(p.letters, rule=need_key)

    def one_letter(model, i, j):
        return sum(model.v[l,(i,j)] for l in model.letters) <= 1
    p.one_letter = Constraint(p.matrix_indices, rule=one_letter)

    # prepare objective
    for i,j in matrix_indices:
        for l in letters:
            objective.append(weights[i][j] * frequencies[l] * p.v[l, (i,j)])

    def objective_rule(model):
        return sum(objective)

    p.objective = Objective(rule=objective_rule)
    return p

def calculate_objective_value(layout):
    instance = create_instance()

    def fix_layout_rule(model, i, j):
        return v[layout[i][j], (i, j)] == 1
    instance.fix_layout = Constraint(instance.letters, instance.matrix_indices, rule=fix_layout_rule)
    return instance

# FIXME layout
LAYOUT = [['q', 'w', 'f', 'p', 'b', 'j', 'l', 'u', 'y', ';'],
    ['a', 'r', 's', 't', 'g', 'k', 'n', 'e', 'i', 'o'],
    ['z', 'x', 'c', 'd', 'v', 'm', 'h', ',', '.', '/']]
def main():
    opt = SolverFactory('cbc')
    frequencies = analyze_frequencies()
    # TODO logging
    print_instance(letters, weights, frequencies)

    # TODO different behaviour if layout is given
    instance = create_instance(frequencies)
    # instance = p.create_instance()
    results = opt.solve(instance)

    layout = [[None for j in range(len(weights[i]))] for i in range(len(weights))]

    for l in instance.letters:
        for i,j in instance.matrix_indices:
            if instance.v[l, (i,j)]:
                layout[i][j] = l

    for l in layout:
        print(l)
    # FIXME
    return

if __name__ == "__main__":
    main()
