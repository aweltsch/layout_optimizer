import sys
from pyomo.environ import *
from pyomo.opt import SolverFactory

letters = "qwertyuiopasdfghjklzxcvbnm,./;" # TODO read as input / from config
weights = [[3.5, 2.4, 2.0, 2.2, 3.5, 3.5, 2.2, 2.0, 2.4, 3.5],
           [1.5, 1.2, 1.0, 1.0, 2.9, 2.9, 1.0, 1.0, 1.2, 1.5],
           [3.5, 2.8, 2.5, 1.7, 2.6, 2.6, 1.7, 2.5, 2.8, 3.5]]
fingers = [[0, 1, 2, 3, 3, 6, 6, 7, 8, 9],
           [0, 1, 2, 3, 3, 6, 6, 7, 8, 9],
           [0, 1, 2, 3, 3, 6, 6, 7, 8, 9]]

def map_char(c):
    c = c.lower()
    return c

def analyze_frequencies():
    _letters = set(letters)
    frequencies = {}
    bigram = ""
    trigram = ""
    for c in sys.stdin.read():
        c = map_char(c)
        if len(bigram) == 0:
            bigram = c
        elif len(bigram) == 1:
            bigram += c
        else:
            bigram = bigram[1:] + c

        #trigram = trigram[1:] + c
        if c in _letters:
            frequencies[c] = frequencies.get(c, 0) + 1
            if len(bigram) == 2 and bigram[0] in _letters:
                assert  bigram[1] in _letters
                frequencies[bigram] = frequencies.get(bigram, 0) + 1
        #if set(trigram).issubset(_letters):
        #    frequencies[trigram] = frequencies.get(trigram, 0) + 1

    for l in letters:
        if l not in frequencies:
            frequencies[l] = 0

    # initialize to float 0.0
    total = 0
    bigram_total = 0
    for l,f in frequencies.items():
        if len(l) == 1:
            total += f
        elif len(l) == 2:
            bigram_total += f
        else:
            pass

    for (i,l) in enumerate(letters):
        frequencies[l] = frequencies[l] / float(total)
        for m in letters[i:]:
            freq = (frequencies.get(l + m, 0) + frequencies.get(m + l, 0)) / float(bigram_total)
            if m + l in frequencies:
                del frequencies[m + l]
            #if freq < 0.01:
            #    if l + m in frequencies:
            #        del frequencies[l + m]
            else:
                frequencies[l + m] = freq

    return frequencies

def print_instance(letters, weights, frequencies):
    print("weights:")
    for l in weights:
        print(l)
    print("frequencies:")
    for l in frequencies:
        print("{} {}".format(l, frequencies[l]))


def main():
    opt = SolverFactory('cbc')

    frequencies = analyze_frequencies()
    matrix_indices = [(i,j) for i in range(len(weights)) for j in range(len(weights[i]))]
    print_instance(letters, weights, frequencies)
    p = ConcreteModel()
    p.letters = Set(initialize=letters)
    p.matrix_indices = Set(initialize=matrix_indices)

    p.v = Var(p.letters, p.matrix_indices, domain=Binary)
    print(len(frequencies))

    def need_key(model, l):
        return sum(model.v[l,i] for i in model.matrix_indices) == 1
    p.need_key = Constraint(p.letters, rule=need_key)
    
    def one_letter(model, i, j):
        return sum(model.v[l,(i,j)] for l in model.letters) <= 1

    p.one_letter = Constraint(p.matrix_indices, rule=one_letter)

    def objective_rule(model):
        objective = []
        for l in model.letters:
            for i in model.matrix_indices:
                objective.append(model.v[l,i] * frequencies[l] * weights[i[0]][i[1]])
        return sum(objective)

    p.objective = Objective(rule=objective_rule)

    # instance = p.create_instance()
    instance = p
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
    fingers = set([0, 1, 2, 3, 6, 7, 8, 9])
    finger_index = [[0, 1, 2, 3, 3, 6, 6, 7, 8, 9],
        [0, 1, 2, 3, 3, 6, 6, 7, 8, 9],
        [0, 1, 2, 3, 3, 6, 6, 7, 8, 9]]

    bigram_penalty = 3.5

    finger_map = {f: [] for f in fingers}
    for (i,j) in matrix_indices:
        finger = finger_index[i][j]
        finger_map[finger].append((i,j))

    w = p.new_variable(binary=True)
    for s in frequencies:
        if len(s) != 2:
            continue
        a, b = s
        for finger in fingers:
            constraint = [-w[s,finger]]
            for i,j in finger_map[finger]:
                constraint.append(v[a,i,j])
                constraint.append(v[b,i,j])
            p.add_constraint(sum(constraint) <= 1)

            objective.append(bigram_penalty * (-frequencies[s]) * w[s,finger])


    p.set_objective(sum(objective))
    print(p.solve())
    solution = p.get_values(v)
    layout = [[None for j in range(len(weights[i]))] for i in range(len(weights))]

    for (l, i, j), val in solution.items():
        if val > 0:
            layout[i][j] = l

    for l in layout:
        print(l)

    w_sol = p.get_values(w)
    for finger in fingers:
        print("finger {}: {}".format(finger, sum(frequencies[l] * solution[l, i, j] for l in letters for i,j in finger_map[finger])))
        print("penalty {}: {}".format(finger, sum(frequencies[l] * weights[i][j] * solution[l, i, j] for l in letters for i,j in finger_map[finger])))
        total = 0
        for i, l in enumerate(letters):
            for m in letters[i:]:
                if l + m in frequencies:
                    total += frequencies[l + m] * w_sol[l + m, finger]
        print("bigrams {}: {}".format(finger, total))

if __name__ == "__main__":
    main()
