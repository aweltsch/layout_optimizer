import sys
from pyomo.environ import *
from pyomo.opt import SolverFactory

letters = "qwertyuiopasdfghjklzxcvbnm,./;" # TODO read as input / from config
weights = [[3.5, 2.4, 2.0, 2.2, 3.4, 3.4, 2.2, 2.0, 2.4, 3.5],
           [1.5, 1.2, 1.0, 1.0, 2.9, 2.9, 1.0, 1.0, 1.2, 1.5],
           [3.3, 2.6, 2.4, 1.7, 3.1, 3.1, 1.7, 2.4, 2.6, 3.3]]
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

same_finger_penalties = {0: 1, 1: 0, 2: 0}
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

def analyze_frequencies(f):
    _letters = set(letters)
    frequencies = {}
    bigram = ""
    total = 0
    bigram_total = 0
    for line in f:
        for c in line:
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

def _get_bigram_indices(finger_1, finger_2, bigrams):
    finger_map = create_finger_map(finger_index)
    seen = set()
    indices = []
    for i_1, j_1 in finger_map[finger_1]:
        seen.add((i_1, j_1))
        for i_2, j_2 in finger_map[finger_2]:
            if (i_2, j_2) in seen:
                continue
            for s in bigrams:
                indices.append((i_1, j_1, i_2, j_2, s))

    return indices

def _add_bigram_penalty(instance, finger_1, finger_2, bigrams, penalties, frequencies):
    finger_map = create_finger_map(finger_index)
    # calculate index set
    index_name = "penalty_indices_{}_{}".format(finger_1, finger_2)
    var_name = "penalty_vars_{}_{}".format(finger_1, finger_2)
    constraint_name = "penalty_constraint_{}_{}".format(finger_1, finger_2)

    var = getattr(instance, var_name)
    index = getattr(instance, index_name)

    def penalty_rule(model, i_1, j_1, i_2, j_2, s):
        s_1, s_2 = s
        return model.v[s_1, (i_1, j_1)] + model.v[s_2, (i_1, j_1)] \
                + model.v[s_1, (i_2, j_2)] + model.v[s_2, (i_2, j_2)] \
                - var[(i_1,j_1), (i_2, j_2), s] <= 1
    setattr(instance, constraint_name, Constraint(index, rule=penalty_rule))
    # TODO
    objective = [penalties[abs(i_1 - i_2)] * frequencies[s] * var[i_1, j_1, i_2, j_2, s] for (i_1, j_1, i_2, j_2, s) in index]
    return objective

def create_finger_map(finger_index):
    finger_map = {finger: [] for finger in fingers}
    for i, row in enumerate(finger_index):
        for j, finger in enumerate(row):
            finger_map[finger].append((i,j))
    return finger_map

def add_bigram_penalties(instance, frequencies):
    objective = []
    # only bigrams with 2 different symbols, the others are irrelevant
    bigrams = set(filter(lambda x: len(x) == 2 and x[0] != x[1], frequencies))
    instance.bigrams = bigrams
    for finger in fingers:
        penalty_index = _get_bigram_indices(finger, finger, bigrams)
        index_name = "penalty_indices_{}_{}".format(finger, finger)
        # FIXME why can penalty index _not_ be a pyomo Set()?
        setattr(instance, index_name, penalty_index)

        var_name = "penalty_vars_{}_{}".format(finger, finger)
        setattr(instance, var_name, Var(penalty_index, domain=Binary))

        objective.extend(
                _add_bigram_penalty(instance, finger, finger, bigrams, same_finger_penalties, frequencies)
                )
    # returns part of objective
    return objective

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

    #objective.extend(add_bigram_penalties(p, frequencies))
    objective.extend(add_bigram_penalties(p, frequencies))

    # prepare objective
    for i,j in matrix_indices:
        for l in letters:
            objective.append(weights[i][j] * frequencies[l] * p.v[l, (i,j)])

    def objective_rule(model):
        return sum(objective)

    p.objective = Objective(rule=objective_rule)
    return p

def calculate_objective_value(frequencies, layout):
    instance = create_instance(frequencies)

    def fix_layout_rule(model, i, j):
        return model.v[layout[i][j], (i, j)] == 1
    instance.fix_layout = Constraint(instance.matrix_indices, rule=fix_layout_rule)
    return instance

def verify_results(instance, layout):
    """Check if bigram penalty variables are correctly set."""
    indices = list(instance.matrix_indices)
    for finger_1 in fingers:
        for finger_2 in fingers:
            var_name = "penalty_vars_{}_{}".format(finger_1, finger_2)
            if hasattr(instance, var_name):
                print("fingers:", finger_1, finger_2)
                vars = getattr(instance, var_name)
                for i_1, j_1 in indices:
                    for i_2, j_2 in indices:
                        for bigram in instance.bigrams:
                            try:
                                x = vars[i_1, j_1, i_2, j_2, bigram]
                                if x:
                                    print(i_1, j_1, i_2, j_2, bigram)
                            except:
                                pass



# FIXME layout
LAYOUT = [['q', 'w', 'f', 'p', 'b', 'j', 'l', 'u', 'y', ';'],
    ['a', 'r', 's', 't', 'g', 'k', 'n', 'e', 'i', 'o'],
    ['z', 'x', 'c', 'd', 'v', 'm', 'h', ',', '.', '/']]
def main():
    # FIXME find a good way to handle this gracefully...
    encoding = 'latin-1'
    with open(0, encoding=encoding) as f:
        # default mode is to optimize
        frequencies = analyze_frequencies(f)
        # TODO logging
        print_instance(letters, weights, frequencies)

        # TODO different behaviour if layout is given
        #instance = create_instance(frequencies)
        instance = calculate_objective_value(frequencies, LAYOUT)

        opt = SolverFactory('cbc')
        results = opt.solve(instance)
        print(results)

        layout = [[None for j in range(len(weights[i]))] for i in range(len(weights))]

        for l in instance.letters:
            for i,j in instance.matrix_indices:
                if instance.v[l, (i,j)]:
                    layout[i][j] = l

        for l in layout:
            print(l)

        return

if __name__ == "__main__":
    main()
