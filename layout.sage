from random import randint
letters = "qwertyuiopasdfghjklzxcvbnm,." # TODO read as input / from config
weights = [[11, 5, 3, 2, 7, 7, 2, 3, 5, 11],
           [3, 1, 1, 1, 2.5, 2.5, 1, 1, 1, 3],
           [10, 6, 3, 1.5, 8, 8, 1.5, 3, 6, 10]]

def analyze_frequencies():
    frequencies = {}
    for c in sys.stdin.read():
        if c in letters:
            frequencies[c] = frequencies.get(c, 0) + 1

    for l in letters:
        if l not in frequencies:
            frequencies[l] = 0

    return frequencies

def print_instance(letters, weights, frequencies):
    print("weights:")
    for l in weights:
        print(l)
    print("frequencies:")
    for l in letters:
        print("{} {}".format(l, frequencies[l]))


def main():
    frequencies = analyze_frequencies()
    print_instance(letters, weights, frequencies)
    p = MixedIntegerLinearProgram()
    v = p.new_variable(binary=True)
    for l in letters:
        p.add_constraint(sum([v[l,i,j] for i in range(len(weights)) for j in range(len(weights[i]))]) == 1)

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            p.add_constraint(sum(v[l,i,j] for l in letters) <= 1)

    objective = []
    for l in letters:
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                objective.append(v[l,i,j] * (-frequencies[l]) * weights[i][j])

    p.set_objective(sum(objective))
    p.solve()
    solution = p.get_values(v)
    layout = [[None for j in range(len(weights[i]))] for i in range(len(weights))]

    for (l, i, j), val in solution.items():
        if val > 0:
            layout[i][j] = l

    for l in layout:
        print(l)

if __name__ == "__main__":
    main()
