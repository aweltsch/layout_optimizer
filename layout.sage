from random import randint
letters = "qwertyuiopasdfghjklzxcvbnm,." # TODO read as input / from config
weights = [[11, 5, 3, 2, 7, 7, 2, 3, 5, 11],
           [3, 1, 1, 1, 2.5, 2.5, 1, 1, 1, 3],
           [10, 6, 3, 1.5, 8, 8, 1.5, 3, 6, 10]]
frequencies = {
        l: 100 for l in letters
        }

def main():
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
    print(p.solve())
    solution = p.get_values(v)
    layout = [[None for j in range(len(weights[i]))] for i in range(len(weights))]

    for (l, i, j), val in solution.items():
        if val > 0:
            layout[i][j] = l

    print(layout)

if __name__ == "__main__":
    main()
