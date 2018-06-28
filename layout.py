from __future__ import division
import argparse
import json
import sys
from enum import Enum
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

same_finger_penalties = {0: 2.5, 1: 2.5, 2: 3.5}
pinky_ring_penalties = {0: 0.5, 1: 1.0, 2: 1.5}
ring_middle_penalties = {0: 0.1, 1: 0.2, 2: 0.3}

bigram_threshold = 1 # use only the major percentage of bigrams

# TODO
def optimize_frequencies(frequencies, bigram_threshold):
    """
    Drop certain bigrams from the frequency dict. Keep only bigrams whose frequencies
    sum up to at least bigram_threshold. Bigrams with higher frequency will be kept
    first.
    This can be used to use only the most relevant bigrams that make up a certain percentage
    of all bigrams.

    Args:
        frequencies (dict): Contains frequencies of bigrams, assumed to be a partition of 1.
        bigram_threshold (float): 0 <= bigram_threshold <= 1.

    Example:
        We want to have only the bigrams with the highest frequency that make up
        80% of the bigrams overall:

        frequencies = {'aa': 0.3, 'bb': 0.3, 'cc': 0.2, 'dd': 0.1, 'ee': 0.1'}
        bigram_threshold = 0.8

        optimize_frequencies(frequencies, bigram_threshold) == {'aa': 0.3, 'bb': 0.3, 'cc': 0.2}
    """
    # TODO implement
    return frequencies

def analyze_frequencies(f):
    """
    Count relative frequencies of single characters and bigrams in the given file.
    Skip any character or bigram containing a symbol that is not given via the parameters.

    Args:
        f: A file.
        letters: A collection of relevant symbols.

    Returns:
        A dictionary, containing all relative frequencies (float <= 1) of characters
        and bigrams found in the file. Any character or bigram not given via the
        letters parameter will be filtered out.
    """
    _letters = set(letters)
    frequencies = {}
    bigram = ""
    total = 0
    bigram_total = 0
    for line in f:
        for c in line:
            c = c.lower()
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

def read_layout(f_name):
    """
    Read a layout (symbols assigned to key positions) from a file.

    Args:
        f_name (str): File name of the layout.

    Returns:
        The symbols accumulated in a list of lists representing the key placement.

    Example:
        The layout file might look like this:
            a b c
            d e f g
            h i j

        This function returns [['a', 'b', 'c'], ['d', 'e', 'f', 'g'], ['h', 'i', 'j']].
    """
    layout = []

    with open(f_name) as f:
        for line in f:
            line = line.strip()
            layout.append(line.split())

    return layout

def print_instance(letters, weights, frequencies):
    """
    Print relevant parameters used to assemble the instance.

    Args:
        letters: Collection of symbols assigned.
        weights: Weights for each key position, as list of lists.
        frequencies: Dictionary containing the frequencies of characters and bigrams.
    """
    print("weights:")
    for l in weights:
        print(l)
    print("frequencies:")
    for l in frequencies:
        print("{} {}".format(l, frequencies[l]))

# TODO extend doc
def _get_bigram_indices(finger_1, finger_2, bigrams):
    """
    Return list of all pairs of indices of a bigram using finger_1 and finger_2.

    Args:
        finger_1 (enum): 
        finger_2 (enum):
        bigrams: Set of all relevant bigrams.
    """

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
    """
    Add constraints for bigram penalties to problem instance.
    Args:
        instance:
        finger_1:
        finger_2:
        bigrams:
        penalties:
        frequencies:
    """

    finger_map = create_finger_map(finger_index)
    penalty_index = _get_bigram_indices(finger_1, finger_2, bigrams)
    index_name = "penalty_indices_{}_{}".format(finger_1, finger_2)
    # FIXME why can penalty index _not_ be a pyomo Set()?
    setattr(instance, index_name, penalty_index)

    var_name = "penalty_vars_{}_{}".format(finger_1, finger_2)
    setattr(instance, var_name, Var(penalty_index, domain=Binary))
    # calculate index set
    constraint_name = "penalty_constraint_{}_{}".format(finger_1, finger_2)

    var = getattr(instance, var_name)
    index = getattr(instance, index_name)

    def penalty_rule(model, i_1, j_1, i_2, j_2, s):
        s_1, s_2 = s
        return model.v[s_1, (i_1, j_1)] + model.v[s_2, (i_1, j_1)] \
                + model.v[s_1, (i_2, j_2)] + model.v[s_2, (i_2, j_2)] \
                - var[(i_1,j_1), (i_2, j_2), s] <= 1
    setattr(instance, constraint_name, Constraint(index, rule=penalty_rule))

    objective = [penalties[abs(i_1 - i_2)] * frequencies[s] * var[i_1, j_1, i_2, j_2, s] for (i_1, j_1, i_2, j_2, s) in index]
    return objective

def create_finger_map(finger_index):
    finger_map = {finger: [] for finger in fingers}
    for i, row in enumerate(finger_index):
        for j, finger in enumerate(row):
            finger_map[finger].append((i,j))
    return finger_map

def add_bigram_penalties(instance, frequencies):
    """
    Add penalties for bigrams to instance.

    Args:
        instance: Pyomo model, needs to be initialized via init_instance,
                  so all relevant variables exist.
    """
    objective = []
    # only bigrams with 2 different symbols, the others are irrelevant
    bigrams = set(filter(lambda x: len(x) == 2 and x[0] != x[1], frequencies))
    instance.bigrams = bigrams
    for finger in fingers:
        objective.extend(
                _add_bigram_penalty(instance, finger, finger, bigrams, same_finger_penalties, frequencies)
                )

    objective.extend(
            _add_bigram_penalty(instance, pinky_left, ring_left, bigrams, pinky_ring_penalties, frequencies)
            )
    objective.extend(
            _add_bigram_penalty(instance, pinky_right, ring_right, bigrams, pinky_ring_penalties, frequencies)
            )

    objective.extend(
            _add_bigram_penalty(instance, ring_left, middle_left, bigrams, ring_middle_penalties, frequencies)
            )
    objective.extend(
            _add_bigram_penalty(instance, ring_right, middle_right, bigrams, ring_middle_penalties, frequencies)
            )

    # returns part of objective
    return objective

def init_instance(instance):
    """
    Initialize basic constraints and variables for the layout optimization.
    Mutates instance.

    Args:
        instance: Pyomo model, needs to have certain parameters set.
    """
    assert hasattr(instance, 'frequencies')
    assert hasattr(instance, 'letters')
    assert hasattr(instance, 'effort')
    assert hasattr(instance, 'keys')
    assert hasattr(instance, 'finger_index')
    assert hasattr(instance, 'same_finger_penalties')
    assert hasattr(instance, 'pinky_ring_penalties')
    assert hasattr(instance, 'ring_middle_penalties')

    matrix_indices = [(i,j) for i in range(len(weights)) for j in range(len(weights[i]))]
    objective = [] # gradually append summands

    p.letters = Set(initialize=letters)
    p.matrix_indices = Set(initialize=matrix_indices)

    p.v = Var(p.letters, p.matrix_indices, domain=Binary)

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

def fix_layout(instance, layout):
    """
    Fix model parameters so that each symbol is assigned to its respective key
    given via the layout. As a consequence solving the instance will yield the
    objective value of the given layout.

    Args:
        instance: Pyomo model.
        layout: The keyboard layout
    """

    def fix_layout_rule(model, i, j):
        return model.v[layout[i][j], (i, j)] == 1
    instance.fix_layout = Constraint(instance.matrix_indices, rule=fix_layout_rule)

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
class ConfigReaderState(Enum):
    START = 0
    READ_EFFORT = 1
    READ_PENALTIES = 2
    READ_FINGERS = 3

EFFORT_LINE = 'effort:'
PENALTIES_LINE = 'penalties:'
FINGERS_LINE = 'fingers:'
N_EXPECTED_PENALTIES = 3

def read_configuration_file(f_name):
    """
    Read config parameters from a file.

    Args:
        f_name (str): File name of configuration.

    Returns:
        A tuple containing the effort table, bigram penalties and finger assignment.

    Example:
        TODO
    """

    state = ConfigReaderState.START
    effort = []
    penalties = []
    fingers = []
    n_rows = 0
    max_cols = 0
    n_penalties = 0
    n_finger_rows = 0

    # implement as state machine
    with open(f_name) as f:
        for i, line in enumerate(f):
            if line.find('#') > -1:
                line = line[line.find('#')] # ignore comments in file
            line = line.strip()

            if state == ConfigReaderState.START:
                if line == EFFORT_LINE:
                    state = ConfigReaderState.READ_EFFORT
                else:
                    raise Exception('Configuration file incorrect expecting {} at line {}, see specified format for details.'.format(EFFORT_LINE, i))

            elif state == ConfigReaderState.READ_EFFORT:
                if line == PENALTIES_LINE:
                    state = ConfigReaderState.READ_PENALTIES
                else:
                    row = list(map(float, line.split()))
                    max_cols = max(max_cols, len(row))
                    effort.append(row)
                    n_rows += 1

            elif state == ConfigReaderState.READ_PENALTIES:
                if line == FINGERS_LINE:
                    if n_penalties != N_EXPECTED_PENALTIES:
                        raise Exception('Configuration file incorrect in line {}. Expecting exactly {} different penalties.'.format(line, N_EXPECTED_PENALTIES).format(i, N_EXPECTED_PENALTIES))
                    state = ConfigReaderState.READ_FINGERS
                else:
                    if n_penalties >= N_EXPECTED_PENALTIES:
                        raise Exception('Configuration file incorrect in line {}. Expecting exactly {} different penalties.'.format(line, N_EXPECTED_PENALTIES).format(i, N_EXPECTED_PENALTIES))
                    penalty = list(map(float, line.split()))
                    if len(penalty) != n_rows:
                        raise Exception('Configuration file incorrect in line {}. Each penalty line needs to specify a penalty parameter for the number of rows of the layout!.'.format(i))
                    penalties.append(penalty)
                    n_penalties += 1

            elif state == ConfigReaderState.READ_FINGERS:
                if n_finger_rows == n_rows:
                    break
                else:
                    finger_row = list(map(int, line.split()))
                    if len(finger_row) != len(effort[n_finger_rows]):
                        raise Exception('Configuration file incorrect in line {}. The finger assignment table has to have the same layout as the effort table.'.format(i))
                    fingers.append(finger_row)
                    n_finger_rows += 1
            else:
                assert False, 'incorrect state, fatal error'

        if n_finger_rows != n_rows:
            raise Exception('Configuration file incorrect. The finger assignment table needs to have the same number of rows as the effort table.')

    return effort, penalties, fingers

def init_argument_parser():
    """
    Initialize an argparse.ArgumentParser instance for this program.

    Returns:
        An argument parser with the arguments
        congig_file
        layout_file
        solver
        bigram_threshold
        input_encoding
        symbols
        frequency_file
        text_file
    """

    # TODO review usage of mutually exclusive group for evaluation / optimization
    parser = argparse.ArgumentParser(description='Evaluate keyboard layouts and find optimal ones.')
    parser.add_argument('--config_file', type=str, default=None, help='filename to give configuration file')
    parser.add_argument('--layout_file', type=str, default=None, help='filename in which a layout to be evaluated is specified')
    parser.add_argument('--solver', type=str, default='glpk', help='Solver to user, for available solvers, see pyomo documentation')
    parser.add_argument('--bigram_threshold', type=float, default=1, help='TODO')
    parser.add_argument('--input_encoding', type=str, default='utf-8', help='Input encoding.')
    parser.add_argument('--symbols', type=str, default=letters, help='Set of symbols to place in the layout')

    corpus_choice = parser.add_mutually_exclusive_group()
    corpus_choice.add_argument('--frequency_file', type=str, default=None, help='File containing frequencies for letters and bigrams as JSON')
    corpus_choice.add_argument('--text_file', type=str, default=None, help='Text file for input corpus')

    return parser

def open_with_encoding(f_name, encoding=None):
    """
    Wrap the built-in open function, so we can use the same code for py2 & py3.
    Attention: in python 2 the encoding parameter is dropped and has no effect

    Args:
        f_name (str): File name
        encoding: Name of encoding to use, has no effect in py2

    Returns:
        A file
    """

    if sys.version_info < (3,):
        return open(f_name)
    return open(f_name, encoding=encoding)

def get_keys(effort):
    # TODO
    return []

def get_frequencies(args):
    """
    Get frequencies from input source specified via command line arguments.
    
    Args:
        args: should match the result of init_argument_parser().parse_arguments()
    """
    f_name = None
    if args.text_file is not None:
        f_name = args.text_file
    elif args.frequency_file is not None:
        f_name = args.frequency_file
    else:
        f_name = '/dev/stdin' # FIXME is this a good idea? non linux definetely not

    encoding = args.input_encoding
    with open(f_name, encoding=encoding) as f:
        # default mode is to optimize
        if args.frequency_file:
            frequencies = json.load(f)
        else:
            frequencies = analyze_frequencies(f)

        frequencies = optimize_frequencies(frequencies, args.bigram_threshold)

    return frequencies

def init_params(instance, args):
    """
    Set all relevant parameters as attributes of the optimization instance.
    Instance is mutated. Call this before calling init_instance!

    Args:
        instance: Pyomo model
        args: command line arguments
    """

    # TODO evaluate usage of pyomo Set / Param
    if args.config_file is not None:
        effort, penalties, fingers = read_configuration_file(args.config_file)
    else:
        effort = weights
        penalties = [same_finger_penalties, pinky_ring_penalties, ring_middle_penalties]
        fingers = finger_index

    instance.frequencies = get_frequencies(args)
    instance.symbols = args.symbols
    instance.effort = effort
    instance.same_finger_penalties = penalties[0]
    instance.pinky_ring_penalties = penalties[1]
    instance.ring_middle_penalties = penalties[2]
    instance.fingers = fingers

def main():
    """
    Execute optimization routine.
    """

    parser = init_argument_parser()
    args = parser.parse_args()

    instance = ConcreteModel()
    init_params(instance, args)
    init_instance(instance)
    if args.layout_file is not None:
        layout = read_layout(args.layout_file)
        # TODO verify if layout fits possible layout
        fix_layout(instance, layout)

    opt = SolverFactory(args.solver)
    results = opt.solve(instance)
    print(results)

    layout = [[None for j in range(len(weights[i]))] for i in range(len(weights))]

    for l in instance.letters:
        for i,j in instance.matrix_indices:
            if instance.v[l, (i,j)]:
                layout[i][j] = l

    for l in layout:
        print(l)

if __name__ == "__main__":
    main()
