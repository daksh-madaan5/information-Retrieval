from preprocessing import tokenize


def build_edit_distance_matrix(source, target):
    """Build the full Levenshtein dynamic-programming matrix."""
    row_count = len(source) + 1
    column_count = len(target) + 1
    matrix = []

    for _ in range(row_count):
        matrix.append([0] * column_count)

    for row in range(row_count):
        matrix[row][0] = row

    for column in range(column_count):
        matrix[0][column] = column

    for row in range(1, row_count):
        for column in range(1, column_count):
            if source[row - 1] == target[column - 1]:
                replacement_cost = 0
            else:
                replacement_cost = 1

            delete_cost = matrix[row - 1][column] + 1
            insert_cost = matrix[row][column - 1] + 1
            replace_cost = (
                matrix[row - 1][column - 1] + replacement_cost
            )

            matrix[row][column] = min(
                delete_cost,
                insert_cost,
                replace_cost,
            )

    return matrix


def edit_distance(source, target):
    """Return unit-cost Levenshtein distance for the supplied strings."""
    matrix = build_edit_distance_matrix(source, target)
    return matrix[len(source)][len(target)]


def normalize_comparison_term(term):
    """Apply the project's normalization to exactly one comparison term."""
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError("Edit-distance comparison requires one term per input.")

    return tokens[0]


def build_weighted_edit_distance_matrix(
    source,
    target,
    replacement_costs=None,
    insertion_cost=1.0,
    deletion_cost=1.0,
):
    """Build a weighted Levenshtein dynamic-programming matrix."""
    if replacement_costs is None:
        replacement_costs = {}

    row_count = len(source) + 1
    column_count = len(target) + 1
    matrix = []

    for _ in range(row_count):
        matrix.append([0.0] * column_count)

    for row in range(row_count):
        matrix[row][0] = row * deletion_cost

    for column in range(column_count):
        matrix[0][column] = column * insertion_cost

    for row in range(1, row_count):
        for column in range(1, column_count):
            source_character = source[row - 1]
            target_character = target[column - 1]

            if source_character == target_character:
                replacement_cost = 0.0
            else:
                replacement_cost = replacement_costs.get(
                    (source_character, target_character),
                    1.0,
                )

            delete_total = matrix[row - 1][column] + deletion_cost
            insert_total = matrix[row][column - 1] + insertion_cost
            replace_total = (
                matrix[row - 1][column - 1] + replacement_cost
            )

            matrix[row][column] = min(
                delete_total,
                insert_total,
                replace_total,
            )

    return matrix


def weighted_edit_distance(
    source,
    target,
    replacement_costs=None,
    insertion_cost=1.0,
    deletion_cost=1.0,
):
    """Return weighted edit distance using caller-supplied costs."""
    matrix = build_weighted_edit_distance_matrix(
        source,
        target,
        replacement_costs=replacement_costs,
        insertion_cost=insertion_cost,
        deletion_cost=deletion_cost,
    )
    return matrix[len(source)][len(target)]
