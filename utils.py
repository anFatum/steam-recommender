def convert_data_to_rating(value):
    if value > 1000:
        return 7
    if value > 200:
        return 6
    if value > 20:
        return 5
    if value > 5:
        return 4
    if value > 1:
        return 3
    return 2

