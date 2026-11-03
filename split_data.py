from config import TEST_START


def split_train_test(data):
    train = data.loc[data.index < TEST_START].iloc[:-1].copy()
    test = data.loc[data.index >= TEST_START].copy()
    return train, test


def split_for_year(data, year):
    train = data.loc[data.index.year < year].iloc[:-1].copy()
    test = data.loc[data.index.year == year].copy()
    return train, test
