def write_pickle(file_name, content):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as handle:
        pickle.dump(content, handle)

def read_pickle(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

