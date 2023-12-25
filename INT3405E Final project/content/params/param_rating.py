list_train=["rating","gender","age"]
with open('./content/dataset/genres.txt', 'r') as f:
    genre_all = f.readlines()
    genre_all = [x.replace('\n','') for x in genre_all]
genre2idx = {genre:idx for idx, genre in enumerate(genre_all)}
