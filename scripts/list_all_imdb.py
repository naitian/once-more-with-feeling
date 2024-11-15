"""
List all the IMDB IDs in the metadata file.
"""
from src.video.meta import VideoMetadata

if __name__ == "__main__":
    for movie in VideoMetadata.itermovies():
        print(movie.imdb)