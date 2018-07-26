import urllib
import zipfile


def download_movie_lens_dataset():
    """Download Data files from http://files.grouplens.org/ """
    output_filepath = '../data/raw/external/'
    zip_filename = 'ml-latest.zip'
    url = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"
    urllib.request.urlretrieve(url, output_filepath + zip_filename)
    directory_to_extract_to = '../data/external/movielens/'
    unzip_data(output_filepath + zip_filename, directory_to_extract_to)


def unzip_data(file, output_directory):
    """Extract files from the downloaded zip folder"""
    zip_ref = zipfile.ZipFile(file, 'r')
    output_directory = '../data/external/movielens/'
    zip_ref.extractall(output_directory)
    zip_ref.close()
