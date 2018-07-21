import numpy as np
import pandas


# Class for Popularity based Recommender System model
class PopularityRecommender():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.movie_id = None
        self.popularity_recommendations = None

    # Create the popularity based recommender system model
    def create(self, train_data, user_id, movie_id):
        self.train_data = train_data
        self.user_id = user_id
        self.movie_id = movie_id

        # Get a count of user_ids for each unique movie as recommendation score
        train_data_grouped = train_data.groupby([self.movie_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={self.user_id: 'score'}, inplace=True)

        # Sort the movies based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.movie_id], ascending=[0, 1])

        # Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        # Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    # Use the popularity based recommender system model to
    # make recommendations
    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id

        # Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations


# Class for Item similarity based Recommender System model

class ItemSimilarityRecommender():

    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.movie_id = None
        self.cooccurence_matrix = None
        self.movies_dict = None
        self.rev_movies_dict = None
        self.item_similarity_recommendations = None

    # Get unique items (movies) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.movie_id].unique())

        return user_items

    # Get unique users for a given item (movie)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.movie_id] == item]
        item_users = set(item_data[self.user_id].unique())

        return item_users

    # Get unique items (movies) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.movie_id].unique())

        return all_items

    # Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_movies, all_movies):

        ####################################
        # Get users for all movies in user_movies.
        ####################################
        user_movies_users = []
        for i in range(0, len(user_movies)):
            user_movies_users.append(self.get_item_users(user_movies[i]))

        ###############################################
        # Initialize the item cooccurence matrix of size
        # len(user_movies) X len(movies)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_movies), len(all_movies))), float)

        #############################################################
        # Calculate similarity between user movies and all unique movies
        # in the training data
        #############################################################
        for i in range(0, len(all_movies)):
            # Calculate unique (users) of movie (item) i
            movies_i_data = self.train_data[self.train_data[self.movie_id] == all_movies[i]]
            users_i = set(movies_i_data[self.user_id].unique())

            for j in range(0, len(user_movies)):

                # Get unique (users) of movie (item) j
                users_j = user_movies_users[j]

                # Calculate intersection of watchers of movies i and j
                users_intersection = users_i.intersection(users_j)

                # Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    # Calculate union of watchers of movies i and j
                    users_union = users_i.union(users_j)

                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        return cooccurence_matrix

    # Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_movies, user_movies):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))

        # Calculate a weighted average of the scores in cooccurence matrix for all user movies.
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # Sort the indices of user_sim_scores based upon their value
        # Also maintain the corresponding score
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)

        # Create a dataframe from the following
        columns = ['user_id', 'movie', 'score', 'rank']
        # index = np.arange(1) # array of numbers for the number of samples
        df = pandas.DataFrame(columns=columns)

        # Fill the dataframe with top 10 item based recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_movies[sort_index[i][1]] not in user_movies and rank <= 10:
                df.loc[len(df)] = [user, all_movies[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        # Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no movies for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    # Create the item similarity based recommender system model
    def create(self, train_data, user_id, movie_id):
        self.train_data = train_data
        self.user_id = user_id
        self.movie_id = movie_id

    # Use the item similarity based recommender system model to
    # make recommendations
    def recommend(self, user):

        ########################################
        # Get all unique movies for this user
        ########################################
        user_movies = self.get_user_items(user)

        print("No. of unique movies for the user: %d" % len(user_movies))

        ######################################################
        # Get all unique items (movies) in the training data
        ######################################################
        all_movies = self.get_all_items_train_data()

        print("no. of unique movies in the training set: %d" % len(all_movies))

        ###############################################
        # Construct item cooccurence matrix of size
        # len(user_movies) X len(movies)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_movies, all_movies)

        #######################################################
        # Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_movies, user_movies)

        return df_recommendations

    # Get similar items to given items
    def get_similar_items(self, item_name):

        user_movies = item_name

        ######################################################
        # Get all unique items (movies) in the training data
        ######################################################
        all_movies = self.get_all_items_train_data()

        print("no. of unique movies in the training set: %d" % len(all_movies))

        ###############################################
        # Construct item cooccurence matrix of size
        # len(user_movies) X len(movies)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_movies, all_movies)

        #######################################################
        # Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_movies, user_movies)

        return df_recommendations


