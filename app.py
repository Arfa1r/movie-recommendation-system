from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import logging
import time



app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
base_dir = os.path.dirname(__file__)

ratings = pd.read_csv(os.path.join(base_dir, "datasets", "ratings.csv"))

movies = pd.read_csv(os.path.join(base_dir, "datasets", "movies.csv"))

data = pd.merge(ratings, movies, on="movieId")

movie_data = data.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)

movie_data = movie_data.apply(lambda col: col.fillna(col.mean()), axis=0)

similarity = cosine_similarity(movie_data.T)


def recommend(movie_name, top_n=4):

    if movie_name not in movie_data.columns:
        return None

    movie_index = movie_data.columns.get_loc(movie_name)
    similarity_scores = list(enumerate(similarity[movie_index]))



    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommendations = []

    for i in sorted_movies:
        if movie_data.columns[i[0]] != movie_name:
            recommendations.append(movie_data.columns[i[0]])
        if len(recommendations) == top_n:
            break

    return recommendations


# -------------------------
# flask part

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():

    start_time = time.time()  

    try:
        movie_name = request.form.get("movie")

        if not movie_name or movie_name.strip() == "":
            return render_template(
                "result.html",
                error="Please enter a movie name."
            )

        movie_name = movie_name.strip()

        if len(movie_name) > 100:
            return render_template(
                "result.html",
                error="Movie name is too long."
            )

        logging.info(f"User searched for: {movie_name}")

        movie_name_lower = movie_name.lower()
        columns_lower = movie_data.columns.str.lower()



        if movie_name_lower in columns_lower.values:
            real_movie_name = movie_data.columns[
                columns_lower == movie_name_lower
            ][0]


        else:
            matches = movie_data.columns[
                columns_lower.str.contains(movie_name_lower)
            ]


            if len(matches) == 0:
                return render_template(
                    "result.html",
                    movie=movie_name,
                    error="Movie not found."
                )

            real_movie_name = matches[0]

        rec = recommend(real_movie_name)


        if rec is None or len(rec) == 0:
            return render_template(
                "result.html",
                movie=real_movie_name,
                error="No similar movies found."
            )



        end_time = time.time()
        execution_time = round(end_time - start_time, 4)

        logging.info(f"Execution time: {execution_time} seconds")


        return render_template(
            "result.html",
            movie=real_movie_name,
            recommendations=rec,
            count=len(rec),
            execution_time=execution_time
        )


    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")

        return render_template(
            "result.html",
            error="Something went wrong. Please try again."
        )



if __name__ == "__main__":
    app.run(debug=True)