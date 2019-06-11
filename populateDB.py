from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import sqlite3 as lite
import sys
import errno
import glob

def create_connection(db_file):
    # Create a database connection to SQLite database
    try:
        conn = lite.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None;

def query(conn, query):
    try:
        c = conn.cursor()
        c.execute(query)

        if "SELECT" in query:
            results = c.fetchall()

            for row in results:
                print(row)
    except lite.Error as e:
        print(e)

def populate_database(conn, filepath):
    i = 0
    for name in filepath:
        i += 1

        try:
            img = load_img(name, target_size = (200,200))
            data = img_to_array(img)
            # default label is dog
            label = 0

            if "cat" in name:
                # if the picture is of a cat the label is reset
                label = 1

            insert_picture_query = "INSERT INTO pictures(label, picture_id) VALUES(" + str(label) + "," + str(i) + ");"
            query(conn, insert_picture_query)

            for row in data:
                for column in row:
                    insert_pixel_query = "INSERT INTO pixels(value_1, value_2, value_3, picture_id) VALUES(" + str(column[0]) + "," + str(column[1]) + "," + str(column[2]) + "," + str(i) + ");"
                    query(conn, insert_pixel_query)

            conn.commit()
        except IOError as e:
            if e.errno != errno.EISDIR:
                print("error")

conn = create_connection("data.db")

# Table should only need to be made once
create_pictures_table_query = "CREATE TABLE pictures(id INTEGER PRIMARY KEY AUTOINCREMENT,label BIT,picture_id SMALLINT);"
query(conn, create_pictures_table_query)
create_pixels_table_query = "CREATE TABLE pixels(id INTEGER PRIMARY KEY AUTOINCREMENT,value_1 TINYINT,value_2 TINYINT,value_3 TINYINT,picture_id SMALLINT FOREIGNKEY REFERENCES pictures(picture_id));"
query(conn, create_pixels_table_query)
# insert_query = "INSERT INTO pictures(label, picture_id) VALUES(1, 1)"
# query(conn, insert_query)
# for i in range(10):
#     insert_query = "INSERT INTO pixels(value_1, picture_id) VALUES(200, 1)"
#     query(conn, insert_query)
select_query = "SELECT count(1) FROM pictures"
query(conn, select_query)
select_query = "SELECT count(1) FROM pixels"
query(conn, select_query)
# select_query = "SELECT * FROM pixels WHERE picture_id = 1"
# query(conn, select_query)

# delete_query = "DELETE FROM pictures WHERE picture_id > 25000"
# query(conn, delete_query)
# delete_query = "DELETE FROM pixels WHERE picture_id > 25000"
# query(conn, delete_query)
populate_database(conn, glob.glob("../Data/DogsVsCats/train/*.jpg"))
