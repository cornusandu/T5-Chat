import sqlite3
import pandas

while True:
    try:
        query = input("Query: ")
        con = sqlite3.connect("db.db")
        df = pandas.read_sql_query(query, con)
        print(df.to_string())
        con.commit()
        con.close()
    except Exception as e:
        print(e)
 