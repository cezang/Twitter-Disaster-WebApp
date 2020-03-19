import sqlite3
import os
conn = sqlite3.connect('tweet.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE Tweet_db'\
        '(Tweet TEXT, Fake INTEGER)')
example1 = 'Wtf happen in minnesota XD lol 3 milion people dies xD'
c.execute("INSERT INTO Tweet_db"\
        "(Tweet, Fake) VALUES"\
        "(?, ?)", (example1,1))
example2 = 'WHOA ambulance come and kill nigger new york man lol'
c.execute("INSERT INTO Tweet_db"\
        "(Tweet, Fake) VALUES"\
        "(?, ?)", (example2,1))
conn.commit()
conn.close()