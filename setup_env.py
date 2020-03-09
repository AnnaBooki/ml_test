import os
import yaml
from sqlalchemy import create_engine
import logging

log = logging.getLogger(__name__)

def get_database():
    try:
        engine = get_connection_from_profile()
        print("DB Engine läuft")
    except IOError:
        print("DB Engine I/O Error")
        return None, 'fail'
    return engine

def get_connection_from_profile(config_file_name="default_profile.yaml"):

    with open(config_file_name, 'r') as f:
        vals = yaml.load(f, Loader=yaml.BaseLoader)

    if not ('PGHOST' in vals.keys() and
            'PGUSER' in vals.keys() and
            'PGPASSWORD' in vals.keys() and
            'PGDATABASE' in vals.keys() and
            'PGPORT' in vals.keys()):
        raise Exception('Credentials pruefen: ' + config_file_name)

    return get_engine(vals['PGDATABASE'], vals['PGUSER'],
                      vals['PGHOST'], vals['PGPORT'],
                      vals['PGPASSWORD'])


def get_engine(db, user, host, port, passwd):
    url = 'postgresql://{user}:{passwd}@{host}:{port}/{db}'.format(
        user=user, passwd=passwd, host=host, port=port, db=db)
    engine = create_engine(url, pool_size = 50)
    return engine

engine = get_connection_from_profile()
print(get_engine)