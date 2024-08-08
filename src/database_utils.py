# *********************************************************************************************** #
#                                                                                                 #
#    database.py                                     For: None                                    #
#                                                                                    __           #
#    By: Myosin <hans95.bourgeois@gmail.com>         .--------..--.--..-----..-----.|__|.-----.   #
#                                                    |        ||  |  ||  _  ||__ --||  ||     |   #
#    Created: 2024/08/05 10:53:26 by Myosin          |__|__|__||___  ||_____||_____||__||__|__|   #
#    Updated: 2024/08/05 10:53:26 by Myosin                    |_____|                            #
#                                                                                                 #
# *********************************************************************************************** #

from pymysql import connect, cursors
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from typing import Dict, Any, List

@contextmanager
def create_connection(db_config : Dict[str, Any]) :
    """
    Create and manage a database connection.

    Args:
        db_config (Dict[str, Any]): Database configuration.

    Yields:
        Connection: Database connection object.
    """

    conn =  None
    try: 
        conn = connect(**db_config, cursorclass=DictCursor)
        yield conn
    finally: 
        if conn :
            conn.close()

# -----------------------------------------------------------------------------------------------------------------

def execute_request(conn : connect, table_name : str, cols : List[str], conditions : List[str]) -> Dict[str, Any] :
    """
    Execute a SQL SELECT query on the database.

    Args:
        conn (connect): Database connection object.
        table_name (str): Name of the table.
        cols (List[str]): List of columns to select.
        conditions (List[str]): List of WHERE conditions.

    Returns:
        Dict[str, Any]: Query result.
    """

    cols_placeholders = ", ".join(f"`{col}`" for col in cols)
    conds_placeholders = " AND ".join(f"({condition})" for condition in conditions)

    request_sql = f"""
        SELECT {cols_placeholders} 
        FROM {table_name}
        WHERE {conds_placeholders}
    """

    with conn.cursor() as cursor :
        print(f"SQL request send to MariaDB database :\n{cursor.mogrify(request_sql)}")
        cursor.execute(request_sql)
    result = cursor.fetchall()
    return result