
# 导入SQLite驱动:
import sqlite3
# 连接到SQLite数据库
# 数据库文件是test.db
# 如果文件不存在，会自动在当前目录创建:
import os



class sqlite3_client:
    def __init__(self, file_path='test.db'):
        self.file_path = file_path
        if os.path.exists(file_path):
            os.remove('test.db')
        conn = sqlite3.connect('test.db')
        # 创建一个Cursor:
        cursor = conn.cursor()
        # 执行一条SQL语句，创建user表:
        cursor.execute(
            'CREATE TABLE employee (id INT NOT NULL PRIMARY KEY, name VARCHAR(50), age INT, department VARCHAR(50));')
        # <sqlite3.Cursor object at 0x10f8aa260>
        cursor.execute(
            "INSERT INTO employee (id, name, age, department) VALUES (1, 'Employee A', 25, 'Marketing'), (2, 'Employee B', 27, 'Marketing'), (3, 'Employee C', 30, 'Marketing'), (4, 'Employee D', 28, 'Marketing'), (5, 'Employee E', 32, 'Marketing'), (6, 'Employee F', 23, 'Technology'), (7, 'Employee G', 24, 'Technology'), (8, 'Employee H', 26, 'Technology'), (9, 'Employee I', 29, 'Technology'), (10, 'Employee J', 31, 'Technology'), (11, 'Employee K', 33, 'Technology'), (12, 'Employee L', 22, 'Public Relations'), (13, 'Employee M', 24, 'Public Relations'), (14, 'Employee N', 27, 'Public Relations');")
        cursor.execute("select * from employee")
        # 关闭Cursor:
        cursor.close()
        # 提交事务:
        conn.commit()
        # 关闭Connection:
        conn.close()

def execute_query(self, query):
    conn = sqlite3.connect(self.file_path)
    # 创建一个Cursor:
    cursor = conn.cursor()
    cursor.execute(query)
    values = cursor.fetchall()
    return values
