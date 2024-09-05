from __future__ import print_function
from datetime import date, datetime, timedelta
import mysql.connector
# ___________________________________
import Insert_data
# __________________________
config = {
        'user': 'root',
        'password': 'Ledzeppelin_7777',
        'host': '127.0.0.1',
        'database': 'employees',
    }

cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()
tomorrow = datetime.now().date() + timedelta(days=1)


Insert_data.departments()
Insert_data.employees()
Insert_data.dept_emp()
Insert_data.dept_manager()
Insert_data.titles()
Insert_data.salaries()

cursor.close()
cnx.close()

# ____________________________________________
# Пример добавления одиночных записей
# __________ Insert new employee
# #  ____ data
# data_employee = ('Geert', 'Vanderkelen', tomorrow, 'M', date(1977, 6, 14))
# #  ____ command
# add_employee = ("INSERT INTO employees "
#                "(first_name, last_name, hire_date, gender, birth_date) "
#                "VALUES (%s, %s, %s, %s, %s)")

# cursor.execute(add_employee, data_employee)
# emp_no = cursor.lastrowid  # Return value AUTO_INCREMENT column

# __________ Insert salary information (uses extended Python format codes)
#  ____ data
# data_salary = {
#   'emp_no': emp_no,
#   'salary': 50000,
#   'from_date': tomorrow,
#   'to_date': date(9999, 1, 1),
# }
# #  ____ command
# add_salary = ("INSERT INTO salaries "
#               "(emp_no, salary, from_date, to_date) "
#               "VALUES (%(emp_no)s, %(salary)s, %(from_date)s, %(to_date)s)")
# cursor.execute(add_salary, data_salary)

# # Data is committed to the database
# cnx.commit()

# Начинение БД данными из 
# https://github.com/datacharmer/test_db/blob/master/employees_partitioned_5.1.sql

# cursor.close()
# cnx.close()