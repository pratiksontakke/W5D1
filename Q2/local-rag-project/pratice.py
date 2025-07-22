# if False:
#     print("Hello World")
# elif False:
#     print("Good bye World")
# else:
#     print("Good-bye World")



# NESTING IF STATEMENT

# gender = "male"
# age = 21

# if gender == "male":
#     if age >= 21:
#         print("can marry")
#     else:
#         print("can't marry")
# else:
#     print("can't marry")

# BILLING SYSTEM

# total_bill = 1100
# if total_bill > 1000:
#     print("give 20% discount")
# elif total_bill > 500:
#     print("give 10% discount")
# else:
#     print("No discount")


# AND OR OPERATOR

# WHILE LOOP

# i = 1;
# while i<10:
#     print(i)
#     i += 1

# # NESTED WHILE LOOP

# i = 1;
# while i<10:
#     j = 1;
#     while j<10:
#         print(i,j)
#         j += 1
#     i += 1

# FOR LOOP

# for i in range(3,10,2):
#     print(i)


# for i in range(13,10,-2):
#     print(i)


# # Print the following pattern:
# # *
# # **
# # ***
# # ****
# # *****
# for i in range(1, 6):
#     print("*" * i)

# WRITING FUNCTIONS 

# def sum(x, y):
#     return x + y

# print(sum(10, 20))

# # PRINT EVEN AND ODD

# def even_odd(x):
#     if x % 2 == 0:
#         print("even")
#     else:
#         print("odd")

# print(even_odd(10)) # None = Null in java
# print(even_odd(11))

# name = "Manoj"
# age = 50
# address = "Rag Nagar"
# pincode = "123456"

# print("Name is",name,",age is",age,",address is",address,",pincode is",pincode)


# print("Name is " + name + ", age is " + str(age) + ", address is " + address + ", pincode is " + pincode)

# # f-string 
# print(f"Name is {name}, age is {age}, address is {address}, pincode is {pincode}")

# # format approch
# print("Name is {} age is {} address is {} pincode is {}".format(name, str(age), address, pincode))

# # lambda function

# add = lambda x, y: x + y

# even = lambda x: x%2==0

# print(even(9))



# a = 10
# b = 5

# max_num = lambda a, b: a if a>b else b

# print(max_num(10, 5))

# if None:
#     print("adf")
# else:
#     print("Faasdfdslse")


#Exception Handling

try:
    x = 10/2
except Exception as e:
    print(e)
finally:
    print("finally block")