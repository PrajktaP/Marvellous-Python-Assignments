from functools import reduce

search = lambda num : num >= 70 and num <= 90
increase_by_ten = lambda num : num + 10
product = lambda num1, num2 : num1 * num2 

def main():
    # test with hardcoded input
    # Data = [4, 34, 36, 76, 68, 24, 89, 23, 86, 90, 45, 70]

    # accept data from user
    size = int(input("Enter number of elements: "))
    Data = []
    for i in range(size):
        Data.append(int(input("Enter a number: ")))

    print("Input list =", Data)

    filtered_data = list(filter(search, Data))
    print("List after filter =", filtered_data)

    mapped_data = list(map(increase_by_ten, filtered_data))
    print("List after map =", mapped_data)

    reduced_value = reduce(product, mapped_data)
    print("Output of reduce =", reduced_value)

if __name__ == "__main__":
    main()



"""
3.Write a program which contains filter(), map() and reduce() in it. Python application which
contains one list of numbers. List contains the numbers which are accepted from user. Filter
should filter out all such numbers which greater than or equal to 70 and less than or equal to
90. Map function will increase each number by 10. Reduce will return product of all that
numbers.
Input List = [4, 34, 36, 76, 68, 24, 89, 23, 86, 90, 45, 70]
List after filter = [76, 89, 86, 90, 70]
List after map = [86, 99, 96, 100, 80]
Output of reduce = 6538752000
"""