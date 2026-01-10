def common_elements(l1, l2):
    count=0

    for element in l1:
        if element in l2:
            count += 1

    return count

l1 = list(map(int, input("enter elements of list 1: ").split()))
l2 = list(map(int, input("enter elements of list 2: ").split()))

counts = common_elements(l1, l2)

print("number of common elements: " , counts)
