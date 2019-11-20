def solution1(list, num):
    for counter,i in enumerate(list[:-1]):
        temp_num = num - i
        for j in list[counter:]:
            if(temp_num==j):
                return i,j;
    return 0,0 

def solution2(list,num):
    list_dict = {i : list[i] for i in range(0, len(list))}
    for i in list:
        compliment = num - i
        if compliment in list_dict.values() and compliment!=i:
            return i,compliment
    return 0,0

numbers = [0, 2, 11, 19, 90]

print(solution1(numbers, 22))
print(solution1(numbers, 109))
print(solution2(numbers,109))
print(solution2(numbers,22))