def count(str):
    vowels = "AEIOUaeiou"
    vowel_count = 0
    cons_count = 0

    for ch in str:
        if ch.isalpha():
            if ch in vowels:
                vowel_count += 1
            else:
                cons_count += 1

    return vowel_count , cons_count

string = input("enter the string: ")
vowels, consonants = count(string)

print("Number of vowles : ", vowels)
print("number of constants: ", consonants)
