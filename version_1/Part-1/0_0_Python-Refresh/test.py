l1=[x for x in range (1,11)]
#l1=list(set(l1))
print(l1)
l1.remove(7)
print(l1)
l1[0]=50
l1.pop(6)
print(l1)
new_list=[x**2  for x in l1 if x%2==0]
print(new_list)
new_list.sort(reverse=True)
print(new_list)