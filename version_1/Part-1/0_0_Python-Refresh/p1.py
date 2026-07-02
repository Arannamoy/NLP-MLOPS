def average_marks(list_of_list1):
    final_list=[]
    for list in list_of_list1:
        sum=0
        temp_list=[]
        temp_list.append(list[0])
        for i in range(1,len(list)):
            sum+=list[i]

        sum=sum/(len(list)-1)
        temp_list.append(sum)
        final_list.append(temp_list)

    return final_list


print(average_marks([["Ali",80,90,70],["Sara",85,95,100],["Zain",75,80]])  )  