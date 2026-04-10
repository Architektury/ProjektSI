# A program for reading and displaying handwritten words downloaded from graphic 
# files based on descriptions from text files
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pdb

num_of_authors = 8
max_num_of_words_per_author = 4

for author_no in range(num_of_authors):
    file_desc_name = "author" + str(author_no + 1) + "/word_places.txt"
    file_desc_ptr = open(file_desc_name, 'r')
    text = file_desc_ptr.read()
    lines = text.split('\n')
    number_of_lines = lines.__len__() - 1
    row_values = lines[0].split()
    number_of_values = row_values.__len__()

    num_of_words = 0
    image_file_name_prev = ""
    for i in range(number_of_lines):
        row_values = lines[i].split()
        
        if row_values[0] != '%':
            number_of_values = len(row_values)
            image_file_name = "author" + str(author_no + 1) + "\\" + row_values[0][1:-1]
            #pdb.set_trace()
            if image_file_name != image_file_name_prev:   
                image = mpimg.imread(str(image_file_name))
                image_file_name_prev = image_file_name
            word = row_values[1]
            row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                int(row_values[4]), int(row_values[5])
            subimage = image[row1:row2,column1:column2] 
            plt.title("Author "+str(author_no+1)+", image = "+row_values[0][1:-1]+", word = "+word)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.imshow(subimage)
            plt.show()

            num_of_words += 1

            #pdb.set_trace()
        if num_of_words >= max_num_of_words_per_author: break


    file_desc_ptr.close()