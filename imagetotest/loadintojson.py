import Image
from pytesseract import image_to_string
import json
import glob

data = [] 
num_document = 10
label = ["vm", "vm", "vm", "vm", "vm", "vm", "opt", "opt", "opt", "opt"]
filenames = glob.glob("./Data/resume*.jpg")
print(filenames)

for i in range(num_document):
	
	text = image_to_string(Image.open(filenames[i]))
	data.append({ 'id' : i, 'text': text.encode('ascii', 'ignore'), 'label': label[i]}) 

with open('./Data/data.json', 'w') as outfile:  
    json.dump(data, outfile)


