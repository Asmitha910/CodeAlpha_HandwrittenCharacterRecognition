import kagglehub

path = kagglehub.dataset_download(
    "sachinpatel21/az-handwritten-alphabets-in-csv-format"
)

print(path)