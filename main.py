import pandas, random


def probability(poet, sentence):
    global words, num_of_words, poets_probablity
    sentence = sentence.split()
    product = 1
    for i in range(len(sentence)):
        if sentence[i] in words: 
            product *= (words[sentence[i]][poet]/num_of_words[poet])
    return product * poets_probablity[poet]

def laplace_probability(poet, sentence):
    global words, num_of_words, poets_probablity
    sentence = sentence.split()
    product = 1
    for i in range(len(sentence)):
        if sentence[i] in words: 
            product *= ((words[sentence[i]][poet]+1)/(num_of_words[poet]+len(words.keys())))
    return product * poets_probablity[poet]

def accuracy(estimation, target_labels_list):
    correct_detected = 0
    for i in range(len(estimation)):
        if estimation[i] == target_labels_list[i]:
            correct_detected += 1
    return correct_detected/len(target_labels_list)*100

def precision(estimation, target_labels_list):
    correct_hafez, total_hafez = 0, 0
    for i in range(len(estimation)):
        if estimation[i] == "hafez":
            total_hafez += 1
            if target_labels_list[i] == 'hafez':
                correct_hafez += 1
    return correct_hafez/total_hafez*100

def recall(estimation, target_labels_list):
    correct_hafez, total_hafez = 0, 0
    for i in range(len(estimation)):
        if target_labels_list[i] == "hafez":
            total_hafez += 1
            if estimation[i] == 'hafez':
                correct_hafez += 1
    return correct_hafez/total_hafez*100




train_test = pandas.read_csv("data/train_test.csv")

training_size, total_size = 16712, len(train_test)

random_numbers, training_indexes, target_indexes = {}, [], []

while len(training_indexes) < training_size:
    num = random.randint(0, total_size-1)
    if num not in random_numbers:
        training_indexes.append(num)
        random_numbers[num] = 0

for i in range(total_size):
    if i not in random_numbers:
        target_indexes.append(i)


labels_list = list(train_test["label"])
text_list = list(train_test["text"])

training_labels_list, target_labels_list, training_text_list, target_text_list = [], [], [], []



for i in range(training_size):
    training_labels_list.append(labels_list[training_indexes[i]])
    training_text_list.append(text_list[training_indexes[i]])

for i in range(len(target_indexes)):
    target_labels_list.append(labels_list[target_indexes[i]])
    target_text_list.append(text_list[target_indexes[i]])


poets_statistics = {"hafez":0, "saadi":0}
for i in range(len(training_labels_list)):
    poets_statistics[training_labels_list[i]] += 1
poets_probablity = {"hafez":poets_statistics["hafez"]/len(training_labels_list), "saadi":poets_statistics["saadi"]/len(training_labels_list)}

words = {}
num_of_words = {"hafez":0, "saadi":0}

for i in range(len(training_text_list)):
    sentence = training_text_list[i].split()
    for j in range(len(sentence)):
        num_of_words[training_labels_list[i]] += 1
        if sentence[j] not in words:
            words[sentence[j]] = {"hafez":0, "saadi":0}
        words[sentence[j]][training_labels_list[i]] += 1


estimation = [None] * len(target_labels_list)



for i in range(len(target_text_list)):
    if probability("hafez", target_text_list[i]) >= probability("saadi", target_text_list[i]):
        estimation[i] = "hafez"
    else:
        estimation[i] = "saadi"


print("accuracy: ", accuracy(estimation, target_labels_list))
print("precision: ", precision(estimation, target_labels_list))
print("recall: ", recall(estimation, target_labels_list))


# Final Estimation

evaluate = pandas.read_csv("data/evaluate.csv")
evaluate_estimation = [None] * len(evaluate)

evaluate_id_list = list(evaluate["id"])
evaluate_text_list = list(evaluate["text"])



for i in range(len(evaluate_text_list)):
    if laplace_probability("hafez", evaluate_text_list[i]) >= laplace_probability("saadi", evaluate_text_list[i]):
        evaluate_estimation[i] = "hafez"
    else:
        evaluate_estimation[i] = "saadi"

pandas.DataFrame({"id":evaluate_id_list, "label":evaluate_estimation}).to_csv("data/output.csv", index=False)

# laplace Smoothing

laplace_estimation = [None] * len(target_text_list)



for i in range(len(target_text_list)):
    if laplace_probability("hafez", target_text_list[i]) >= laplace_probability("saadi", target_text_list[i]):
        laplace_estimation[i] = "hafez"
    else:
        laplace_estimation[i] = "saadi"

print("Laplace accuracy: ", accuracy(laplace_estimation, target_labels_list))
print("Laplace precision: ", precision(laplace_estimation, target_labels_list))
print("Laplace recall: ", recall(laplace_estimation, target_labels_list))
