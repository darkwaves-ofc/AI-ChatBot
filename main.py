from assistants import BasicAssistant

from training import trainer

trainer()
# assistant = BasicAssistant('intents.json')

# assistant.fit_model(epochs=50,data_set_path="./data")
# assistant.save_model(path="./models/")

# done = False

# while not done:
#     message = input("Enter a message: ")
#     if message == "STOP":
#         done = True
#     else:
#         print(assistant.process_input(message))
