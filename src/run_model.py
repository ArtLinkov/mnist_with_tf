import model

# print("------------------------------------------------------")
# print("Before training:")
# show_stats(confusion_matrix=True)
# print("------------------------------------------------------")

# train(epochs=10, batch_size=64, print_freq=1)
# show_stats(confusion_matrix=True)
# print("------------------------------------------------------")

# train(epochs=90, batch_size=64, print_freq=10)
# show_stats(confusion_matrix=True)
# print("------------------------------------------------------")

model.train(epochs=1000, batch_size=64, print_freq=100)
model.show_stats(all=True)
print("------------------------------------------------------")

model.session.close()
