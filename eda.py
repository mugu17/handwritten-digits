# Python code for EDA
import seaborn as sns

# Plot sample digits
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap='gray')
    plt.title(f"Label: {np.argmax(y_train[i])}")
    plt.axis('off')
plt.show()

# Class distribution
plt.figure(figsize=(8,4))
sns.countplot(x=np.argmax(y_train, axis=1))
plt.title('Class Distribution in Training Set')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.show()