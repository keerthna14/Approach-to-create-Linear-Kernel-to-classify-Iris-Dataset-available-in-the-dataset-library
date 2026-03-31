import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# ================= DATA & MODEL =================
iris = load_iris()
X = iris.data[:, 2:4]  # petal length & width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Price mapping
price_map = {
    "setosa": 10,
    "versicolor": 20,
    "virginica": 30
}

# ================= FUNCTIONS =================

def predict():
    try:
        pl = float(entry_length.get())
        pw = float(entry_width.get())

        pred = model.predict([[pl, pw]])[0]
        species = iris.target_names[pred]
        price = price_map[species]

        result_label.config(
            text=f"🌸 Species: {species}\n💰 Price: ₹{price}",
            fg="#00ffcc"
        )

    except ValueError:
        messagebox.showerror("Error", "Enter valid numeric values!")

def show_graph():
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("Decision Boundary (SVM)")
    plt.show()

# ================= GUI =================

root = tk.Tk()
root.title("🌸 Iris Classifier Pro")
root.geometry("420x380")
root.configure(bg="#1e1e2f")

# Title
tk.Label(root, text="🌸 Iris Classifier Pro",
         font=("Arial", 18, "bold"),
         bg="#1e1e2f", fg="white").pack(pady=10)

# Input fields
tk.Label(root, text="Petal Length", bg="#1e1e2f", fg="white").pack()
entry_length = tk.Entry(root, justify="center")
entry_length.pack(pady=5)

tk.Label(root, text="Petal Width", bg="#1e1e2f", fg="white").pack()
entry_width = tk.Entry(root, justify="center")
entry_width.pack(pady=5)

# Buttons
tk.Button(root, text="Predict",
          command=predict,
          bg="#4CAF50", fg="white",
          width=15).pack(pady=10)

tk.Button(root, text="Show Decision Graph",
          command=show_graph,
          bg="#2196F3", fg="white",
          width=20).pack(pady=5)

# Accuracy label
tk.Label(root,
         text=f"Model Accuracy: {accuracy*100:.2f}%",
         bg="#1e1e2f", fg="#ffcc00",
         font=("Arial", 12, "bold")).pack(pady=10)

# Result
result_label = tk.Label(root,
                        text="",
                        font=("Arial", 14, "bold"),
                        bg="#1e1e2f", fg="white")
result_label.pack(pady=10)

# Footer
tk.Label(root,
         text="AI-Based Flower Pricing System",
         bg="#1e1e2f", fg="gray").pack(side="bottom", pady=5)

root.mainloop()
