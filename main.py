import numpy as np # type: ignore
from src.models.logistic_regression import LogisticRegression
from src.optimizers.gradient_descent import GradientDescent
from src.optimizers.newton_raphson import NewtonRaphson

X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 5.0], [6.0, 7.0]])
y = np.array([0, 0, 0, 1, 1])

gd = GradientDescent(learning_rate=0.1, max_iter=1000)
model_gd = LogisticRegression(optimizer=gd)
model_gd.fit(X, y)

nr = NewtonRaphson(max_iter=20)
model_nr = LogisticRegression(optimizer=nr)
model_nr.fit(X, y)

X_test = np.array([[2.0, 2.0], [5.0, 6.0]])

print("Predicciones GD:", model_gd.predict(X_test))
print("Predicciones NR:", model_nr.predict(X_test))
print("Iteraciones GD:", len(gd.history))
print("Iteraciones NR:", len(nr.history))