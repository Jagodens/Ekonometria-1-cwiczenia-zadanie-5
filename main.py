import matplotlib.pyplot as plt
import numpy as np

# Dane z Zadania 5
x = np.array([400, 450, 550, 600, 700, 750])  # Wielkość produkcji (jednostki)
y = np.array([4000, 5000, 5400, 5900, 6400, 7000])  # Koszt całkowity ($)

# a. Wyznaczenie estymatorów parametrów równania regresji (b0 i b1) za pomocą MNK
b1, b0 = np.polyfit(x, y, 1)

print("--- Podpunkt a: Równanie regresji ---")
print(f"y_hat = {b0:.2f} + {b1:.2f} * x\n")

# b. Koszt zmienny na wyprodukowaną jednostkę
print("--- Podpunkt b: Koszt zmienny na jednostkę ---")
print(f"Koszt zmienny na 1 jednostkę wynosi: {b1:.2f} $\n")

# c. Współczynnik determinacji (R^2)
y_pred = b0 + b1 * x
ss_res = np.sum((y - y_pred) ** 2)  # Suma kwadratów reszt (SSE)
ss_tot = np.sum((y - np.mean(y)) ** 2)  # Całkowita suma kwadratów (SST)
r2 = 1 - (ss_res / ss_tot)

print("--- Podpunkt c: Współczynnik determinacji ---")
print(f"R^2 = {r2:.4f} (co stanowi ok. {r2 * 100:.2f}% wyjaśnionej wariancji)\n")

# d. Prognoza całkowitego kosztu dla 500 jednostek
x_new = 500
y_pred_500 = b0 + b1 * x_new

print("--- Podpunkt d: Prognoza dla 500 jednostek ---")
print(f"Przewidywany koszt całkowity dla 500 jednostek: {y_pred_500:.2f} $\n")

# Generowanie wykresu dopasowania regresji
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Dane empiryczne (obserwacje)', zorder=5)
plt.plot(x, y_pred, color='red', linewidth=2, label=f'Prosta regresji: y = {b0:.2f} + {b1:.2f}x')

# Oznaczenie punktu prognozy dla 500 jednostek
plt.scatter(x_new, y_pred_500, color='green', marker='*', s=150, label=f'Prognoza (500 szt. = {y_pred_500:.2f}$)', zorder=6)

plt.title('Szacowanie kosztów produkcji metodą najmniejszych kwadratów (MNK)')
plt.xlabel('Wielkość produkcji (jednostki)')
plt.ylabel('Koszt całkowity ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Zapisanie wykresu do pliku
plt.savefig('regresja_kosztow.png', dpi=300)
