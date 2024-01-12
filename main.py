import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Assuming 'data' is your DataFrame
# Replace it with your actual dataset or data loading code
data = pd.read_csv('data_uas.csv', sep=',', header=0, engine='python', encoding='utf-8')

# Descriptive statistics
descriptive_stats = data.describe()

# Display the results
print(descriptive_stats)

# Interpretasi
print("\nInterpretasi Statistik Deskriptif:")
print("===================================")

# Hari (Day)
print("\nHari (Day):")
print("Data ini mencakup 548 observasi dengan nilai rata-rata hari sebesar {:.2f}.".format(descriptive_stats.loc['mean', 'Day']))

# Interaksi (Interaction)
print("\nInteraksi (Interaction):")
print("Variabel interaksi memiliki nilai rata-rata {:.2f} dengan deviasi standar {:.2f}.".format(descriptive_stats.loc['mean', 'Interaction'], descriptive_stats.loc['std', 'Interaction']))

# Tempat Tinggal (Residences)
print("\nTempat Tinggal (Residences):")
print("Variabel Tempat Tinggal memiliki total 704 dengan nilai rata-rata {:.2f}.".format(descriptive_stats.loc['mean', 'Residences']))

# Pengetahuan (Knowledge)
print("\nPengetahuan (Knowledge):")
print("Variabel Pengetahuan memiliki nilai rata-rata {:.2f} dengan deviasi standar {:.2f}.".format(descriptive_stats.loc['mean', 'Knowledge'], descriptive_stats.loc['std', 'Knowledge']))

# Curah Hujan (Rainfall)
print("\nCurah Hujan (Rainfall):")
print("Variabel Curah Hujan memiliki nilai minimum {:.2f} dan nilai maksimum {:.2f}.".format(descriptive_stats.loc['min', 'Rainfall'], descriptive_stats.loc['max', 'Rainfall']))

# Kelembaban (%) (Humidity (%))
print("\nKelembaban (%):")
print("Variabel Kelembaban memiliki nilai rata-rata {:.2f} dengan deviasi standar {:.2f}.".format(descriptive_stats.loc['mean', 'Humidity (%)'], descriptive_stats.loc['std', 'Humidity (%)']))

# Suhu (Temperature)
print("\nSuhu (Temperature):")
print("Variabel Suhu memiliki nilai rata-rata {:.2f} dengan deviasi standar {:.2f}.".format(descriptive_stats.loc['mean', 'Temperature'], descriptive_stats.loc['std', 'Temperature']))

# Daya (Power)
print("\nDaya (Power):")
print("Variabel Daya memiliki nilai rata-rata {:.2f} dan nilai maksimum {:.2f}.".format(descriptive_stats.loc['mean', 'Power'], descriptive_stats.loc['max', 'Power']))

print("\n")
# Check validity and reliability (example: using correlation matrix)
correlation_matrix = data.corr()

# Display the correlation matrix
print("\nMatrix Korelasi:")
print(correlation_matrix)

# Assuming 'data' is your DataFrame
# Replace it with your actual dataset or data loading code

# Descriptive statistics for specific columns
power_stats = data['Power'].describe()
humidity_stats = data['Humidity (%)'].describe()
rainfall_stats = data['Rainfall'].mean()

# Display the results
print("\nPower Stats:", power_stats)
print("Humidity Stats:", humidity_stats)
print("Rainfall Mean:", rainfall_stats)

# Plot histograms
data[['Power', 'Humidity (%)', 'Rainfall']].hist(bins=20, figsize=(10, 6))
plt.show()
