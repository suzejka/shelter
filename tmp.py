# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
from sodapy import Socrata

# %%
def send_message_to_telegram(message):
    chatId = '5303880405'
    botToken = '5660046213:AAHCSDYbdW7E5rc5MnoL1n8QCY-Qh8M1ZgI'
    url = f"https://api.telegram.org/bot{botToken}/sendMessage?chat_id={chatId}&text={message}"
    requests.get(url)

# %%
# client = Socrata("data.sonomacounty.ca.gov", None)

# results = client.get("924a-vesw", limit = 100000)

# df = pd.DataFrame.from_records(results)

# %%
df = pd.read_csv('dataset\Animal_Shelter_Intake_and_Outcome.csv', sep=';')
# df =  pandas.read_csv("https://data.sonomacounty.ca.gov/resource/924a-vesw.csv")

# %%
df

# %% [markdown]
# Name - Name of the animal. Animal names with an asterisk before them were given by shelter staff.<br>
# Type - Type of animal<br>
# Breed - Breed of animal<br>
# Color - Color of animal, Black, Chocolate, White….
# <br>Sex - Male, Female, Neutered Male, Spayed Female
# <br>Size - Large, medium, small, toy
# <br>Date Of Birth - Approximate date of birth.
# <br>Impound Number - Animal impound number
# <br>Kennel Number - Kennel number indicating its current location.
# <br>Animal ID - Unique ID
# <br>Intake Date	- Date animal was taken into the shelter
# <br>Outcome Date - Date animal left the shelter
# <br>Days in Shelter	- Number of days the animal was in the shelter
# <br>Intake Type	- Reason for intake
# <br>Intake Subtype - Sub reason for intake
# <br>Outcome Type - Reason for release from shelter
# <br>Outcome Subtype	- Sub reason for release from shelter
# <br>Intake Condition - Animals condition at intake
# <br>Outcome Condition - Animals condition at release from shelter
# <br>Intake Jurisdiction	- Jurisdiction responsible for animal intake
# <br>Outcome Jurisdiction - Area animal went to.
# <br>Outcome Zip Code - Zip code where animal went to.
# <br>Location - Latitude, Longitude coordinates for outcome jurisdiction
# <br>Count - Column for performing arithmetic and creating groups for views and visualizations

# %%
df.isnull().sum()

# %%
df = df.drop(['Impound Number', 'Animal ID'], axis=1)

# %%
df

# %%
unique_names = df['Name'].unique().tolist()
unique_names

# %%
df['Name'] = df['Name'].str.replace('*', '')
df['Name']

# %%
df.dtypes

# %%
df = df.drop('Location', axis=1)

# %%
unique_types = df['Type'].unique().tolist()
unique_types

# %%
otherAnimals = df[df["Type"] == "OTHER"]
otherAnimals

# %%
uniqueBreedsOfDifferentAnimals = otherAnimals['Breed'].value_counts().to_dict()
uniqueBreedsOfDifferentAnimals

# %% [markdown]
# Zbiór zwiera różne zwierzęta

# %%
topTenBreeds = df["Breed"].value_counts().head(10)
topTenBreeds = topTenBreeds.to_dict()
plt.figure(figsize=(20,10))
plt.bar(topTenBreeds.keys(), topTenBreeds.values())
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.bar(df['Color'], df['Days in Shelter'])
# plt.xlabel('Kolor psa')
# plt.ylabel('Dni w schronisku')
# plt.title('Zależność dni w schronisku względem koloru psa')
# plt.xticks(rotation=45)
# plt.show()


# %%
df = df.drop('Outcome Zip Code', axis=1)
df = df.drop('Intake Date', axis=1)
df = df.drop('Outcome Date', axis=1)
df = df.drop('Outcome Type', axis=1)
df = df.drop('Outcome Subtype', axis=1)
df = df.drop('Kennel Number', axis=1)
df = df.drop('Intake Jurisdiction', axis=1)
df = df.drop('Name', axis=1)

# %%
df = df.drop('Outcome Condition', axis=1)
df = df.drop('Outcome Jurisdiction', axis=1)
df = df.drop('Count', axis=1)

# %%
df.drop(df[df["Days in Shelter"] == "0"].index, inplace=True)
df

# %%
df["Sex"].unique()

# %%
df = df.replace("Neutered", "Male")
df = df.replace("Spayed", "Female")

# %%
df["Size"].unique()

# %%
df = df.replace("KITTN", "SMALL")
df = df.replace("PUPPY", "SMALL")
df = df.replace("TOY", "SMALL")
df = df.replace("MED", "MEDIUM")
df = df.replace("X-LRG", "LARGE")
df['Size'] = df['Size'].fillna("MEDIUM")

# %%
df["Size"].unique()

# %%
df

# %%
df.drop(df[df["Intake Condition"] == "UNKNOWN"].index, inplace=True)
df

# %%
sorted(df["Color"].unique())

# %%
len(df["Color"].unique())

# %%
def check_and_replace_color(colors):
    unique_colors = {}
    
    for color in colors:
        parts = color.split("/")
        if len(parts) == 2:
            reversed_color = "/".join(reversed(parts))
            if reversed_color in unique_colors:
                unique_colors[color] = unique_colors[reversed_color]
            else:
                unique_colors[color] = color
        else:
            unique_colors[color] = color
    
    updated_colors = [unique_colors[color] for color in colors]
    return updated_colors

# %%
def replace_same_colors(colors):
    updated_colors = []
    
    for color in colors:
        parts = color.split("/")
        if len(parts) == 2 and parts[0] == parts[1]:
            updated_colors.append(parts[0])
        else:
            updated_colors.append(color)
    
    return updated_colors

# %%
not_fixed_colors = df["Color"].copy()
df["Color"] = check_and_replace_color(df["Color"])
df["Color"] = replace_same_colors(df["Color"])

# %%
len(not_fixed_colors.unique())

# %%
len(df["Color"].unique())

# %%
unique_colors_before = not_fixed_colors.unique()
unique_colors_after = df["Color"].unique()

# %%
diff = len(unique_colors_before) - len(unique_colors_after)

tmp = list(unique_colors_after)
for i in range(diff):
    tmp.append(0)
    
unique_colors_after = np.asarray(tmp)

data = {
    "Before": unique_colors_before,
    "After": unique_colors_after
}

df_for_colors_check = pd.DataFrame(data)
df_for_colors_check
df_for_colors_check.to_csv("./test_data/test.csv")

# %%
df["Intake Condition"].unique()

# %%
df = df.replace("TREATABLE/MANAGEABLE", "TREATABLE")
df = df.replace("TREATABLE/REHAB", "TREATABLE")

# %%
# df[df["Days in Shelter"].str.contains(",")]

# %%
# df['Days in Shelter'] = df['Days in Shelter'].str.replace(',', '')

# %%
# df[df["Days in Shelter"].str.contains(",")]

# %%
import matplotlib.pyplot as plt
import seaborn as sns

df["Days in Shelter"] = df["Days in Shelter"].astype(float)
sns.boxplot(x=df["Days in Shelter"])
plt.show()


# %%
df[df["Days in Shelter"] > 50].count()

# %%
df.drop(df[df["Days in Shelter"] > 50].index, inplace=True)
df

# %%
sns.boxplot(x=df["Days in Shelter"])
plt.show()

# %%
df.isnull().sum()

# %%
from datetime import datetime

df.dropna(subset=['Date Of Birth'], inplace=True)
current_year = datetime.now().year
df["Date Of Birth"] = pd.to_datetime(df["Date Of Birth"])
df['Age'] = (current_year - df['Date Of Birth'].dt.year.astype(int))
df.drop(['Date Of Birth'], axis=1, inplace=True)
df

# %%
sns.boxplot(x=df["Age"])
plt.show()

# %%
df.drop(df[df["Age"] > 20].index, inplace=True)
df

# %%
df

# %%
df.drop(df[df["Sex"] == "Unknown"].index, inplace=True)
df

# %%
other_types = df[df["Type"] == "OTHER"]
other_types["Breed"].unique()

# %%
df.loc[(df["Type"] == "OTHER") & (df["Breed"] == "PALOMINO/MIX"), "Breed"] = "PALOMINO"

# %%
other_types = df[df["Type"] == "OTHER"]
other_types["Breed"].unique()

# %%
df.loc[(df["Type"] == "OTHER") & (df["Breed"].str.contains("MIX")), "Breed"] = "MIX"

# %%
other_types = df[df["Type"] == "OTHER"]
other_types["Breed"].unique()

# %%
df.loc[(df["Type"] == "OTHER") & (df["Breed"].str.contains("GOAT")), "Breed"] = "GOAT"

# %%
other_types = df[df["Type"] == "OTHER"]
other_types["Breed"].unique()

# %%
df.loc[(df["Type"] == "OTHER") & (df["Breed"] == "AMERICAN/REX"), "Breed"] = "MIX"

# %%
other_types = df[df["Type"] == "OTHER"]
other_types["Breed"].unique()

# %%
df.loc[(df["Type"] == "OTHER") & (df["Breed"].isin(["RABBIT SH", "RABBIT LH"])), "Breed"] = "RABBIT"

# %%
rabbit_breeds = ["LOP-AMER FUZZY",
                    "LOP-HOLLAND",
                    "RABBIT",
                    "DWARF HOTOT",
                    "MIX",
                    "REX",
                    "LOP-MINI",
                    "LOP-FRENCH",
                    "SILVER",
                    "HOTOT",
                    "ANGORA-ENGLISH",
                    "DUTCH",
                    "AMERICAN",
                    "CALIFORNIAN",
                    "LOP-ENGLISH",
                    "ENGLISH SPOT"]

df.loc[(df["Type"] == "OTHER") & (df["Breed"].isin(rabbit_breeds)), "Breed"] = "RABBIT"

roden_breeds = ["GUINEA PIG",
                "HAMSTER",
                "RAT",
                "MOUSE"]

df.loc[(df["Type"] == "OTHER") & (df["Breed"].isin(roden_breeds)), "Breed"] = "RODENT"

live_stock_breeds = ["GOAT",
                    "CHICKEN",
                    "SHEEP",
                    "BOER",
                    "BARRED ROCK"]

df.loc[(df["Type"] == "OTHER") & (df["Breed"].isin(live_stock_breeds)), "Breed"] = "LIVESTOCK"

bird_breeds = ["PARAKEET",
                "COCKATIEL",
                "CANARY",
                "DOVE"]

df.loc[(df["Type"] == "OTHER") & (df["Breed"].isin(bird_breeds)), "Breed"] = "BIRD"

horse_breeds = ["HORSE",
                "SHETLAND",
                "PALOMINO"]

df.loc[(df["Type"] == "OTHER") & (df["Breed"].isin(horse_breeds)), "Breed"] = "HORSE"

if "RACCOON" in df["Breed"].values:
    df.loc[df["Breed"] == "RACCOON", "Type"] = "RACCOON"

# %%
dog_types = df[df["Type"] == "DOG"]
sorted(dog_types["Breed"].unique())

# %%
len(dog_types["Breed"].unique())

# %%
df.loc[(df["Type"] == "DOG") & (df["Breed"].str.contains("/")), "Breed"] = "MIX"

# %%
dog_types = df[df["Type"] == "DOG"]
len(dog_types["Breed"].unique())

# %%
decision = df["Days in Shelter"]
attributes = df.drop("Days in Shelter", axis=1)

# %%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

attributes = pd.get_dummies(attributes).astype(int)

# %%
attributes

# %%
decision

# %%
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# decision = scaler.fit_transform(decision)

# %%
from sklearn.model_selection import train_test_split

# decision = df["Days in Shelter"]
# attributes = df.drop("Days in Shelter", axis=1)

X_train, X_test, y_train, y_test = train_test_split(attributes, decision, test_size=0.25)

# %%
def graph_for_model(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, c='b', marker='o', label='Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('MODEL - {model_name}'.format(model_name=model_name))
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.show()

# %% [markdown]
# Random forest

# %%
# send_message_to_telegram("Starting training")

# %%
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error

# model_LinearRegression = LinearRegression()
# model_LinearRegression.fit(X_train, y_train)
# y_pred_LinearRegression = model_LinearRegression.predict(X_test)
# mse_LinearRegression = mean_squared_error(y_test, y_pred_LinearRegression)
# r2_LinearRegression = r2_score(y_test, y_pred_LinearRegression)

# rounded_mse_LinearRegression = str(round(mse_LinearRegression, 3))
# rounded_r2_LinearRegression = str(round(r2_LinearRegression, 3))

# print("Mean Squared Error:", rounded_mse_LinearRegression)
# print("R^2 Score:", rounded_r2_LinearRegression)

# %%
# graph_for_model(y_test, y_pred_LinearRegression, "Linear Regression")

# %%
# from sklearn.linear_model import Ridge, Lasso, BayesianRidge

# model_Ridge = Ridge()
# model_Ridge.fit(X_train, y_train)
# y_pred_Ridge = model_Ridge.predict(X_test)
# mse_Ridge = mean_squared_error(y_test, y_pred_Ridge)
# r2_Ridge = r2_score(y_test, y_pred_Ridge)

# rounded_mse_Ridge = str(round(mse_Ridge, 3))
# rounded_r2_Ridge = str(round(r2_Ridge, 3))

# print("Mean Squared Error:", rounded_mse_Ridge)
# print("R^2 Score:", rounded_r2_Ridge)

# %%
# graph_for_model(y_test, y_pred_Ridge, "Ridge")

# %%
# model_Lasso = Lasso()
# model_Lasso.fit(X_train, y_train)
# y_pred_Lasso = model_Lasso.predict(X_test)
# mse_Lasso = mean_squared_error(y_test, y_pred_Lasso)
# r2_Lasso = r2_score(y_test, y_pred_Lasso)

# rounded_mse_Lasso = str(round(mse_Lasso, 3))
# rounded_r2_Lasso = str(round(r2_Lasso, 3))

# print("Mean Squared Error:", rounded_mse_Lasso)
# print("R^2 Score:", rounded_r2_Lasso)

# %%
# graph_for_model(y_test, y_pred_Ridge, "Ridge")

# %%
# model_BayesianRidge = BayesianRidge()
# model_BayesianRidge.fit(X_train, y_train)
# y_pred_BayesianRidge = model_BayesianRidge.predict(X_test)
# mse_BayesianRidge = mean_squared_error(y_test, y_pred_BayesianRidge)
# r2_BayesianRidge = r2_score(y_test, y_pred_BayesianRidge)

# rounded_mse_BayesianRidge = str(round(mse_BayesianRidge, 3))
# rounded_r2_BayesianRidge = str(round(r2_BayesianRidge, 3))

# print("Mean Squared Error:", rounded_mse_BayesianRidge)
# print("R^2 Score:", rounded_r2_BayesianRidge)

# %%
# graph_for_model(y_test, y_pred_BayesianRidge, "BayesianRidge")

# %%
# from sklearn.svm import SVR

# model_SVR = SVR(kernel="rbf")
# model_SVR.fit(X_train, y_train)
# y_pred_SVR = model_SVR.predict(X_test)
# mse_SVR = mean_squared_error(y_test, y_pred_SVR)
# r2_SVR = r2_score(y_test, y_pred_SVR)

# rounded_mse_SVR = str(round(mse_SVR, 3))
# rounded_r2_SVR = str(round(r2_SVR, 3))

# print("Mean Squared Error:", rounded_mse_SVR)
# print("R^2 Score:", rounded_r2_SVR)

# %%
# graph_for_model(y_test, y_pred_SVR, "SVR")

# %%
# from sklearn.tree import DecisionTreeRegressor

# model_DecisionTreeRegressor = DecisionTreeRegressor()
# model_DecisionTreeRegressor.fit(X_train, y_train)
# y_pred_DecisionTreeRegressor = model_DecisionTreeRegressor.predict(X_test)
# mse_DecisionTreeRegressor = mean_squared_error(y_test, y_pred_DecisionTreeRegressor)
# r2_DecisionTreeRegressor = r2_score(y_test, y_pred_DecisionTreeRegressor)

# rounded_mse_DecisionTreeRegressor = str(round(mse_DecisionTreeRegressor, 3))
# rounded_r2_DecisionTreeRegressor = str(round(r2_DecisionTreeRegressor, 3))

# print("Mean Squared Error:", rounded_mse_DecisionTreeRegressor)
# print("R^2 Score:", rounded_r2_DecisionTreeRegressor)

# %%
# graph_for_model(y_test, y_pred_DecisionTreeRegressor, "DecisionTreeRegressor")

# %% [markdown]
# NN

# %%
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3)
    # tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.3),
    # tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(1)
])

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'activation': ['relu', 'swish'],
    'dropout_rate': [0.2, 0.3],
    'kernel_regularizer': [l1(0.01), l2(0.01)]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(X_train, y_train)

print("Najlepsze hiperparametry:", grid_result.best_params_)

# %%
model.compile(optimizer='adam',
              metrics=['mean_squared_error'],
              loss='mean_absolute_error')

# %%
model.summary()

# %%
import matplotlib.pyplot as plt

epochs = 100

history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['mean_squared_error'], label='Training MSE')
plt.plot(epochs_range, history.history['val_mean_squared_error'], label='Validation MSE')
plt.legend(loc='upper right')
plt.title('Training and Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')

plt.show()


# %%
y_pred = model.predict(X_test)

y_pred

# %%
y_pred = y_pred.flatten()
y_pred

# %%
df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(df_pred.to_string(index=False))

# %%
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, c='b', marker='o', label='Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend(loc='upper left')
plt.grid(True)

plt.show()

# %%
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

rounded_mse = str(round(mse, 3))
rounded_r2 = str(round(r2, 3))

print("Mean Squared Error:", rounded_mse)
print("R^2 Score:", rounded_r2)

# %%
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def create_model(activation='relu', dropout_rate=0.3, kernel_regularizer=l2(0.01)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=(X_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(256, activation=activation, kernel_regularizer=kernel_regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(128, activation=activation, kernel_regularizer=kernel_regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_squared_error'])
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)

param_grid = {
    'activation': ['relu', 'swish'],
    'dropout_rate': [0.2, 0.3],
    'kernel_regularizer': [l1(0.01), l2(0.01)]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(X_train, y_train)

print("Najlepsze hiperparametry:", grid_result.best_params_)


# %%
#Remove all files from test_data folder
import os

folder_path = "./test_data"

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Usunięto plik: {filename}")
    except Exception as e:
        print(f"Błąd podczas usuwania pliku {filename}: {e}")

# %%



