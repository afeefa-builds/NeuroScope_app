import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv('human_cognitive_performance.csv')
df.drop(columns='User_ID', inplace=True)

# --- Encode categoricals and save mappings (category -> code) ---
df['Gender'] = df['Gender'].astype('category')
gender_mapping = dict(zip(df['Gender'].cat.categories, range(len(df['Gender'].cat.categories))))
df['Gender'] = df['Gender'].cat.codes

df['Diet_Type'] = df['Diet_Type'].astype('category')
diet_mapping = dict(zip(df['Diet_Type'].cat.categories, range(len(df['Diet_Type'].cat.categories))))
df['Diet_Type'] = df['Diet_Type'].cat.codes

df['Exercise_Frequency'] = df['Exercise_Frequency'].astype('category')
exercise_mapping = dict(zip(df['Exercise_Frequency'].cat.categories, range(len(df['Exercise_Frequency'].cat.categories))))
df['Exercise_Frequency'] = df['Exercise_Frequency'].cat.codes

X = df[['Gender', 'Sleep_Duration', 'Stress_Level', 'Diet_Type', 'Daily_Screen_Time',
        'Exercise_Frequency', 'Caffeine_Intake', 'Reaction_Time', 'Memory_Test_Score']]
y = df['Cognitive_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=300,
    random_state=33
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("Train R²:", r2_score(y_train, model.predict(X_train)))

# --- Save everything the app needs ---
joblib.dump(model, 'model2.pkl', compress = 4)
joblib.dump(gender_mapping, 'gender_mapping2.pkl')
joblib.dump(diet_mapping, 'diet_mapping2.pkl')
joblib.dump(exercise_mapping, 'exercise_mapping2.pkl')

print("Model and mappings saved successfully.")