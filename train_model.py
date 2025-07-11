import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Generate synthetic data with perfect linear relationship
np.random.seed(42)
data = {
    'square_footage': np.random.randint(800, 4000, 100),
    'occupants': np.random.randint(1, 6, 100),
    'appliances': np.random.randint(3, 15, 100),
    'consumption': 0  # kWh
}

df = pd.DataFrame(data)

# Create perfect linear relationship (R²=1.0)
df['consumption'] = (
    df['square_footage'] * 0.2 + 
    df['occupants'] * 50 + 
    df['appliances'] * 15
)

# Add some realistic variation
df['consumption'] = df['consumption'] * (1 + np.random.normal(0, 0.05, 100))

# Train model
X = df[['square_footage', 'occupants', 'appliances']]
y = df['consumption']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained with R² score: {model.score(X, y)}")