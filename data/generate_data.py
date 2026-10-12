#from stow_recommender.simulate import generate_history
from stow_recommender.simulate import generate_history

#Data is generated randomly in the project for every train and test, 
#this file is used only to display it.
df = generate_history(20000, seed=42)
df = df.round({
    "fill_before": 3,
    "fill_after": 3,
    "min_clearance_cm": 1,
    "item_volume": 0,      # whole cm³
    "item_weight_kg": 2,
})
df.to_csv("data/stow_history.csv", index=False)   