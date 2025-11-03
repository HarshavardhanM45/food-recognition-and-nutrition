# -----------------------------
# Indian Food Classification & Nutrition
# -----------------------------
import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data_path = "Indian Food Images"  # Replace with full path if needed

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_path,
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_path,
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

print("Classes detected:", train_generator.class_indices)
print("Training images:", train_generator.samples)
print("Validation images:", val_generator.samples)

# -----------------------------
# Step 2: Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Step 3: Train CNN
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15  # Increase for better accuracy
)

# -----------------------------
# Step 4: Load Nutrition CSV
# -----------------------------
nutrition_df = pd.read_csv("nutrition.csv")
print("Nutrition CSV loaded:", nutrition_df.head())

# -----------------------------
# Step 5: Save Model & Other Files in Project Folder
# -----------------------------
project_folder = os.getcwd()  # Get current folder (same as main.py)

# Save CNN model
model_path = os.path.join(project_folder, "food_cnn_model.h5")
model.save(model_path)

# Save class indices
class_indices_path = os.path.join(project_folder, "class_indices.pkl")
with open(class_indices_path, "wb") as f:
    pickle.dump(train_generator.class_indices, f)

# Save nutrition dataframe as pickle
nutrition_path = os.path.join(project_folder, "nutrition.pkl")
nutrition_df.to_pickle(nutrition_path)

print("Model, class indices, and nutrition saved in project folder!")

# -----------------------------
# Step 6: Predict Food Image
# -----------------------------
def predict_food(image_path):
    img = load_img(image_path, target_size=(128,128))
    x = img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    pred_class = list(train_generator.class_indices.keys())[np.argmax(pred)]
    return pred_class

# -----------------------------
# Step 7: Get Nutrition Info
# -----------------------------
def get_nutrition(food_name):
    info = nutrition_df[nutrition_df['Food Name'] == food_name]
    if not info.empty:
        return info
    else:
        return pd.DataFrame({"Food Name":[food_name], "Calories":[0], "Protein":[0], "Fat":[0], "Carbs":[0]})

# -----------------------------
# Step 8: Test Single Image Prediction
# -----------------------------
test_folder = "Indian Food Images/aloo_gobi"
test_image_file = os.listdir(test_folder)[0]
test_image_path = os.path.join(test_folder, test_image_file)
print("Using test image:", test_image_path)

predicted_food = predict_food(test_image_path)
print("Predicted Food:", predicted_food)

nutrition_info = get_nutrition(predicted_food)
print("Calories & Nutrients:")
print(nutrition_info)

# -----------------------------
# Step 9: Predict Multiple Images (Meal)
# -----------------------------
def predict_meal(folder_path):
    total_calories = 0
    total_protein = 0
    total_fat = 0
    total_carbs = 0
    
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        food = predict_food(img_path)
        nut = get_nutrition(food)
        total_calories += int(nut['Calories'].values[0])
        total_protein += int(nut['Protein'].values[0])
        total_fat += int(nut['Fat'].values[0])
        total_carbs += int(nut['Carbs'].values[0])
        
    print(f"\nTotal Nutrition for all images in {folder_path}:")
    print(f"Calories: {total_calories}, Protein: {total_protein}, Fat: {total_fat}, Carbs: {total_carbs}")

# Example usage:
predict_meal(test_folder)
