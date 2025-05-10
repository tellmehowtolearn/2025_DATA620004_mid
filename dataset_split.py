import os
import shutil
from sklearn.model_selection import train_test_split

def main():
    # Check if the original dataset directory exists
    if not os.path.exists("101_ObjectCategories"):
        print("Original dataset directory does not exist. Please check the path.")
        return

    # Create directories for train, validation, and test sets
    original_data_dir = "101_ObjectCategories"
    output_dir = "caltech101_split"
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in os.listdir(original_data_dir):
        category_path = os.path.join(original_data_dir, category)
        if os.path.isdir(category_path):
            images = [os.path.join(category_path, img)
                      for img in os.listdir(category_path)
                      if img.endswith(('.jpg', '.png'))]

            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(val_dir, category), exist_ok=True)
            os.makedirs(os.path.join(test_dir, category), exist_ok=True)

            for img in train_imgs:
                shutil.copy(img, os.path.join(train_dir, category))
            for img in val_imgs:
                shutil.copy(img, os.path.join(val_dir, category))
            for img in test_imgs:
                shutil.copy(img, os.path.join(test_dir, category))

    print("Dataset split completed.")

if __name__ == "__main__":
    main()
